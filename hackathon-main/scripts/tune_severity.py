#!/usr/bin/env python3
"""Leakage-safe offline hyperparameter search for the severity SVC.

Protocol:
1. Fit StandardScaler on train.csv only.
2. Use val.csv only as the semantic reference bank to construct severity
   prototypes and pseudo-severity targets for train.csv.
3. Split pseudo-labeled train rows by engine_id with GroupKFold.
4. Fit each candidate SVC only on the training fold and score on its holdout.
5. Never load test.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
UNKNOWN_CONFIDENCE = 0.45
SEVERITY_MIN_CONFIDENCE = 0.30
FAULTS = ("zakoksowany", "lejacy", "pompa", "iglica")
SEVERITIES = ("male", "srednie", "duze")
SEV_TO_INT = {name: i for i, name in enumerate(SEVERITIES)}

C_VALUES = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
GAMMA_VALUES = (0.003, 0.005, 0.0075, 0.01, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, 0.05)
WEIGHTS = ("balanced", None)


def d1(df: pd.DataFrame) -> np.ndarray:
    cols = [f"mV_{i}" for i in range(21)]
    raw = df[cols].to_numpy(dtype=np.float32)
    return np.diff(raw, axis=1).astype(np.float32, copy=False)


def pseudo_severity_dataset(train: pd.DataFrame, val: pd.DataFrame):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(d1(train)).astype(np.float32)
    X_val = scaler.transform(d1(val)).astype(np.float32)

    ref_labels = val["label"].astype(str).to_numpy()
    label_prototypes = {}
    for label in val["label"].astype(str).unique():
        mask = ref_labels == label
        if mask.any():
            label_prototypes[label] = np.median(X_val[mask], axis=0)

    class_names = [x for x in ("ok", *FAULTS, "unknown") if x in label_prototypes]
    if not class_names:
        raise RuntimeError("No supported label prototypes found in val.csv")
    proto_matrix = np.vstack([label_prototypes[x] for x in class_names])
    distances = np.stack(
        [np.linalg.norm(X_train - proto, axis=1) for proto in proto_matrix], axis=1
    )
    pseudo_label = np.asarray(class_names, dtype=object)[distances.argmin(axis=1)]
    scale = np.median(distances, axis=1, keepdims=True) + 1e-6
    logits = -distances / scale
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -30, 0))
    weights /= weights.sum(axis=1, keepdims=True)
    pseudo_confidence = weights.max(axis=1)

    sev_prototypes = {}
    for fault in FAULTS:
        fault_mask = ref_labels == fault
        if not fault_mask.any():
            continue
        ref_sev = val.loc[fault_mask, "severity"].astype(str).to_numpy()
        fault_X = X_val[fault_mask]
        for sev in SEVERITIES:
            sev_mask = ref_sev == sev
            if sev_mask.any():
                sev_prototypes[(fault, sev)] = np.median(fault_X[sev_mask], axis=0)

    use = np.isin(pseudo_label, FAULTS) & (pseudo_confidence >= SEVERITY_MIN_CONFIDENCE)
    if use.sum() < 30:
        use = np.isin(pseudo_label, FAULTS)

    rows = []
    valid_indices = []
    for idx in np.flatnonzero(use):
        row_x = X_train[idx]
        fault = pseudo_label[idx]
        candidates = [
            (sev, proto)
            for (candidate_fault, sev), proto in sev_prototypes.items()
            if candidate_fault == fault
        ]
        if len(candidates) < 2:
            continue
        distances = [np.linalg.norm(row_x - proto) for _, proto in candidates]
        severity = candidates[int(np.argmin(distances))][0]
        rows.append((row_x, SEV_TO_INT[severity], float(pseudo_confidence[idx])))
        valid_indices.append(idx)

    if len(rows) < 30:
        raise RuntimeError("Could not create enough pseudo-severity training rows")

    X = np.vstack([row[0] for row in rows]).astype(np.float32)
    y = np.asarray([row[1] for row in rows], dtype=np.int32)
    sample_weight = np.clip(
        0.5 + 0.75 * np.asarray([row[2] for row in rows], dtype=np.float32),
        0.5,
        1.0,
    )
    groups = train.iloc[valid_indices]["engine_id"].astype(str).to_numpy()
    return X, y, groups, sample_weight


def score_candidate(X, y, groups, sample_weight, C, gamma, weight):
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two unique engines for GroupKFold")
    splitter = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, ho in splitter.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_ho = scaler.transform(X[ho])
        model = SVC(
            kernel="rbf",
            C=float(C),
            gamma=float(gamma),
            class_weight=weight,
            cache_size=512,
        )
        model.fit(X_tr, y[tr], sample_weight=sample_weight[tr])
        scores.append(float((model.predict(X_ho) == y[ho]).mean()))
    return float(np.mean(scores)), float(np.std(scores)), float(np.min(scores))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    if "label" in train.columns:
        raise AssertionError("train.csv unexpectedly contains label")
    if not {"label", "severity", "engine_id"}.issubset(val.columns):
        raise AssertionError("val.csv must contain label, severity and engine_id")

    X, y, groups, sample_weight = pseudo_severity_dataset(train, val)
    print(f"Pseudo-severity rows: {len(y)}")
    print(f"Engines: {len(np.unique(groups))}")
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    results = []
    for weight in WEIGHTS:
        for C in C_VALUES:
            for gamma in GAMMA_VALUES:
                mean, std, minimum = score_candidate(
                    X, y, groups, sample_weight, C, gamma, weight
                )
                results.append(
                    {
                        "model": "svc_rbf",
                        "representation": "d1",
                        "C": C,
                        "gamma": gamma,
                        "class_weight": weight if weight is not None else "none",
                        "accuracy_mean": mean,
                        "accuracy_std": std,
                        "accuracy_min": minimum,
                    }
                )

    result_df = pd.DataFrame(results).sort_values(
        ["accuracy_mean", "accuracy_min", "accuracy_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORTS / "severity_tuning.csv", index=False)

    best = result_df.iloc[0].to_dict()
    params = {
        "C": float(best["C"]),
        "gamma": float(best["gamma"]),
        "class_weight": None if best["class_weight"] == "none" else "balanced",
        "selection_metric": "grouped_cv_accuracy",
        "accuracy_mean": float(best["accuracy_mean"]),
        "accuracy_min": float(best["accuracy_min"]),
        "accuracy_std": float(best["accuracy_std"]),
        "representation": "d1",
        "data_protocol": "train.csv estimator fitting; val.csv semantic reference only; test.csv unused",
    }
    (MODELS / "severity_hyperparams.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Best severity configuration:")
    print(json.dumps(params, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
