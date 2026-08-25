#!/usr/bin/env python3
"""Reproducible model evaluator and optional submission generator.

If the input has labels, the script evaluates with GroupKFold by engine_id,
matching the hackathon protocol. It reports Macro-F1(label), severity accuracy
on true fault rows, and Raw_Score. It never writes a CSV unless --output is
explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import FAULT_LABELS, FREQ_COLS, LABELS, clean_spectrum
from src.semi_supervised import SemiSupervisedDiagnosticPipeline

MODEL_PATH = ROOT / "models" / "svc_model.pkl"
DEFAULT_INPUT = ROOT / "data" / "test.csv"
LABEL_PARAMS = {"C": 1.0, "gamma": 0.0175, "class_weight": "balanced"}
SEVERITY_PARAMS = {"C": 10.0, "gamma": 0.03, "class_weight": "balanced"}
SEVERITY_TO_INT = {"male": 0, "srednie": 1, "duze": 2}
INT_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_INT.items()}


def load_model() -> SemiSupervisedDiagnosticPipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run 'python train_semisupervised.py' first."
        )
    return SemiSupervisedDiagnosticPipeline.load(str(MODEL_PATH))


def d1(df: pd.DataFrame) -> np.ndarray:
    raw = df[FREQ_COLS].to_numpy(dtype=np.float32, copy=False)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.diff(raw, axis=1).astype(np.float32)


def cv_evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict[str, float | int]:
    """Calculate the hackathon score without evaluating on the training fold."""
    if "label" not in df.columns:
        raise ValueError("Evaluation requires a 'label' column.")
    if "engine_id" not in df.columns:
        raise ValueError("Group-aware evaluation requires 'engine_id'.")

    y = df["label"].astype(str).to_numpy()
    groups = df["engine_id"].astype(str).to_numpy()
    X = d1(df)
    splitter = GroupKFold(n_splits=min(n_splits, df["engine_id"].nunique()))
    label_oof = np.empty(len(df), dtype=object)
    severity_oof = np.full(len(df), "nie_dotyczy", dtype=object)

    for train_idx, holdout_idx in splitter.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_ho = scaler.transform(X[holdout_idx])

        label_model = SVC(**LABEL_PARAMS, kernel="rbf", cache_size=512)
        label_model.fit(X_tr, y[train_idx])
        label_oof[holdout_idx] = label_model.predict(X_ho)

        if "severity" in df.columns:
            fault_tr = np.isin(y[train_idx], FAULT_LABELS)
            sev = df["severity"].astype(str).to_numpy()
            sev_y = np.asarray([SEVERITY_TO_INT.get(v, -1) for v in sev[train_idx]])
            valid = fault_tr & (sev_y >= 0)
            if valid.sum() >= 10 and len(np.unique(sev_y[valid])) >= 2:
                severity_model = SVC(**SEVERITY_PARAMS, kernel="rbf", cache_size=512)
                severity_model.fit(X_tr[valid], sev_y[valid])
                pred = severity_model.predict(X_ho)
                severity_oof[holdout_idx] = [INT_TO_SEVERITY[int(v)] for v in pred]

    macro_f1 = float(f1_score(y, label_oof, average="macro", zero_division=0))
    result: dict[str, float | int] = {"macro_f1_label": macro_f1}

    if "severity" in df.columns:
        fault_mask = np.isin(y, FAULT_LABELS)
        if fault_mask.any():
            severity_accuracy = float(
                accuracy_score(df.loc[fault_mask, "severity"].astype(str), severity_oof[fault_mask])
            )
            result["severity_accuracy_faults"] = severity_accuracy
            result["fault_rows"] = int(fault_mask.sum())
            result["raw_score"] = 0.75 * macro_f1 + 0.25 * severity_accuracy
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Aesteel SVC model reproducibly.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None, help="Write predictions only when explicitly requested.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model = load_model()
    predictions = model.predict(clean_spectrum(df.copy()))

    if args.output is not None:
        submission = predictions[["engine_id", "cylinder", "label", "severity"]].copy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output, index=False)
        print(f"Predictions written to: {args.output}")

    if "label" in df.columns:
        metrics = cv_evaluate(df)
        if args.json:
            print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True))
        else:
            print("Evaluation: 5-fold GroupKFold by engine_id (out-of-fold)")
            print(f"Macro F1 (label):             {metrics['macro_f1_label']:.6f}")
            if "severity_accuracy_faults" in metrics:
                print(f"Severity accuracy (faults):   {metrics['severity_accuracy_faults']:.6f}")
                print(f"Raw_Score:                    {metrics['raw_score']:.6f}")
                score = float(metrics["raw_score"])
                points = max(0.0, min(40.0, 40.0 * (score - 0.80) / 0.20))
                print(f"Hackathon points:             {points:.2f}/40")
    else:
        print(f"Predicted rows: {len(predictions)}")
        print("No labels supplied; Raw_Score was not calculated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
