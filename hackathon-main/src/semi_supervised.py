"""Production diagnostic pipeline.

The labelled validation/reference set is the supervised training source for the
classification models. ``train.csv`` is unlabeled and is used only for the
unsupervised IsolationForest/anomaly distribution and optional pseudo-label
research; it is never treated as ground-truth. ``test.csv`` is never loaded
here.

Production classifiers:
- label: d1 -> StandardScaler -> RBF SVC
- severity: d1 -> StandardScaler -> RBF SVC, fault rows only
- anomaly: independent IsolationForest trained on unlabeled train.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .common import FAULT_LABELS, FREQ_COLS, LABELS

FAULT_SEVERITY_LABELS = tuple(FAULT_LABELS)
SEVERITY_VALUES = ("male", "srednie", "duze")
SEVERITY_TO_INT = {"male": 0, "srednie": 1, "duze": 2}
INT_TO_SEVERITY = {0: "male", 1: "srednie", 2: "duze"}
LABEL_SVC_PARAMS = {"C": 1.0, "gamma": 0.0175, "class_weight": "balanced"}
DEFAULT_SEVERITY_SVC_PARAMS = {"C": 10.0, "gamma": 0.03, "class_weight": "balanced"}
SEVERITY_MIN_CONFIDENCE = 0.30


class SemiSupervisedDiagnosticPipeline:
    """CPU-only SVC diagnostic pipeline with reproducible labelled training."""

    def __init__(
        self,
        feature_count: int = 20,
        random_state: int = 42,
        unknown_confidence: float = 0.35,
        label_params: Optional[Dict[str, object]] = None,
        severity_params: Optional[Dict[str, object]] = None,
    ) -> None:
        self.feature_count = feature_count
        self.random_state = random_state
        self.unknown_confidence = unknown_confidence
        self.label_params = {**LABEL_SVC_PARAMS, **(label_params or {})}
        self.severity_params = {**DEFAULT_SEVERITY_SVC_PARAMS, **(severity_params or {})}
        self.params = {
            "label": self.label_params.copy(),
            "severity": self.severity_params.copy(),
            "representation": "d1",
            "scaler": "StandardScaler",
            "kernel": "rbf",
            "random_state": random_state,
        }
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[SVC] = None
        self.severity_model: Optional[SVC] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.class_prototypes: Dict[str, np.ndarray] = {}
        self.selected_prototypes: Dict[str, np.ndarray] = {}
        self.severity_prototypes: Dict[str, np.ndarray] = {}
        self.feature_names: List[str] = [f"d1_{i}" for i in range(len(FREQ_COLS) - 1)]
        self.selected_features: List[str] = self.feature_names.copy()
        self.feature_analysis: List[Dict[str, float]] = []
        self.severity_feature_analysis: List[Dict[str, float]] = []
        self.training_metrics: Dict[str, object] = {}

    @staticmethod
    def _raw_matrix(df: pd.DataFrame) -> np.ndarray:
        raw = df[FREQ_COLS].to_numpy(dtype=np.float32, copy=False)
        # Preserve local shape while making missing measurements deterministic.
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _d1(cls, df: pd.DataFrame) -> np.ndarray:
        return np.diff(cls._raw_matrix(df), axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _decision_confidence(model: SVC, X: np.ndarray) -> np.ndarray:
        scores = model.decision_function(X)
        if scores.ndim == 1:
            # Binary SVC: convert signed margin to two comparable class scores.
            scores = np.column_stack([-scores, scores])
        centered = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(centered, -30.0, 0.0))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs.max(axis=1).astype(np.float32)

    @staticmethod
    def _make_svc(params: Dict[str, object]) -> SVC:
        return SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            class_weight=params.get("class_weight", "balanced"),
            probability=False,
            cache_size=512,
        )

    def _feature_importance(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, float]]:
        if len(np.unique(y)) < 2:
            values = np.zeros(X.shape[1], dtype=float)
        else:
            values = mutual_info_classif(
                X, y, random_state=self.random_state, discrete_features=False
            )
        order = np.argsort(values)[::-1]
        return [
            {"feature": self.feature_names[int(i)], "importance": float(values[i])}
            for i in order
        ]

    def fit(self, train_df: pd.DataFrame, reference_df: pd.DataFrame) -> "SemiSupervisedDiagnosticPipeline":
        """Fit classifiers on the labelled reference set and anomaly model on train.

        ``reference_df`` is the labelled 40-engine set used for GroupKFold
        model selection and final supervised fitting. ``train_df`` has no labels
        and is never assigned synthetic ground truth for the production SVC.
        """
        if "label" not in reference_df.columns:
            raise ValueError("reference_df must contain the labeled 'label' column")
        reference_df = reference_df.copy()
        if "severity" not in reference_df.columns:
            reference_df["severity"] = "nie_dotyczy"

        X_train_raw = self._d1(train_df)
        X_ref_raw = self._d1(reference_df)

        # Fit the supervised scaler only on the labelled reference set.
        self.scaler = StandardScaler()
        X_ref = self.scaler.fit_transform(X_ref_raw).astype(np.float32)
        X_train = self.scaler.transform(X_train_raw).astype(np.float32)

        ref_labels = reference_df["label"].astype(str).to_numpy()
        available_labels = [label for label in LABELS if label in set(ref_labels)]
        if len(available_labels) < 2:
            raise ValueError("At least two supported labels are required in reference_df")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(available_labels)
        y_label = self.label_encoder.transform(ref_labels)

        # Keep all d1 features. The selected configuration is explicitly d1;
        # feature ranking is explanatory metadata, not a second model-selection step.
        self.feature_analysis = self._feature_importance(X_ref, y_label)
        self.selected_features = self.feature_names.copy()
        selected_idx = np.arange(X_ref.shape[1])

        # Final label model: the selected RBF SVC configuration.
        self.model = self._make_svc(self.label_params)
        self.model.fit(X_ref[:, selected_idx], y_label)

        # Reference prototypes are used only for explanations.
        self.class_prototypes = {
            label: np.median(X_ref[ref_labels == label], axis=0)
            for label in self.label_encoder.classes_
            if np.any(ref_labels == label)
        }
        self.selected_prototypes = self.class_prototypes.copy()

        # Unlabelled train data is valuable for anomaly detection and does not
        # contaminate the supervised label/severity targets.
        self.isolation_forest = IsolationForest(
            n_estimators=160,
            max_samples="auto",
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_train[:, selected_idx])

        self._fit_severity_model(X_ref[:, selected_idx], reference_df, ref_labels)

        self.training_metrics = {
            "training_rows": int(len(train_df)),
            "reference_rows": int(len(reference_df)),
            "models": {"label": "RBF SVC", "severity": "RBF SVC", "anomaly": "IsolationForest"},
            "label_training_data": "val.csv labelled reference",
            "severity_training_data": "val.csv labelled fault rows",
            "anomaly_training_data": "train.csv unlabeled",
            "representation": "d1",
            "features_before_selection": len(self.feature_names),
            "features_after_selection": len(self.selected_features),
            "label_params": self.label_params.copy(),
            "severity_params": self.severity_params.copy(),
            "severity_min_confidence": SEVERITY_MIN_CONFIDENCE,
        }
        return self

    def _fit_severity_model(self, X_ref: np.ndarray, reference_df: pd.DataFrame, ref_labels: np.ndarray) -> None:
        fault_mask = np.isin(ref_labels, FAULT_SEVERITY_LABELS)
        severity = reference_df.loc[fault_mask, "severity"].astype(str).to_numpy()
        valid = np.isin(severity, SEVERITY_VALUES)
        if valid.sum() < 10 or len(np.unique(severity[valid])) < 2:
            self.severity_model = None
            return
        self.severity_model = self._make_svc(self.severity_params)
        self.severity_model.fit(X_ref[fault_mask][valid], np.asarray([SEVERITY_TO_INT[s] for s in severity[valid]], dtype=np.int32))

        values = mutual_info_classif(
            X_ref[fault_mask][valid],
            np.asarray([SEVERITY_TO_INT[s] for s in severity[valid]], dtype=np.int32),
            random_state=self.random_state,
            discrete_features=False,
        )
        order = np.argsort(values)[::-1]
        names = np.asarray(self.feature_names)
        self.severity_feature_analysis = [
            {"feature": str(names[int(i)]), "importance": float(values[i])}
            for i in order
        ]

    def _transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if self.scaler is None:
            raise RuntimeError("Model is not fitted")
        X = self.scaler.transform(self._d1(df)).astype(np.float32)
        indices = [self.feature_names.index(name) for name in self.selected_features]
        return df, X[:, indices]

    def _explain_row(self, row_idx: int, features: pd.DataFrame, X: np.ndarray, label: str, confidence: float, anomaly_score: float) -> Dict[str, object]:
        raw = features.iloc[row_idx][FREQ_COLS].to_numpy(dtype=float)
        engine_id = features.iloc[row_idx]["engine_id"]
        same_engine = features["engine_id"].astype(str) == str(engine_id)
        peer = features.loc[same_engine, FREQ_COLS].astype(float).median(axis=0).to_numpy()
        deviation = np.abs(raw - peer)
        band_idx = int(np.argmax(np.nan_to_num(deviation, nan=0.0)))
        band_pct = float(deviation[band_idx] / (abs(peer[band_idx]) + 1e-6) * 100)
        proto = self.selected_prototypes.get(label)
        top_features: List[Dict[str, float]] = []
        if proto is not None:
            vals = np.abs(X[row_idx] - proto)
            top_idx = np.argsort(vals)[::-1][:3]
            for i in top_idx:
                top_features.append({"feature": self.selected_features[int(i)], "impact": float(vals[int(i)])})
        if anomaly_score >= 0.65:
            action = "Anomaly signal is high; inspect the cylinder against its engine peers."
        elif anomaly_score >= 0.45:
            action = "Moderate anomaly signal; compare the indicated spectrum band before service."
        else:
            action = "Pattern is close to the learned training distribution."
        return {
            "anomaly_score": round(float(anomaly_score), 4),
            "anomalous_band": f"mV_{band_idx}",
            "band_deviation_pct": round(band_pct, 1),
            "peer_comparison": f"cylinder spectrum differs most at mV_{band_idx} vs median of engine {engine_id}",
            "top_features": top_features,
            "action": action,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.label_encoder is None or self.isolation_forest is None:
            raise RuntimeError("Model is not fitted")
        if df.empty:
            return pd.DataFrame(columns=["row_index", "engine_id", "cylinder", "label", "severity", "confidence", "anomaly_score", "explanation"])
        features, X = self._transform(df)
        pred_idx = self.model.predict(X).astype(int)
        labels = self.label_encoder.inverse_transform(pred_idx).astype(object)
        confidence = self._decision_confidence(self.model, X)
        anomaly_raw = self.isolation_forest.decision_function(X)
        anomaly_score = np.clip(0.5 - anomaly_raw, 0.0, 1.0)
        if "unknown" in self.label_encoder.classes_:
            labels[confidence < self.unknown_confidence] = "unknown"
        severity = self._predict_severity(X, labels)
        explanations = [
            self._explain_row(i, features, X, str(labels[i]), confidence[i], anomaly_score[i])
            for i in range(len(df))
        ]
        return pd.DataFrame({
            "row_index": np.arange(len(df), dtype=int),
            "engine_id": features["engine_id"].to_numpy(),
            "cylinder": features["cylinder"].to_numpy(),
            "label": labels,
            "severity": severity,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "explanation": explanations,
        })

    def _predict_severity(self, X: np.ndarray, labels: np.ndarray) -> List[str]:
        output = np.full(len(labels), "nie_dotyczy", dtype=object)
        if self.severity_model is None:
            return output.tolist()
        fault_mask = np.isin(labels, FAULT_SEVERITY_LABELS)
        idx = np.flatnonzero(fault_mask)
        if idx.size:
            pred = self.severity_model.predict(X[idx]).astype(int)
            output[idx] = [INT_TO_SEVERITY.get(int(v), "male") for v in pred]
        return output.tolist()

    def feature_importance(self, limit: int = 12) -> List[Dict[str, float]]:
        return (self.feature_analysis or [])[:limit]

    def severity_feature_importance(self, limit: int = 12) -> List[Dict[str, float]]:
        return (self.severity_feature_analysis or [])[:limit]

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, filepath: str | Path) -> "SemiSupervisedDiagnosticPipeline":
        return joblib.load(filepath)


def grouped_svc_score(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, gamma: float, class_weight: object = "balanced", n_splits: int = 5) -> Dict[str, float]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two engine groups are required")
    splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
    scores: List[float] = []
    for train_idx, holdout_idx in splitter.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_ho = scaler.transform(X[holdout_idx])
        model = SVC(kernel="rbf", C=float(C), gamma=float(gamma), class_weight=class_weight, cache_size=512)
        model.fit(X_tr, y[train_idx])
        scores.append(float((model.predict(X_ho) == y[holdout_idx]).mean()))
    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "accuracy_min": float(np.min(scores)),
    }
