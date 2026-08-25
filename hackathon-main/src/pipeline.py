"""Production diagnostic pipeline using RBF SVC."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from .common import FREQ_COLS, FAULT_LABELS
    from .features import FeatureExtractor
except ImportError:
    from common import FREQ_COLS, FAULT_LABELS
    from features import FeatureExtractor

logger = logging.getLogger(__name__)


class EngineDiagnosticPipeline:
    """RBF SVC pipeline with d1 features, anomaly detection and severity models."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_folds: int = 5,
        unknown_threshold: float = 0.35,
        random_state: int = 42,
        ensemble_weights: Optional[Dict[str, float]] = None,
        C: float = 1.0,
        gamma: float = 0.0175,
        class_weight: str = "balanced",
        anomaly_quantile: float = 0.95,
    ):
        # Legacy constructor arguments are retained so existing callers do not break.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_folds = n_folds
        self.unknown_threshold = unknown_threshold
        self.random_state = random_state
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.ensemble_weights = ensemble_weights or {"svc": 1.0}
        self.anomaly_quantile = anomaly_quantile
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.models: Dict[str, Any] = {}
        self.severity_models: Dict[str, Any] = {}
        self.isolation_forest: Optional[IsolationForest] = None
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.training_metrics: Optional[Dict] = None
        self.labels_: Optional[np.ndarray] = None
        self.severity_: Optional[np.ndarray] = None
        self.anomaly_threshold: float = 0.0
        self._train_anomaly_stats: Dict[str, float] = {}

    def _d1(self, df: pd.DataFrame) -> np.ndarray:
        values = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.diff(values, axis=1)

    def _create_svc(self) -> SVC:
        return SVC(
            C=self.C,
            gamma=self.gamma,
            kernel="rbf",
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_state,
        )

    def fit(self, df: pd.DataFrame, label_col: str = "label", severity_col: str = "severity"):
        self.labels_ = df[label_col].astype(str).to_numpy()
        self.severity_ = df.get(severity_col, pd.Series(["nie_dotyczy"] * len(df))).astype(str).to_numpy()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.fit(df)
        self.feature_names = [f"d1_{i}" for i in range(20)]
        X = self._d1(df)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.labels_)
        self.classes_ = self.label_encoder.classes_

        groups = df["engine_id"].astype(str).to_numpy() if "engine_id" in df else np.arange(len(df))
        n_splits = min(self.n_folds, len(np.unique(groups)))
        if n_splits >= 2:
            scores = cross_val_score(
                self._create_svc(),
                X_scaled,
                y,
                groups=groups,
                cv=GroupKFold(n_splits=n_splits),
                scoring="f1_macro",
            )
            cv_results = {
                "svc": {
                    "f1_macro_mean": float(scores.mean()),
                    "f1_macro_std": float(scores.std()),
                    "scores": scores.tolist(),
                }
            }
        else:
            cv_results = {"svc": {"f1_macro_mean": None, "f1_macro_std": None, "scores": []}}

        svc = self._create_svc()
        svc.fit(X_scaled, y)
        self.models = {"svc": svc}
        self._train_severity_models(X_scaled, y, self.severity_)
        self.isolation_forest = IsolationForest(
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)
        # Calibrate the anomaly cutoff from training data using a configurable quantile.
        # Higher scores indicate more anomalous samples.
        train_anomaly_scores = -self.isolation_forest.decision_function(X_scaled)
        self._train_anomaly_stats = {
            "min": float(np.min(train_anomaly_scores)),
            "max": float(np.max(train_anomaly_scores)),
            "mean": float(np.mean(train_anomaly_scores)),
            "median": float(np.median(train_anomaly_scores)),
            "quantile": float(self.anomaly_quantile),
        }
        self.anomaly_threshold = float(np.quantile(train_anomaly_scores, self.anomaly_quantile))
        self.training_metrics = {
            "cv_results": cv_results,
            "n_samples": len(df),
            "n_features": 20,
            "feature_type": "first derivative (d1)",
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "anomaly_threshold": self.anomaly_threshold,
            "timestamp": datetime.now().isoformat(),
        }
        return self

    def _train_severity_models(self, X: np.ndarray, y: np.ndarray, severity: np.ndarray):
        severity_map = {"male": 0, "srednie": 1, "duze": 2}
        self.severity_models = {}
        for fault_label in FAULT_LABELS:
            if fault_label not in self.classes_:
                continue
            fault_idx = int(np.where(self.classes_ == fault_label)[0][0])
            mask = y == fault_idx
            if mask.sum() < 3:
                continue
            y_sev = np.array([severity_map.get(v, -1) for v in severity[mask]], dtype=int)
            valid = y_sev >= 0
            if valid.sum() < 3 or len(np.unique(y_sev[valid])) < 2:
                continue
            model = SVC(
                C=10.0,
                gamma=0.03,
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=self.random_state,
            )
            model.fit(X[mask][valid], y_sev[valid])
            self.severity_models[fault_label] = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.models["svc"].predict_proba(X)

    def predict(self, df: pd.DataFrame, detect_unknown: bool = True) -> pd.DataFrame:
        X_scaled = self.scaler.transform(self._d1(df))
        probs = self.predict_proba(X_scaled)
        y_pred = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        anomaly_scores = -self.isolation_forest.decision_function(X_scaled)
        if detect_unknown:
            unknown_mask = (confidence < self.unknown_threshold) | (anomaly_scores > self.anomaly_threshold)
            # Mark samples as 'unknown' either by assigning to an 'unknown' class if it exists,
            # or by setting their label to 'unknown' directly for detection purposes.
            labels = self.label_encoder.inverse_transform(y_pred).astype(object)
            labels[unknown_mask] = "unknown"
        else:
            labels = self.label_encoder.inverse_transform(y_pred)

        severity_out = np.array(["nie_dotyczy"] * len(df), dtype=object)
        sev_conf = np.full(len(df), np.nan)
        for i, label in enumerate(labels):
            model = self.severity_models.get(label)
            if model is None:
                continue
            p = model.predict_proba(X_scaled[i : i + 1])[0]
            j = int(np.argmax(p))
            severity_out[i] = {0: "male", 1: "srednie", 2: "duze"}.get(int(model.classes_[j]), "nie_dotyczy")
            sev_conf[i] = float(np.max(p))
        return pd.DataFrame({
            "engine_id": df["engine_id"].values if "engine_id" in df else np.arange(len(df)),
            "cylinder": df["cylinder"].values if "cylinder" in df else np.arange(len(df)),
            "label": labels,
            "severity": severity_out,
            "confidence": confidence,
            "severity_confidence": sev_conf,
            "anomaly_score": anomaly_scores,
        })

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = self.predict(df, detect_unknown=False)
        y_true = df["label"].astype(str).to_numpy()
        y_pred = result["label"].to_numpy()
        return {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "raw_score": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=self.classes_).tolist(),
        }

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EngineDiagnosticPipeline":
        return joblib.load(filepath)
