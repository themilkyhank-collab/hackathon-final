"""Train-only semi-supervised XGBoost diagnostic pipeline with explanations.

The architecture deliberately has two separate XGBoost models:
1. label_model (self.model) predicts the fault class.
2. severity_model predicts severity only for fault classes.

Severity is always ``nie_dotyczy`` for ``ok`` and ``unknown``. Isolation Forest
is kept as an independent anomaly signal and never replaces either classifier.

train.csv is the only dataset used to fit ML estimators. val.csv is used only
as a semantic reference bank to create pseudo-labels because train.csv has no
labels. test.csv is never loaded by this module.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .common import FAULT_LABELS, FREQ_COLS, LABELS
from .features import FeatureExtractor


FAULT_SEVERITY_LABELS = ("zakoksowany", "lejacy", "pompa", "iglica")
SEVERITY_VALUES = ("male", "srednie", "duze")
SEVERITY_TO_INT = {"male": 0, "srednie": 1, "duze": 2}
INT_TO_SEVERITY = {0: "male", 1: "srednie", 2: "duze"}


class SemiSupervisedDiagnosticPipeline:
    def __init__(self, n_estimators=360, max_depth=4, learning_rate=0.035,
                 subsample=0.85, colsample_bytree=0.80, min_child_weight=5,
                 reg_alpha=0.20, reg_lambda=3.0, gamma=0.05, max_bin=128,
                 feature_count=32, random_state=42, unknown_confidence=0.45):
        self.params = {"n_estimators": n_estimators, "max_depth": max_depth,
                       "learning_rate": learning_rate, "subsample": subsample,
                       "colsample_bytree": colsample_bytree, "min_child_weight": min_child_weight,
                       "reg_alpha": reg_alpha, "reg_lambda": reg_lambda, "gamma": gamma,
                       "max_bin": max_bin, "random_state": random_state, "n_jobs": -1,
                       "tree_method": "hist", "eval_metric": "mlogloss", "objective": "multi:softprob"}
        self.feature_count = feature_count
        self.unknown_confidence = unknown_confidence
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.severity_model: Optional[xgb.XGBClassifier] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.class_prototypes: Dict[str, np.ndarray] = {}
        self.selected_prototypes: Dict[str, np.ndarray] = {}
        self.severity_prototypes: Dict[str, np.ndarray] = {}
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.feature_analysis: List[Dict[str, float]] = []
        self.severity_feature_analysis: List[Dict[str, float]] = []
        self.training_metrics: Dict[str, object] = {}

    @staticmethod
    def _matrix(features, names):
        return np.nan_to_num(features[names].to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _distance_confidence(distances):
        scale = np.median(distances, axis=1, keepdims=True) + 1e-6
        logits = -distances / scale; logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(np.clip(logits, -30, 0)); weights /= weights.sum(axis=1, keepdims=True)
        return weights.max(axis=1)

    def _new_xgb(self, num_class):
        return xgb.XGBClassifier(**self.params, num_class=num_class)

    def fit(self, train_df, reference_df):
        if "label" not in reference_df.columns:
            raise ValueError("reference_df must contain the labeled 'label' column")
        if "severity" not in reference_df.columns:
            reference_df = reference_df.copy(); reference_df["severity"] = "nie_dotyczy"

        self.feature_extractor = FeatureExtractor()
        train_features = self.feature_extractor.fit_transform(train_df)
        ref_features = self.feature_extractor.transform(reference_df)
        self.feature_names = self.feature_extractor.get_feature_names()
        self.scaler = StandardScaler()
        X_train_all = self.scaler.fit_transform(self._matrix(train_features, self.feature_names)).astype(np.float32)
        X_ref_all = self.scaler.transform(self._matrix(ref_features, self.feature_names)).astype(np.float32)

        ref_labels = reference_df["label"].astype(str).to_numpy()
        labels = [label for label in LABELS if label in set(ref_labels)]
        if not labels: raise ValueError("No supported labels found in reference_df")
        self.label_encoder = LabelEncoder(); self.label_encoder.fit(labels)
        for label in self.label_encoder.classes_:
            mask = ref_labels == label
            if mask.any(): self.class_prototypes[label] = np.median(X_ref_all[mask], axis=0)

        class_names = list(self.label_encoder.classes_)
        prototype_matrix = np.vstack([self.class_prototypes[name] for name in class_names])
        distances = np.stack([np.linalg.norm(X_train_all - prototype, axis=1) for prototype in prototype_matrix], axis=1)
        pseudo_labels = np.asarray(class_names, dtype=object)[distances.argmin(axis=1)]
        confidence = self._distance_confidence(distances)
        y_train = self.label_encoder.transform(pseudo_labels)
        sample_weight = np.clip(0.35 + 0.95 * confidence, 0.35, 1.0).astype(np.float32)

        # First XGBoost is used only to rank features; final label model uses the selected subset.
        analysis_model = self._new_xgb(len(class_names)); analysis_model.fit(X_train_all, y_train, sample_weight=sample_weight)
        importances = analysis_model.feature_importances_.astype(float); order = np.argsort(importances)[::-1]
        self.feature_analysis = [{"feature": self.feature_names[int(i)], "importance": float(importances[i])} for i in order]
        self.selected_features = [x["feature"] for x in self.feature_analysis[:min(self.feature_count, len(self.feature_names))]]
        selected_idx = [self.feature_names.index(name) for name in self.selected_features]

        # MODEL 1: fault/label classifier.
        self.model = self._new_xgb(len(class_names)); self.model.fit(X_train_all[:, selected_idx], y_train, sample_weight=sample_weight)
        self.selected_prototypes = {label: proto[selected_idx] for label, proto in self.class_prototypes.items()}

        # Anomaly detector is independent from both supervised classifiers.
        self.isolation_forest = IsolationForest(n_estimators=240, max_samples="auto", contamination="auto",
                                                random_state=self.params["random_state"], n_jobs=-1)
        self.isolation_forest.fit(X_train_all[:, selected_idx])

        # MODEL 2: one global severity classifier. It is trained only on pseudo-severity
        # targets assigned to train.csv rows whose pseudo label is a real fault.
        self._fit_severity_model(X_train_all[:, selected_idx], X_ref_all[:, selected_idx],
                                 pseudo_labels, reference_df, ref_labels, confidence)
        self.training_metrics = {
            "training_rows": int(len(train_df)), "reference_rows": int(len(reference_df)),
            "pseudo_label_counts": pd.Series(pseudo_labels).value_counts().to_dict(),
            "mean_pseudo_confidence": float(confidence.mean()),
            "low_confidence_fraction": float((confidence < self.unknown_confidence).mean()),
            "features_before_selection": len(self.feature_names), "features_after_selection": len(self.selected_features),
            "models": {"label": "XGBoost", "severity": "XGBoost", "anomaly": "IsolationForest"},
            "training_data": "train.csv only",
            "reference_role": "semantic pseudo-label and pseudo-severity source only"
        }
        return self

    def _fit_severity_model(self, X_train, X_ref, pseudo_labels, reference_df, ref_labels, confidence):
        self.severity_model = None
        self.severity_prototypes = {}

        # Build severity prototypes from the labeled reference set for each fault.
        # They provide semantic targets; the actual XGBoost fit below uses train.csv rows.
        for fault in FAULT_SEVERITY_LABELS:
            ref_mask = ref_labels == fault
            if not ref_mask.any():
                continue
            ref_fault = X_ref[ref_mask]
            ref_sev = reference_df.loc[ref_mask, "severity"].astype(str).to_numpy()
            for severity in SEVERITY_VALUES:
                mask = ref_sev == severity
                if mask.any():
                    self.severity_prototypes[f"{fault}:{severity}"] = np.median(ref_fault[mask], axis=0)

        available = [k for k in self.severity_prototypes if k.split(":", 1)[1] in SEVERITY_VALUES]
        if len(available) < 2:
            return

        # Only rows pseudo-labeled as one of the four fault classes participate.
        train_mask = np.isin(pseudo_labels, FAULT_SEVERITY_LABELS) & (confidence >= self.unknown_confidence)
        if train_mask.sum() < 30:
            return

        fault_X = X_train[train_mask]
        fault_labels = pseudo_labels[train_mask]
        y_severity = np.empty(len(fault_X), dtype=np.int32)
        valid = np.ones(len(fault_X), dtype=bool)

        for i, (row_x, fault) in enumerate(zip(fault_X, fault_labels)):
            candidates = [(sev, proto) for key, proto in self.severity_prototypes.items()
                          if key.startswith(f"{fault}:")
                          for sev in [key.split(":", 1)[1]]]
            if len(candidates) < 2:
                valid[i] = False
                continue
            distances = np.asarray([np.linalg.norm(row_x - proto) for _, proto in candidates])
            y_severity[i] = SEVERITY_TO_INT[candidates[int(distances.argmin())][0]]

        fault_X = fault_X[valid]
        y_severity = y_severity[valid]
        if len(fault_X) < 30 or len(np.unique(y_severity)) < 2:
            return

        # One model, three ordered severity classes. It never sees ok/unknown rows.
        self.severity_model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.20,
            reg_lambda=3.0,
            gamma=0.05,
            max_bin=128,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.params["random_state"],
            n_jobs=-1,
        )
        self.severity_model.fit(fault_X, y_severity)
        self.severity_feature_analysis = [
            {"feature": self.selected_features[i], "importance": float(value)}
            for i, value in enumerate(self.severity_model.feature_importances_)
        ]
        self.severity_feature_analysis.sort(key=lambda x: x["importance"], reverse=True)

    def _transform(self, df):
        if self.feature_extractor is None or self.scaler is None: raise RuntimeError("Model is not fitted")
        features = self.feature_extractor.transform(df)
        scaled = self.scaler.transform(self._matrix(features, self.feature_names)).astype(np.float32)
        indices = [self.feature_names.index(name) for name in self.selected_features]
        return features, scaled[:, indices]

    def _explain_row(self, row_idx, features, X, label, confidence, anomaly_score):
        raw = features.iloc[row_idx][FREQ_COLS].to_numpy(dtype=float)
        engine_id = features.iloc[row_idx]["engine_id"]
        same_engine = features["engine_id"].astype(str) == str(engine_id)
        peer = features.loc[same_engine, FREQ_COLS].astype(float).median(axis=0).to_numpy()
        deviation = np.abs(raw - peer)
        band_idx = int(np.argmax(deviation)); band_pct = float(deviation[band_idx] / (abs(peer[band_idx]) + 1e-6) * 100)

        proto = self.selected_prototypes.get(label)
        top_features = []
        if proto is not None:
            importances = np.array([x["importance"] for x in self.feature_analysis if x["feature"] in self.selected_features])
            vals = np.abs(X[row_idx] - proto) * (importances + 1e-8)
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

    def predict(self, df):
        if self.model is None or self.label_encoder is None or self.isolation_forest is None: raise RuntimeError("Model is not fitted")
        if df.empty: return pd.DataFrame(columns=["row_index", "engine_id", "cylinder", "label", "severity", "confidence", "anomaly_score", "explanation"])
        features, X = self._transform(df)
        probabilities = self.model.predict_proba(X); pred_idx = probabilities.argmax(axis=1); confidence = probabilities.max(axis=1)
        labels = self.label_encoder.inverse_transform(pred_idx).astype(object)
        anomaly_raw = self.isolation_forest.decision_function(X)
        anomaly_score = np.clip(0.5 - anomaly_raw, 0.0, 1.0)
        if "unknown" in self.label_encoder.classes_:
            unknown = self.label_encoder.classes_[np.where(self.label_encoder.classes_ == "unknown")[0][0]]
            labels[confidence < self.unknown_confidence] = unknown
        severity = self._predict_severity(X, labels)
        explanations = [self._explain_row(i, features, X, labels[i], confidence[i], anomaly_score[i]) for i in range(len(df))]
        return pd.DataFrame({"row_index": np.arange(len(df), dtype=int), "engine_id": features["engine_id"].to_numpy(),
            "cylinder": features["cylinder"].to_numpy(), "label": labels, "severity": severity,
            "confidence": confidence, "anomaly_score": anomaly_score, "explanation": explanations})

    def _predict_severity(self, X, labels):
        # Hard rule from the competition specification: ok/unknown are never assigned severity.
        output = np.full(len(labels), "nie_dotyczy", dtype=object)
        if self.severity_model is None:
            return output.tolist()
        fault_mask = np.isin(labels, FAULT_SEVERITY_LABELS)
        idx = np.flatnonzero(fault_mask)
        if idx.size:
            pred = self.severity_model.predict(X[idx]).astype(int)
            output[idx] = [INT_TO_SEVERITY.get(int(v), "male") for v in pred]
        return output.tolist()

    def feature_importance(self, limit=12): return (self.feature_analysis or [])[:limit]
    def severity_feature_importance(self, limit=12): return (self.severity_feature_analysis or [])[:limit]
    def save(self, filepath): Path(filepath).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, filepath, compress=3)
    @classmethod
    def load(cls, filepath): return joblib.load(filepath)
