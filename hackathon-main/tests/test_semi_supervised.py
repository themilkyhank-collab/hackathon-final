import numpy as np
import pandas as pd

from src.common import FREQ_COLS
from src.semi_supervised import SemiSupervisedDiagnosticPipeline


def _make_rows(n_per_class=20, include_labels=True):
    labels = ["ok", "zakoksowany", "lejacy", "pompa", "iglica", "unknown"]
    fault_labels = {"zakoksowany", "lejacy", "pompa", "iglica"}
    severities = ("male", "srednie", "duze")
    rows = []
    rng = np.random.default_rng(42)
    for class_idx, label in enumerate(labels):
        for i in range(n_per_class):
            d1 = np.full(20, (class_idx - 2.5) * 0.8, dtype=float)
            d1 += np.sin(np.arange(20) * (0.15 + class_idx * 0.02)) * 0.15
            severity = "nie_dotyczy"
            if label in fault_labels:
                severity = severities[i % 3]
                severity_shift = {"male": -0.12, "srednie": 0.0, "duze": 0.12}[severity]
                d1 += severity_shift
            d1 += rng.normal(0.0, 0.01, 20)
            spectrum = np.r_[10.0, 10.0 + np.cumsum(d1)]
            row = {"engine_id": f"engine_{class_idx}_{i}", "cylinder": 1}
            row.update({FREQ_COLS[j]: float(spectrum[j]) for j in range(21)})
            if include_labels:
                row["label"] = label
                row["severity"] = severity
            rows.append(row)
    return pd.DataFrame(rows)


def test_production_svc_pipeline_fit_predict():
    reference = _make_rows(n_per_class=10, include_labels=True)
    train = reference.drop(columns=["label", "severity"]).copy()
    train = pd.concat([train, train.copy()], ignore_index=True)
    train["engine_id"] = [f"train_{i}" for i in range(len(train))]

    pipeline = SemiSupervisedDiagnosticPipeline(
        feature_count=20,
        unknown_confidence=0.2,
        severity_params={"C": 10.0, "gamma": 0.03, "class_weight": "balanced"},
    )
    pipeline.fit(train, reference)

    assert pipeline.model is not None
    assert pipeline.severity_model is not None
    assert pipeline.scaler is not None
    assert pipeline.params["representation"] == "d1"
    assert pipeline.params["label"]["C"] == 1.0
    assert pipeline.params["label"]["gamma"] == 0.0175
    assert pipeline.params["severity"]["C"] == 10.0
    assert pipeline.params["severity"]["gamma"] == 0.03
    assert len(pipeline.selected_features) == 20

    result = pipeline.predict(train.head(12))
    assert len(result) == 12
    assert {"label", "severity", "confidence", "anomaly_score"}.issubset(result.columns)
    assert (result.loc[result["label"].isin(["ok", "unknown"]), "severity"] == "nie_dotyczy").all()
    assert result["confidence"].between(0.0, 1.0).all()
    assert result["anomaly_score"].between(0.0, 1.0).all()
