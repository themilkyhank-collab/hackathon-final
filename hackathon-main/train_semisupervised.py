#!/usr/bin/env python3
"""Build the production CPU-only SVC diagnostic model.

- label SVC is trained on the labelled reference set (val.csv) using the
  selected d1 / StandardScaler / RBF configuration;
- severity SVC is trained on labelled fault rows from val.csv;
- IsolationForest is trained on the unlabeled train.csv distribution;
- test.csv is never loaded here.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.common import clean_spectrum
from src.semi_supervised import SemiSupervisedDiagnosticPipeline

DATA = ROOT / "data"
MODEL = ROOT / "models" / "svc_model.pkl"
SEVERITY_CONFIG = ROOT / "models" / "severity_hyperparams.json"

train = clean_spectrum(pd.read_csv(DATA / "train.csv"))
val = clean_spectrum(pd.read_csv(DATA / "val.csv"))

severity_params = None
if SEVERITY_CONFIG.exists():
    config = json.loads(SEVERITY_CONFIG.read_text(encoding="utf-8"))
    severity_params = {key: config[key] for key in ("C", "gamma", "class_weight") if key in config}

pipeline = SemiSupervisedDiagnosticPipeline(
    feature_count=20,
    label_params={"C": 1.0, "gamma": 0.0175, "class_weight": "balanced"},
    severity_params=severity_params,
)
pipeline.fit(train, val)
pipeline.save(MODEL)

print("Model written:", MODEL)
print("Label model: RBF SVC | d1 | StandardScaler | C=1.0 | gamma=0.0175 | class_weight=balanced")
print("Severity model:", pipeline.severity_params)
print("Label training rows:", pipeline.training_metrics["reference_rows"])
print("Unlabeled anomaly rows:", pipeline.training_metrics["training_rows"])
print("Features:", pipeline.training_metrics["features_before_selection"])
