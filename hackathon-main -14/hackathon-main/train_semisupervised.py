#!/usr/bin/env python3
"""Build the production XGBoost model.

train.csv is unlabeled, so val.csv is used only as a semantic reference to
create pseudo-labels. No classifier fitting, feature scaling, feature selection
or threshold tuning uses test.csv.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.common import clean_spectrum
from src.semi_supervised import SemiSupervisedDiagnosticPipeline

DATA = ROOT / "data"
MODEL = ROOT / "models" / "xgboost_model.pkl"

train = clean_spectrum(pd.read_csv(DATA / "train.csv"))
val = clean_spectrum(pd.read_csv(DATA / "val.csv"))

pipeline = SemiSupervisedDiagnosticPipeline(feature_count=32)
pipeline.fit(train, val)
pipeline.save(MODEL)

print("Model written:", MODEL)
print("Training rows:", pipeline.training_metrics["training_rows"])
print("Features:", pipeline.training_metrics["features_before_selection"], "->", pipeline.training_metrics["features_after_selection"])
print("Pseudo confidence:", round(pipeline.training_metrics["mean_pseudo_confidence"], 4))
print("Top features:")
for item in pipeline.feature_importance(12):
    print(f"  {item['feature']}: {item['importance']:.6f}")
