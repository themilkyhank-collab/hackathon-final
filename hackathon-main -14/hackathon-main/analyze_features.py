#!/usr/bin/env python3
"""Analyze XGBoost feature importance before feature reduction.

The script uses train.csv for fitting and val.csv only as the semantic source
for pseudo-labels. It never reads test.csv. The same importance ranking is used
by the production pipeline to keep the strongest 32 features.
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
OUT = ROOT / "models" / "feature_analysis.json"

train = clean_spectrum(pd.read_csv(DATA / "train.csv"))
val = clean_spectrum(pd.read_csv(DATA / "val.csv"))

pipeline = SemiSupervisedDiagnosticPipeline(feature_count=32)
pipeline.fit(train, val)

report = {
    "features_before_selection": len(pipeline.feature_names),
    "features_after_selection": len(pipeline.selected_features),
    "selected_features": pipeline.selected_features,
    "importance": pipeline.feature_analysis,
    "training_policy": "train.csv only; val.csv semantic pseudo-label reference; test.csv excluded",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Analyzed {len(pipeline.feature_names)} features")
print(f"Selected top {len(pipeline.selected_features)} features")
for i, item in enumerate(pipeline.feature_analysis[:32], 1):
    print(f"{i:02d}. {item['feature']:<28} {item['importance']:.6f}")
print("Report:", OUT)
