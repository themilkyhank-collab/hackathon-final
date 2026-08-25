#!/usr/bin/env python3
"""Script to evaluate model on validation data and print raw scores."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.common import clean_spectrum
from src.semi_supervised import SemiSupervisedDiagnosticPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline on validation set")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "val.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "svc_model.pkl")
    args = parser.parse_args()

    # Load or build model
    if args.model.exists():
        print(f"Loading model from {args.model}")
        pipeline = SemiSupervisedDiagnosticPipeline.load(args.model)
    else:
        print("Model not found, training new model...")
        train_df = clean_spectrum(pd.read_csv(args.train))
        val_df = clean_spectrum(pd.read_csv(args.val))
        pipeline = SemiSupervisedDiagnosticPipeline(feature_count=20)
        pipeline.fit(train_df, val_df)
        pipeline.save(args.model)

    # Load validation data
    if not args.val.exists():
        print(f"Validation file not found: {args.val}")
        return
    val_df = clean_spectrum(pd.read_csv(args.val))
    print(f"\n=== Validation Set: {len(val_df)} samples ===")

    # Predict
    results = pipeline.predict(val_df)

    # Merge true labels with predictions
    merged = pd.concat([
        val_df[["label", "severity"]].reset_index(drop=True),
        results[["label", "severity", "confidence", "anomaly_score"]].reset_index(drop=True)
    ], axis=1)
    merged.columns = ["true_label", "true_severity", "pred_label", "pred_severity", "confidence", "anomaly_score"]

    # Overall metrics
    print("\n=== Overall Metrics ===")
    accuracy = (merged["true_label"] == merged["pred_label"]).mean()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean confidence: {merged['confidence'].mean():.4f}")
    print(f"Mean anomaly score: {merged['anomaly_score'].mean():.4f}")

    # Per-class breakdown
    print("\n=== Per-Class Accuracy ===")
    for lbl in sorted(merged["true_label"].unique()):
        subset = merged[merged["true_label"] == lbl]
        correct = (subset["true_label"] == subset["pred_label"]).sum()
        total = len(subset)
        print(f"  {lbl}: {correct}/{total} ({correct/total:.2%})")

    # Confidence by correctness
    print("\n=== Confidence Statistics ===")
    merged["correct"] = merged["true_label"] == merged["pred_label"]
    for correct_val in [True, False]:
        subset = merged[merged["correct"] == correct_val]
        if len(subset) > 0:
            status = "Correct" if correct_val else "Incorrect"
            print(f"  {status}: mean={subset['confidence'].mean():.4f}, std={subset['confidence'].std():.4f}")

    # Severity accuracy (for fault labels only)
    print("\n=== Severity Classification (fault labels only) ===")
    fault_mask = merged["true_label"].isin(["zakoksowany", "lejacy", "pompa", "iglica"])
    if fault_mask.sum() > 0:
        fault_data = merged[fault_mask]
        sev_accuracy = (fault_data["true_severity"] == fault_data["pred_severity"]).mean()
        print(f"Severity accuracy: {sev_accuracy:.4f} ({fault_data['true_severity'].notna().sum()} samples)")

    # Raw score export
    print("\n=== Raw Scores (first 20 rows) ===")
    print(merged[["true_label", "pred_label", "confidence", "anomaly_score"]].head(20).to_string())

    # Save full results
    output_path = ROOT / "validation_results.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
