"""Plot horizontal feature importance from reports/feature_importance.csv.

Usage:
    python plot_feature_importance.py
    python plot_feature_importance.py --top 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/feature_importance.csv")
    parser.add_argument("--output", default="reports/feature_importance.png")
    parser.add_argument("--top", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.input).sort_values("importance", ascending=False).head(args.top)
    df = df.sort_values("importance")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(6, len(df) * 0.28)))
    plt.barh(df["feature"], df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
