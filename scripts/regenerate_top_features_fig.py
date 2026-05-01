#!/usr/bin/env python3
"""Regenerate the single_voice top features figure using MS-CLAP data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# Paths - MS-CLAP data
REAL_PATH = PROJECT_ROOT / "data/features/msclap/single_voice_real.json"
FAKE_PATH = PROJECT_ROOT / "data/features/msclap/single_voice_fake.json"
IMPORTANCE_CSV = PROJECT_ROOT / "data/analysis_scientific/msclap/single_voice/feature_importance.csv"
OUTPUT_PATH = PROJECT_ROOT / "papers/audio_deepfake/figures/single_voice_top_features.png"


def load_features(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for item in data:
        row = {"label": item.get("label", 0)}
        row.update(item.get("features", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    real_df = load_features(REAL_PATH)
    fake_df = load_features(FAKE_PATH)
    importance_df = pd.read_csv(IMPORTANCE_CSV)

    top_features = list(zip(importance_df["feature"].head(8), importance_df["auc"].head(8)))

    n_cols = 4
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5))
    axes = axes.flatten()

    for i, (feature, auc) in enumerate(top_features):
        ax = axes[i]
        real_vals = real_df[feature].dropna()
        fake_vals = fake_df[feature].dropna()

        bins = np.linspace(
            min(real_vals.min(), fake_vals.min()),
            max(real_vals.max(), fake_vals.max()),
            30,
        )
        ax.hist(real_vals, bins=bins, density=True, alpha=0.6, color="#2166ac", label="Real")
        ax.hist(fake_vals, bins=bins, density=True, alpha=0.6, color="#b2182b", label="Fake")
        ax.set_title(f"{feature}\nAUC={auc:.3f}", fontsize=10)
        ax.set_xlabel("")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Discriminative Features (Single Voice, MS-CLAP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
