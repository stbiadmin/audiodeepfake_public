#!/usr/bin/env python3
"""Prepare data directories for SOTA benchmarking experiments.

Splits combined JSON feature files into real.json/fake.json pairs,
and sets up symlinks for eval data and 5% sample datasets.

Usage:
    python sota_benchmarking/prepare_data.py
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED_DIR = PROJECT_ROOT / "data" / "features" / "msclap" / "combined"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_training_data" / "msclap"
EVAL_DIR = PROJECT_ROOT / "data" / "eval_features" / "msclap" / "in_the_wild"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

# Combined JSONs that have embeddings and need splitting
COMBINED_DATASETS = {
    "single_voice": "single_voice_combined.json",
    "deepspeak_v2": "deepspeak_v2_train_combined.json",
    "music_instrumental": "music_instrumental_combined.json",
    "music_with_vocals": "music_with_vocals_combined.json",
}

# Datasets only available as 5% samples (no full embeddings yet)
SAMPLE_DATASETS = ["mlaad", "fakeavceleb"]


def split_combined_json(combined_path, output_dir):
    """Split a combined JSON (label=0 real, label=1 fake) into real.json and fake.json."""
    print(f"  Loading {combined_path.name}...")
    with open(combined_path) as f:
        entries = json.load(f)

    real = [e for e in entries if e.get("label") == 0]
    fake = [e for e in entries if e.get("label") == 1]

    output_dir.mkdir(parents=True, exist_ok=True)

    real_path = output_dir / "real.json"
    fake_path = output_dir / "fake.json"

    with open(real_path, "w") as f:
        json.dump(real, f)
    with open(fake_path, "w") as f:
        json.dump(fake, f)

    print(f"    -> {output_dir.name}: {len(real)} real, {len(fake)} fake")
    return len(real), len(fake)


def symlink_or_copy(src, dst):
    """Create a symlink from dst -> src."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    print("Preparing SOTA benchmarking data directories\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split combined JSONs into real/fake
    print("Splitting combined JSONs (datasets with embeddings):")
    for dataset_name, filename in COMBINED_DATASETS.items():
        combined_path = COMBINED_DIR / filename
        if not combined_path.exists():
            print(f"  WARNING: {combined_path} not found, skipping")
            continue
        dataset_out = OUTPUT_DIR / dataset_name
        split_combined_json(combined_path, dataset_out)

    # Create sv_ds_combined by merging single_voice + deepspeak_v2
    print("\nCreating sv_ds_combined (merged single_voice + deepspeak_v2):")
    sv_dir = OUTPUT_DIR / "single_voice"
    ds_dir = OUTPUT_DIR / "deepspeak_v2"
    svds_dir = OUTPUT_DIR / "sv_ds_combined"
    if sv_dir.exists() and ds_dir.exists():
        svds_dir.mkdir(parents=True, exist_ok=True)
        for split in ["real", "fake"]:
            with open(sv_dir / f"{split}.json") as f:
                sv_data = json.load(f)
            with open(ds_dir / f"{split}.json") as f:
                ds_data = json.load(f)
            merged = sv_data + ds_data
            with open(svds_dir / f"{split}.json", "w") as f:
                json.dump(merged, f)
            print(f"    {split}: {len(sv_data)} (sv) + {len(ds_data)} (ds) = {len(merged)}")

    # Symlink 5% sample datasets
    print("\nLinking 5% sample datasets (mlaad, fakeavceleb):")
    for dataset_name in SAMPLE_DATASETS:
        src = SAMPLE_DIR / dataset_name
        dst = OUTPUT_DIR / dataset_name
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                dst.unlink()
            else:
                print(f"  {dataset_name} already exists (not a symlink), skipping")
                continue
        os.symlink(src.resolve(), dst)
        print(f"  {dataset_name} -> {src}")

    # Symlink In-the-Wild eval data
    print("\nLinking In-the-Wild eval data:")
    itw_dst = OUTPUT_DIR / "in_the_wild"
    if EVAL_DIR.exists():
        if itw_dst.exists() or itw_dst.is_symlink():
            if itw_dst.is_symlink():
                itw_dst.unlink()
        os.symlink(EVAL_DIR.resolve(), itw_dst)
        print(f"  in_the_wild -> {EVAL_DIR}")
    else:
        print(f"  WARNING: {EVAL_DIR} not found")

    # Summary
    print("\n--- Data directory summary ---")
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_symlink():
            print(f"  {d.name}/ -> {d.resolve()}")
        elif d.is_dir():
            real_p = d / "real.json"
            fake_p = d / "fake.json"
            if real_p.exists() and fake_p.exists():
                with open(real_p) as f:
                    n_real = len(json.load(f))
                with open(fake_p) as f:
                    n_fake = len(json.load(f))
                print(f"  {d.name}/: {n_real} real, {n_fake} fake")
            else:
                print(f"  {d.name}/: (no real.json/fake.json)")

    print("\nDone.")


if __name__ == "__main__":
    main()
