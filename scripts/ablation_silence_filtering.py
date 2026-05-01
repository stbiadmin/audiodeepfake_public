#!/usr/bin/env python3
"""Ablation study: Impact of silence filtering on deepfake detection.

Tests whether filtering out low-energy (silent/quiet) segments improves
classification performance on In-the-Wild evaluation.
"""

import json
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_FEATURES = PROJECT_ROOT / "data" / "features" / "msclap"
EVAL_FEATURES = PROJECT_ROOT / "data" / "eval_features" / "msclap" / "in_the_wild"

# For raw audio analysis
ITW_AUDIO_REAL = PROJECT_ROOT / "data" / "eval_datasets" / "in_the_wild" / "organized" / "real"
ITW_AUDIO_FAKE = PROJECT_ROOT / "data" / "eval_datasets" / "in_the_wild" / "organized" / "fake"


def compute_segment_energy(audio: np.ndarray, sr: int,
                           segment_duration: float = 2.0,
                           segment_hop: float = 1.0) -> List[float]:
    """Compute RMS energy for each segment."""
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_hop * sr)

    energies = []
    start = 0
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        rms = np.sqrt(np.mean(segment ** 2))
        energies.append(rms)
        start += hop_samples

    return energies


def analyze_silence_in_dataset(audio_dir: Path, max_files: int = 100) -> dict:
    """Analyze silence prevalence in a dataset."""
    if not audio_dir.exists():
        return None

    all_energies = []
    silent_segment_counts = []
    total_segment_counts = []

    # Energy threshold: -40dB relative to max is typically considered silence
    # RMS threshold ~0.01 for normalized audio
    silence_threshold = 0.01

    files = list(audio_dir.glob("**/*.wav")) + list(audio_dir.glob("**/*.mp3"))
    files = files[:max_files]

    for f in files:
        try:
            audio, sr = librosa.load(str(f), sr=48000, mono=True)
            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            energies = compute_segment_energy(audio, sr)
            if len(energies) < 3:
                continue

            all_energies.extend(energies)
            silent_count = sum(1 for e in energies if e < silence_threshold)
            silent_segment_counts.append(silent_count)
            total_segment_counts.append(len(energies))
        except Exception:
            continue

    if not all_energies:
        return None

    return {
        "mean_energy": np.mean(all_energies),
        "median_energy": np.median(all_energies),
        "silent_segments": sum(silent_segment_counts),
        "total_segments": sum(total_segment_counts),
        "silent_ratio": sum(silent_segment_counts) / sum(total_segment_counts) if sum(total_segment_counts) > 0 else 0,
        "files_analyzed": len(total_segment_counts),
        "percentile_1": np.percentile(all_energies, 1),
        "percentile_5": np.percentile(all_energies, 5),
        "percentile_10": np.percentile(all_energies, 10),
    }


def load_features(filepath: Path) -> pd.DataFrame:
    """Load features from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    records = []
    for item in data:
        if "features" in item and item["features"]:
            record = item["features"].copy()
            record["file"] = item.get("file", item.get("file_path", "unknown"))
            # segment_count is stored at top level, n_samples in features
            record["segment_count"] = item.get("segment_count", 0)
            record["similarity_count"] = item.get("similarity_count", 0)
            records.append(record)

    return pd.DataFrame(records)


def filter_by_segment_count(df: pd.DataFrame, misegment_count: int) -> pd.DataFrame:
    """Filter samples by minimum segment count (proxy for non-silent segments)."""
    if "segment_count" in df.columns:
        return df[df["segment_count"] >= misegment_count].copy()
    return df


def load_training_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load DeepSpeak training data."""
    real_path = TRAIN_FEATURES / "deepspeak_v2_train_real.json"
    fake_path = TRAIN_FEATURES / "deepspeak_v2_train_fake.json"

    real_df = load_features(real_path)
    fake_df = load_features(fake_path)

    real_df["label"] = 0
    fake_df["label"] = 1

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ["file", "label", "segment_count"]]

    # Balance classes
    min_count = min(len(real_df), len(fake_df))
    real_sample = combined[combined["label"] == 0].sample(n=min_count, random_state=42)
    fake_sample = combined[combined["label"] == 1].sample(n=min_count, random_state=42)
    balanced = pd.concat([real_sample, fake_sample], ignore_index=True)

    X = balanced[feature_cols]
    y = balanced["label"].values

    return X, y, feature_cols


def load_eval_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load In-the-Wild evaluation data."""
    real_df = load_features(EVAL_FEATURES / "real.json")
    fake_df = load_features(EVAL_FEATURES / "fake.json")

    real_df["label"] = 0
    fake_df["label"] = 1

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ["file", "label", "segment_count"]]

    return combined, feature_cols


def evaluate_with_segment_filter(X_train, y_train, X_eval, y_eval,
                                  features: List[str], n_features: int = 8) -> dict:
    """Train and evaluate with top-N features by AUC."""
    # Select top N features by AUC
    rankings = {}
    for col in features:
        if col in X_train.columns:
            try:
                auc = roc_auc_score(y_train, X_train[col].values)
                auc = max(auc, 1 - auc)
                rankings[col] = auc
            except:
                rankings[col] = 0.5

    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ranked[:n_features]]

    X_train_sel = X_train[selected]
    X_eval_sel = X_eval[selected]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_eval_scaled = scaler.transform(X_eval_sel)

    clf = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=100,
        random_state=42, verbosity=0
    )
    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_eval_scaled)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "f1": f1_score(y_eval, y_pred),
        "auc": roc_auc_score(y_eval, y_prob),
        "accuracy": accuracy_score(y_eval, y_pred),
    }


def main():
    print("=" * 70)
    print("Ablation Study: Silence/Low-Energy Segment Analysis")
    print("=" * 70)

    # Part 1: Analyze silence prevalence in datasets
    print("\n" + "-" * 70)
    print("Part 1: Silence Prevalence Analysis")
    print("-" * 70)

    print("\nAnalyzing In-the-Wild real audio...")
    itw_real_stats = analyze_silence_in_dataset(ITW_AUDIO_REAL, max_files=200)
    if itw_real_stats:
        print(f"  Files analyzed: {itw_real_stats['files_analyzed']}")
        print(f"  Mean RMS energy: {itw_real_stats['mean_energy']:.4f}")
        print(f"  Median RMS energy: {itw_real_stats['median_energy']:.4f}")
        print(f"  Silent segments (<0.01 RMS): {itw_real_stats['silent_segments']}/{itw_real_stats['total_segments']} ({itw_real_stats['silent_ratio']*100:.1f}%)")
        print(f"  1st percentile energy: {itw_real_stats['percentile_1']:.4f}")
        print(f"  5th percentile energy: {itw_real_stats['percentile_5']:.4f}")
    else:
        print("  Could not analyze (directory not found)")

    print("\nAnalyzing In-the-Wild fake audio...")
    itw_fake_stats = analyze_silence_in_dataset(ITW_AUDIO_FAKE, max_files=200)
    if itw_fake_stats:
        print(f"  Files analyzed: {itw_fake_stats['files_analyzed']}")
        print(f"  Mean RMS energy: {itw_fake_stats['mean_energy']:.4f}")
        print(f"  Median RMS energy: {itw_fake_stats['median_energy']:.4f}")
        print(f"  Silent segments (<0.01 RMS): {itw_fake_stats['silent_segments']}/{itw_fake_stats['total_segments']} ({itw_fake_stats['silent_ratio']*100:.1f}%)")
        print(f"  1st percentile energy: {itw_fake_stats['percentile_1']:.4f}")
        print(f"  5th percentile energy: {itw_fake_stats['percentile_5']:.4f}")
    else:
        print("  Could not analyze (directory not found)")

    # Part 2: Test impact of filtering by segment_count (proxy for content density)
    print("\n" + "-" * 70)
    print("Part 2: Impact of Segment Count Filtering on Classification")
    print("-" * 70)

    print("\nLoading training data...")
    X_train, y_train, feature_cols = load_training_data()
    print(f"  Training samples: {len(X_train)}")

    print("\nLoading evaluation data...")
    eval_df, eval_feature_cols = load_eval_data()
    common_features = [c for c in feature_cols if c in eval_feature_cols]

    print(f"  Evaluation samples: {len(eval_df)}")
    if "segment_count" in eval_df.columns:
        print(f"  segment_count range: {eval_df['segment_count'].min()} - {eval_df['segment_count'].max()}")
        print(f"  segment_count median: {eval_df['segment_count'].median()}")

    results = []

    # Baseline: no filtering
    print("\nEvaluating baseline (no segment filtering)...")
    X_eval = eval_df[common_features]
    y_eval = eval_df["label"].values
    baseline = evaluate_with_segment_filter(X_train, y_train, X_eval, y_eval, common_features)
    baseline["config"] = "Baseline (all)"
    baseline["n_samples"] = len(eval_df)
    results.append(baseline)
    print(f"  F1: {baseline['f1']:.4f}, AUC: {baseline['auc']:.4f}")

    # Test different minimum segment thresholds
    if "segment_count" in eval_df.columns:
        for min_seg in [4, 5, 6, 8, 10]:
            filtered = filter_by_segment_count(eval_df, min_seg)
            if len(filtered) < 100:
                continue

            print(f"\nEvaluating with misegment_count >= {min_seg}...")
            X_eval_filt = filtered[common_features]
            y_eval_filt = filtered["label"].values

            res = evaluate_with_segment_filter(X_train, y_train, X_eval_filt, y_eval_filt, common_features)
            res["config"] = f"min_seg >= {min_seg}"
            res["n_samples"] = len(filtered)
            results.append(res)
            print(f"  Samples: {len(filtered)} ({len(filtered)/len(eval_df)*100:.1f}% of original)")
            print(f"  F1: {res['f1']:.4f}, AUC: {res['auc']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<20} {'Samples':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<20} {r['n_samples']:<10} {r['f1']:.4f}     {r['auc']:.4f}")

    # Key finding
    if len(results) > 1:
        best = max(results, key=lambda x: x['f1'])
        baseline_f1 = results[0]['f1']
        print(f"\nBest by F1: {best['config']} (F1={best['f1']:.4f})")
        if best['f1'] > baseline_f1:
            print(f"  Improvement: +{(best['f1'] - baseline_f1)*100:.2f} percentage points")
        else:
            print("  No improvement from segment filtering")

    # Conclusion for reviewer
    print("\n" + "=" * 70)
    print("CONCLUSION FOR REVIEWER")
    print("=" * 70)
    if itw_real_stats and itw_fake_stats:
        real_silent = itw_real_stats['silent_ratio'] * 100
        fake_silent = itw_fake_stats['silent_ratio'] * 100
        print(f"Silent segment prevalence: Real={real_silent:.1f}%, Fake={fake_silent:.1f}%")
        if real_silent < 5 and fake_silent < 5:
            print("Finding: Silence is rare (<5% of segments) in both classes.")
            print("Explicit silence filtering would have minimal impact.")
        else:
            print("Finding: Non-trivial silence prevalence detected.")
            print("Silence filtering may warrant further investigation.")

    if len(results) > 1:
        improvement = max(r['f1'] for r in results) - results[0]['f1']
        if improvement < 0.01:
            print("Classification impact: Segment count filtering shows <1pp improvement.")
            print("Conclusion: Current approach (no silence filtering) is adequate.")
        else:
            print(f"Classification impact: +{improvement*100:.1f}pp improvement possible with filtering.")


if __name__ == "__main__":
    main()
