#!/usr/bin/env python3
"""Cross-dataset evaluation of the paper's 5-Expert MoE ensemble.

Evaluates the same MoE used as the paper's SOTA result on all available
datasets, for comparison with naive embeddings and perceptual straightening
cross-dataset results.

MoE Configuration:
    ds_msclap (0.30) + audeter_msclap (0.30) + sv_ds_msclap (0.20) +
    sv_msclap (0.10) + mlaad_msclap (0.10), threshold=0.30

Usage:
    python sota_benchmarking/run_paper_moe_crosseval.py
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.data_loader import ALL_FEATURES

# Per-model scaler feature sets (models were trained with different versions)
# 27 features: ALL_FEATURES without shapiro_p, normaltest_p
FEATURES_27 = [f for f in ALL_FEATURES if f not in ['shapiro_p', 'normaltest_p']]
# 29 features: current ALL_FEATURES
FEATURES_29 = list(ALL_FEATURES)
# 30 features: ALL_FEATURES + n_samples (mlaad used all features)
FEATURES_30 = list(ALL_FEATURES) + ['n_samples'] if 'n_samples' not in ALL_FEATURES else list(ALL_FEATURES)
# Actually check mlaad's exact order
MLAAD_FEATURES = ['mean', 'std', 'variance', 'min', 'max', 'peak_to_peak', 'skewness', 'kurtosis',
                  'bimodality_coefficient', 'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr',
                  'tail_weight_ratio', 'trimmed_mean', 'entropy', 'gini_coefficient',
                  'coefficient_of_variation', 'variance_mean_ratio', 'kurtosis_variance_ratio',
                  'skewness_kurtosis_ratio', 'iqr_range_ratio', 'median_mean_diff', 'n_samples',
                  'shapiro_p', 'normaltest_p']

# Map each model to the feature set its scaler was trained on
SCALER_FEATURES = {
    'ds_msclap': FEATURES_29,
    'audeter_msclap': FEATURES_29,
    'sv_ds_msclap': FEATURES_27,
    'sv_msclap': FEATURES_29,
    'mlaad_msclap': MLAAD_FEATURES,
}

# MoE Configuration (from inference/model_registry.py)
MODELS_DIR = PROJECT_ROOT / "models" / "trained"

MOE_CONFIG = {
    'name': '5_expert_moe',
    'experts': ['ds_msclap', 'audeter_msclap', 'sv_ds_msclap', 'sv_msclap', 'mlaad_msclap'],
    'weights': [0.30, 0.30, 0.20, 0.10, 0.10],
    'threshold': 0.30,
}

# Evaluation Datasets
# Matches the datasets used in naive embeddings / perceptual straightening
# cross-dataset evaluation

# Non-overlapping eval sets only (none used in MoE training)
EVAL_DATASETS = [
    {
        'name': 'in_the_wild',
        'path': PROJECT_ROOT / 'data' / 'eval_features' / 'msclap' / 'in_the_wild' / 'features_combined.json',
        'format': 'flat',  # features as top-level keys
    },
    {
        'name': 'fakeavceleb',
        'path': PROJECT_ROOT / 'data' / 'features' / 'msclap' / 'combined' / 'fakeavceleb_combined.json',
        'format': 'nested',
    },
    {
        'name': 'spoofceleb',
        'path': PROJECT_ROOT / 'data' / 'features' / 'msclap' / 'combined' / 'spoofceleb_combined.json',
        'format': 'nested',
    },
    {
        'name': 'asvspoof5',
        'path': PROJECT_ROOT / 'data' / 'eval_features' / 'msclap' / 'asvspoof5' / 'features_combined.json',
        'format': 'flat',
    },
]


def load_models():
    """Load all 5 expert models."""
    models = []
    for i, expert_name in enumerate(MOE_CONFIG['experts']):
        model_path = MODELS_DIR / f"{expert_name}_model.pkl"
        if not model_path.exists():
            print(f"  ERROR: Model not found: {model_path}")
            sys.exit(1)

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        models.append({
            'name': expert_name,
            'model': data['model'],
            'features': data['features'],
            'scaler': data.get('scaler'),
            'weight': MOE_CONFIG['weights'][i],
        })
        print(f"  Loaded {expert_name}: {len(data['features'])} features, "
              f"weight={MOE_CONFIG['weights'][i]:.2f}, "
              f"scaler={'yes' if data.get('scaler') else 'no'}")

    return models


def moe_predict_proba(models, X_all_features):
    """Get weighted average probabilities from MoE ensemble.

    Args:
        models: List of model dicts
        X_all_features: DataFrame with all available feature columns

    Returns:
        Array of fake-class probabilities
    """
    weighted_probas = np.zeros(len(X_all_features))

    for model_info in models:
        scaler_feats = SCALER_FEATURES.get(model_info['name'])

        if model_info['scaler'] is not None and scaler_feats is not None:
            # Use the feature set this model's scaler was trained on
            available = [f for f in scaler_feats if f in X_all_features.columns]
            X_for_scaler = X_all_features[available].copy().fillna(0)

            # If some scaler features are missing from data, add them as 0
            for f in scaler_feats:
                if f not in X_for_scaler.columns:
                    X_for_scaler[f] = 0.0
            X_for_scaler = X_for_scaler[scaler_feats]  # enforce order

            X_scaled = model_info['scaler'].transform(X_for_scaler.values)
            X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_feats, index=X_all_features.index)
            X_model = X_scaled_df[model_info['features']]
        else:
            X_model = X_all_features[model_info['features']].copy().fillna(0)

        probas = model_info['model'].predict_proba(X_model)[:, 1]
        weighted_probas += model_info['weight'] * probas

    return weighted_probas


def load_dataset(config):
    """Load a dataset and return (X_all_features, y) DataFrame.

    Handles three formats:
    - 'flat': features as top-level keys (eval features)
    - 'nested': features under 'features' dict key (training combined JSONs)
    - 'split': separate real.json/fake.json files with nested features
    """
    fmt = config['format']
    path = config['path']

    if fmt == 'flat':
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        X = df[ALL_FEATURES].copy().fillna(df[ALL_FEATURES].median())
        y = df['label'].values
        return X, y

    elif fmt == 'nested':
        with open(path, 'r') as f:
            data = json.load(f)
        rows = []
        for entry in data:
            row = dict(entry['features'])
            row['label'] = entry['label']
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        X = df[ALL_FEATURES].copy().fillna(df[ALL_FEATURES].median())
        y = df['label'].values
        return X, y

    elif fmt == 'split':
        all_rows = []
        for label, fname in [(0, 'real.json'), (1, 'fake.json')]:
            fpath = path / fname
            if not fpath.exists():
                print(f"    WARNING: {fpath} not found, skipping")
                continue
            with open(fpath, 'r') as f:
                data = json.load(f)
            for entry in data:
                row = dict(entry['features'])
                row['label'] = label
                all_rows.append(row)
        df = pd.DataFrame(all_rows)
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        X = df[ALL_FEATURES].copy().fillna(df[ALL_FEATURES].median())
        y = df['label'].values
        return X, y

    else:
        raise ValueError(f"Unknown format: {fmt}")


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate (EER) from true labels and predicted scores."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = float('nan')
    return float(eer)


def evaluate(models, X, y, threshold=0.30):
    """Evaluate MoE ensemble on a dataset."""
    y_proba = moe_predict_proba(models, X)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        'eer': compute_eer(y, y_proba),
    }

    cm = confusion_matrix(y, y_pred)
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])

    return metrics


def main():
    print("=" * 70)
    print("Paper's 5-Expert MoE Ensemble: Cross-Dataset Evaluation")
    print("=" * 70)
    print(f"\nExperts: {MOE_CONFIG['experts']}")
    print(f"Weights: {MOE_CONFIG['weights']}")
    print(f"Threshold: {MOE_CONFIG['threshold']}")
    print()

    # Load models
    print("Loading models...")
    models = load_models()
    print(f"\nLoaded {len(models)} expert models.\n")

    # Evaluate on each dataset
    results = []
    print("-" * 70)

    for ds_config in EVAL_DATASETS:
        name = ds_config['name']
        path = ds_config['path']

        # Check if path exists
        if isinstance(path, Path):
            if path.is_file() and not path.exists():
                print(f"\n{name}: SKIPPED (file not found: {path})")
                continue
            elif path.is_dir() and not path.exists():
                print(f"\n{name}: SKIPPED (dir not found: {path})")
                continue

        try:
            X, y = load_dataset(ds_config)
        except Exception as e:
            print(f"\n{name}: ERROR loading data: {e}")
            continue

        n_real = int(np.sum(y == 0))
        n_fake = int(np.sum(y == 1))

        metrics = evaluate(models, X, y, threshold=MOE_CONFIG['threshold'])

        print(f"\n{name}:")
        print(f"  Samples: {len(y)} (real={n_real}, fake={n_fake})")
        print(f"  F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  "
              f"EER={metrics['eer']:.3f}  Acc={metrics['accuracy']:.3f}  "
              f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}")
        print(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} "
              f"FN={metrics['fn']} TN={metrics['tn']}")

        results.append({
            'dataset': name,
            'n_samples': len(y),
            'n_real': n_real,
            'n_fake': n_fake,
            **metrics,
        })

    # Output results
    print("\n" + "=" * 70)
    print("SUMMARY: 5-Expert MoE Cross-Dataset Results")
    print("=" * 70)

    print(f"\n{'Dataset':<25} {'N':>6} {'F1':>7} {'AUC':>7} {'EER':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7}")
    print("-" * 78)
    for r in results:
        print(f"{r['dataset']:<25} {r['n_samples']:>6} {r['f1']:>7.3f} {r['auc']:>7.3f} "
              f"{r['eer']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['accuracy']:>7.3f}")

    # Save results
    output_dir = PROJECT_ROOT / 'sota_benchmarking' / 'results' / 'paper_moe'
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df_results = pd.DataFrame(results)
    csv_path = output_dir / 'moe_crosseval_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Text report
    report_path = output_dir / 'MOE_CROSSEVAL_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write("5-Expert MoE Ensemble: Cross-Dataset Evaluation\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Experts: {MOE_CONFIG['experts']}\n")
        f.write(f"Weights: {MOE_CONFIG['weights']}\n")
        f.write(f"Threshold: {MOE_CONFIG['threshold']}\n\n")
        f.write(f"{'Dataset':<25} {'N':>6} {'F1':>7} {'AUC':>7} {'EER':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7}\n")
        f.write("-" * 78 + "\n")
        for r in results:
            f.write(f"{r['dataset']:<25} {r['n_samples']:>6} {r['f1']:>7.3f} {r['auc']:>7.3f} "
                    f"{r['eer']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['accuracy']:>7.3f}\n")
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
