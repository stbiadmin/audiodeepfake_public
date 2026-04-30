#!/usr/bin/env python3
"""Evaluate temporal coherence pipeline with WavLM embeddings.

Compares CLAP vs WavLM-derived temporal features by training XGBoost classifiers
on each and evaluating on In-the-Wild and ASVspoof5 datasets.

This answers: does temporal coherence analysis work better with speech-specific
SSL embeddings (WavLM) or general-purpose CLAP?

Usage:
    python sota_benchmarking/run_ssl_temporal_eval.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.data_loader import ALL_FEATURES

# Dataset Paths
# Training data: features extracted from training datasets using different embedding models
TRAINING_FEATURES = {
    'msclap': {
        'single_voice': PROJECT_ROOT / 'data' / 'features' / 'msclap' / 'combined' / 'single_voice_combined.json',
        'deepspeak_v2_train': PROJECT_ROOT / 'data' / 'features' / 'msclap' / 'combined' / 'deepspeak_v2_train_combined.json',
    },
    'wavlm': {
        'single_voice': PROJECT_ROOT / 'data' / 'features' / 'wavlm' / 'combined' / 'single_voice_combined.json',
        'deepspeak_v2_train': PROJECT_ROOT / 'data' / 'features' / 'wavlm' / 'combined' / 'deepspeak_v2_train_combined.json',
    },
    'aasist': {
        'single_voice': PROJECT_ROOT / 'data' / 'features' / 'aasist' / 'combined' / 'single_voice_combined.json',
        'deepspeak_v2_train': PROJECT_ROOT / 'data' / 'features' / 'aasist' / 'combined' / 'deepspeak_v2_train_combined.json',
    },
}

# Eval data: pre-extracted temporal features
EVAL_FEATURES = {
    'msclap': {
        'in_the_wild': PROJECT_ROOT / 'data' / 'eval_features' / 'msclap' / 'in_the_wild' / 'features_combined.json',
        'asvspoof5': PROJECT_ROOT / 'data' / 'eval_features' / 'msclap' / 'asvspoof5' / 'features_combined.json',
    },
    'wavlm': {
        'in_the_wild': PROJECT_ROOT / 'data' / 'eval_features' / 'wavlm' / 'in_the_wild' / 'features_combined.json',
        'asvspoof5': PROJECT_ROOT / 'data' / 'eval_features' / 'wavlm' / 'asvspoof5' / 'features_combined.json',
    },
    'aasist': {
        'in_the_wild': PROJECT_ROOT / 'data' / 'eval_features' / 'aasist' / 'in_the_wild' / 'features_combined.json',
        'asvspoof5': PROJECT_ROOT / 'data' / 'eval_features' / 'aasist' / 'asvspoof5' / 'features_combined.json',
    },
}

OUTPUT_DIR = PROJECT_ROOT / 'sota_benchmarking' / 'results' / 'ssl_temporal'


def load_features(path, fmt='auto'):
    """Load features from JSON, handling flat and nested formats."""
    with open(path, 'r') as f:
        data = json.load(f)

    if not data:
        return pd.DataFrame(), np.array([])

    # Detect format
    if isinstance(data[0], dict) and 'features' in data[0]:
        # Nested format: {'features': {...}, 'label': ...}
        rows = []
        for entry in data:
            row = dict(entry['features'])
            row['label'] = entry['label']
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        # Flat format: features as top-level keys
        df = pd.DataFrame(data)

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    X = df[ALL_FEATURES].copy().fillna(df[ALL_FEATURES].median())
    y = df['label'].values
    return X, y


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = float('nan')
    return float(eer)


def train_and_evaluate(X_train, y_train, X_eval, y_eval, model_name="XGBoost"):
    """Train XGBoost and evaluate, returning metrics dict."""
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_eval_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_eval, y_pred)),
        'precision': float(precision_score(y_eval, y_pred, zero_division=0)),
        'recall': float(recall_score(y_eval, y_pred, zero_division=0)),
        'f1': float(f1_score(y_eval, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_eval, y_proba)) if len(np.unique(y_eval)) > 1 else 0.0,
        'eer': compute_eer(y_eval, y_proba),
    }

    return metrics, model, scaler


def main():
    print("=" * 70)
    print("SSL Temporal Coherence Evaluation: CLAP vs WavLM Embeddings")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for embedding_model in ['msclap', 'wavlm', 'aasist']:
        print(f"\n{'─' * 60}")
        print(f"Embedding Model: {embedding_model.upper()}")
        print(f"{'─' * 60}")

        # Check available training data
        train_paths = TRAINING_FEATURES.get(embedding_model, {})
        available_train = {k: v for k, v in train_paths.items() if v.exists()}

        if not available_train:
            print(f"  No training data found for {embedding_model}, skipping.")
            continue

        # Load and combine training data
        X_trains = []
        y_trains = []
        for dtype, path in available_train.items():
            print(f"  Loading training data: {dtype}...")
            X, y = load_features(path)
            if len(X) > 0:
                X_trains.append(X)
                y_trains.append(y)
                print(f"    {len(y)} samples (real={np.sum(y==0)}, fake={np.sum(y==1)})")

        if not X_trains:
            print(f"  No valid training data for {embedding_model}, skipping.")
            continue

        X_train_all = pd.concat(X_trains, ignore_index=True)
        y_train_all = np.concatenate(y_trains)
        print(f"  Combined training: {len(y_train_all)} samples "
              f"(real={np.sum(y_train_all==0)}, fake={np.sum(y_train_all==1)})")

        # Balance classes via undersampling
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        indices = np.arange(len(y_train_all)).reshape(-1, 1)
        idx_resampled, y_balanced = rus.fit_resample(indices, y_train_all)
        X_balanced = X_train_all.iloc[idx_resampled.ravel()].reset_index(drop=True)
        print(f"  After balancing: {len(y_balanced)} samples "
              f"(real={np.sum(y_balanced==0)}, fake={np.sum(y_balanced==1)})")

        # Evaluate on each eval dataset
        eval_paths = EVAL_FEATURES.get(embedding_model, {})
        for eval_name, eval_path in eval_paths.items():
            if not eval_path.exists():
                print(f"  Eval {eval_name}: NOT FOUND, skipping.")
                continue

            print(f"\n  Evaluating on {eval_name}...")
            X_eval, y_eval = load_features(eval_path)
            print(f"    {len(y_eval)} samples (real={np.sum(y_eval==0)}, fake={np.sum(y_eval==1)})")

            metrics, model, scaler = train_and_evaluate(
                X_balanced, y_balanced, X_eval, y_eval
            )

            print(f"    F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  "
                  f"EER={metrics['eer']:.3f}  Acc={metrics['accuracy']:.3f}")

            all_results.append({
                'embedding_model': embedding_model,
                'eval_dataset': eval_name,
                'n_train': len(y_balanced),
                'n_eval': len(y_eval),
                **metrics,
            })

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Temporal Coherence with Different Embeddings")
    print(f"{'=' * 70}")

    print(f"\n{'Embedding':<12} {'Eval Dataset':<15} {'F1':>7} {'AUC':>7} {'EER':>7} {'Acc':>7} {'N_eval':>7}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['embedding_model']:<12} {r['eval_dataset']:<15} "
              f"{r['f1']:>7.3f} {r['auc']:>7.3f} {r['eer']:>7.3f} "
              f"{r['accuracy']:>7.3f} {r['n_eval']:>7}")

    # Save results
    df_results = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / 'ssl_temporal_comparison.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Text report
    report_path = OUTPUT_DIR / 'SSL_TEMPORAL_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write("SSL Temporal Coherence Evaluation: CLAP vs WavLM\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Question: Does temporal coherence work better with speech-specific SSL\n")
        f.write("embeddings (WavLM) or general-purpose audio embeddings (CLAP)?\n\n")
        f.write(f"{'Embedding':<12} {'Eval Dataset':<15} {'F1':>7} {'AUC':>7} {'EER':>7} {'Acc':>7}\n")
        f.write("-" * 60 + "\n")
        for r in all_results:
            f.write(f"{r['embedding_model']:<12} {r['eval_dataset']:<15} "
                    f"{r['f1']:>7.3f} {r['auc']:>7.3f} {r['eer']:>7.3f} {r['accuracy']:>7.3f}\n")
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
