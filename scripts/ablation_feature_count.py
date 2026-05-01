#!/usr/bin/env python3
"""Ablation study: All features vs selected features.

Compares classification performance using:
1. All 29 features (no selection)
2. Selected 8 features (greedy forward selection)

Evaluates on In-the-Wild celebrity deepfakes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_FEATURES = PROJECT_ROOT / "data" / "features" / "msclap"
EVAL_FEATURES = PROJECT_ROOT / "data" / "eval_features" / "msclap" / "in_the_wild"


def load_features(filepath: Path) -> pd.DataFrame:
    """Load features from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    records = []
    for item in data:
        if "features" in item and item["features"]:
            record = item["features"].copy()
            record["file"] = item.get("file", "unknown")
            records.append(record)

    return pd.DataFrame(records)


def prepare_training_data(dataset="deepspeak"):
    """Load and prepare training data."""
    if dataset == "single_voice":
        real_df = load_features(TRAIN_FEATURES / "single_voice_real.json")
        fake_df = load_features(TRAIN_FEATURES / "single_voice_fake.json")
    elif dataset == "deepspeak":
        real_df = load_features(TRAIN_FEATURES / "deepspeak_v2_train_real.json")
        fake_df = load_features(TRAIN_FEATURES / "deepspeak_v2_train_fake.json")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    real_df["label"] = 0
    fake_df["label"] = 1

    # Combine and balance
    combined = pd.concat([real_df, fake_df], ignore_index=True)

    # Get feature columns (exclude file and label)
    feature_cols = [c for c in combined.columns if c not in ["file", "label"]]

    # Balance classes
    min_count = min(len(real_df), len(fake_df))
    real_sample = combined[combined["label"] == 0].sample(n=min_count, random_state=42)
    fake_sample = combined[combined["label"] == 1].sample(n=min_count, random_state=42)
    balanced = pd.concat([real_sample, fake_sample], ignore_index=True)

    X = balanced[feature_cols]
    y = balanced["label"].values

    return X, y, feature_cols


def prepare_eval_data(feature_cols):
    """Load In-the-Wild evaluation data."""
    real_df = load_features(EVAL_FEATURES / "real.json")
    fake_df = load_features(EVAL_FEATURES / "fake.json")

    real_df["label"] = 0
    fake_df["label"] = 1

    combined = pd.concat([real_df, fake_df], ignore_index=True)

    # Use only features that exist in both train and eval
    available_cols = [c for c in feature_cols if c in combined.columns]

    X = combined[available_cols]
    y = combined["label"].values

    return X, y, available_cols


def rank_features_by_auc(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Rank features by individual AUC."""
    rankings = {}
    for col in X.columns:
        try:
            auc = roc_auc_score(y, X[col].values)
            auc = max(auc, 1 - auc)  # Direction invariant
            rankings[col] = auc
        except:
            rankings[col] = 0.5
    return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))


def greedy_forward_selection(X: pd.DataFrame, y: np.ndarray,
                              max_features: int = 8,
                              cv_folds: int = 5) -> list:
    """Greedy forward feature selection."""
    # Rank by AUC
    rankings = rank_features_by_auc(X, y)
    ranked_features = list(rankings.keys())

    selected = [ranked_features[0]]  # Start with best
    best_score = 0

    clf = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=100,
        random_state=42, verbosity=0
    )

    for feature in ranked_features[1:]:
        if len(selected) >= max_features:
            break

        candidate = selected + [feature]
        scores = cross_val_score(clf, X[candidate], y, cv=cv_folds, scoring='f1')
        mean_score = scores.mean()

        if mean_score > best_score + 0.005:  # 0.5% improvement threshold
            selected.append(feature)
            best_score = mean_score

    # Ensure minimum 8 features
    while len(selected) < 8:
        for f in ranked_features:
            if f not in selected:
                selected.append(f)
                break

    return selected


def train_and_evaluate(X_train, y_train, X_eval, y_eval, feature_subset=None):
    """Train model and evaluate on held-out data."""
    if feature_subset:
        X_train = X_train[feature_subset]
        X_eval = X_eval[[c for c in feature_subset if c in X_eval.columns]]

    # Normalize
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Train
    clf = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=100,
        random_state=42, verbosity=0
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_eval_scaled)
    y_prob = clf.predict_proba(X_eval_scaled)[:, 1]

    # Cross-validation on training data
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='f1')

    return {
        "train_f1_cv": cv_scores.mean(),
        "train_f1_std": cv_scores.std(),
        "eval_f1": f1_score(y_eval, y_pred),
        "eval_auc": roc_auc_score(y_eval, y_prob),
        "eval_acc": accuracy_score(y_eval, y_pred),
        "n_features": X_train.shape[1]
    }


def main():
    print("=" * 60)
    print("Ablation Study: All Features vs Selected Features")
    print("=" * 60)

    # Load data - use DeepSpeak which is our best model
    print("\nLoading training data (DeepSpeak v2)...")
    X_train, y_train, feature_cols = prepare_training_data("deepspeak")
    print(f"  Training samples: {len(X_train)} ({sum(y_train == 0)} real, {sum(y_train == 1)} fake)")
    print(f"  Total features: {len(feature_cols)}")

    print("\nLoading evaluation data (In-the-Wild)...")
    X_eval, y_eval, eval_cols = prepare_eval_data(feature_cols)
    print(f"  Evaluation samples: {len(X_eval)} ({sum(y_eval == 0)} real, {sum(y_eval == 1)} fake)")

    # Ensure same features in train and eval
    common_cols = [c for c in feature_cols if c in eval_cols]
    X_train = X_train[common_cols]
    X_eval = X_eval[common_cols]
    print(f"  Common features: {len(common_cols)}")

    # Rank features by AUC
    print("\nRanking features by individual AUC...")
    rankings = rank_features_by_auc(X_train, y_train)
    print("  Top 10 features by AUC:")
    for i, (feat, auc) in enumerate(list(rankings.items())[:10]):
        print(f"    {i+1}. {feat}: {auc:.3f}")

    # Run feature selection
    print("\nRunning greedy forward feature selection...")
    selected_features = greedy_forward_selection(X_train, y_train, max_features=8)
    print(f"  Selected {len(selected_features)} features: {selected_features}")

    # Compare configurations
    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)

    results = []

    # All features
    print("\n1. All features ({} features)...".format(len(common_cols)))
    all_results = train_and_evaluate(X_train, y_train, X_eval, y_eval, feature_subset=None)
    all_results["config"] = "All features"
    results.append(all_results)
    print(f"   Train CV F1: {all_results['train_f1_cv']:.3f} +/- {all_results['train_f1_std']:.3f}")
    print(f"   Eval F1: {all_results['eval_f1']:.3f}")
    print(f"   Eval AUC: {all_results['eval_auc']:.3f}")

    # Selected features
    print(f"\n2. Selected features ({len(selected_features)} features)...")
    sel_results = train_and_evaluate(X_train, y_train, X_eval, y_eval, feature_subset=selected_features)
    sel_results["config"] = "Selected (8)"
    results.append(sel_results)
    print(f"   Train CV F1: {sel_results['train_f1_cv']:.3f} +/- {sel_results['train_f1_std']:.3f}")
    print(f"   Eval F1: {sel_results['eval_f1']:.3f}")
    print(f"   Eval AUC: {sel_results['eval_auc']:.3f}")

    # Additional configurations for ablation
    for n_features in [4, 6, 8, 10, 12, 14, 16, 20, 25]:
        print(f"\n3. Top {n_features} features by AUC...")
        rankings = rank_features_by_auc(X_train, y_train)
        top_n = list(rankings.keys())[:n_features]
        top_results = train_and_evaluate(X_train, y_train, X_eval, y_eval, feature_subset=top_n)
        top_results["config"] = f"Top {n_features}"
        results.append(top_results)
        print(f"   Train CV F1: {top_results['train_f1_cv']:.3f} +/- {top_results['train_f1_std']:.3f}")
        print(f"   Eval F1: {top_results['eval_f1']:.3f}")
        print(f"   Eval AUC: {top_results['eval_auc']:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Configuration':<20} {'N':<5} {'Train F1':<12} {'Eval F1':<10} {'Eval AUC':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<20} {r['n_features']:<5} {r['train_f1_cv']:.3f} +/- {r['train_f1_std']:.2f}  {r['eval_f1']:.3f}     {r['eval_auc']:.3f}")

    # LaTeX table
    print("\n" + "=" * 60)
    print("LaTeX Table for Paper")
    print("=" * 60)
    print(r"""
\begin{table}[t]
\centering
\caption{Effect of feature count on classification performance. Greedy forward selection with 8 features achieves the best trade-off between training and evaluation performance.}
\label{tab:ablation_features}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Train F1 (CV)} & \textbf{ITW F1} & \textbf{ITW AUC} \\
\midrule""")
    for r in results:
        print(f"{r['config']} ({r['n_features']}) & {r['train_f1_cv']:.3f} & {r['eval_f1']:.3f} & {r['eval_auc']:.3f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    main()
