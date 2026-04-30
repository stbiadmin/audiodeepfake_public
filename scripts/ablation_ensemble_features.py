#!/usr/bin/env python3
"""Ablation study: Ensemble with different feature selection methods.

Compares 5-expert ensemble performance using:
1. Greedy forward selection (8 features) - original approach
2. Top 4 features by AUC
3. Top 16 features by AUC

Evaluates on In-the-Wild celebrity deepfakes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from typing import Dict, List, Tuple

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_FEATURES = PROJECT_ROOT / "data" / "features" / "msclap"
EVAL_FEATURES = PROJECT_ROOT / "data" / "eval_features" / "msclap" / "in_the_wild"

# Expert configurations (dataset -> file paths)
EXPERT_CONFIGS = {
    "ds": {
        "real": "deepspeak_v2_train_real.json",
        "fake": "deepspeak_v2_train_fake.json",
    },
    "sv": {
        "real": "single_voice_real.json",
        "fake": "single_voice_fake.json",
    },
    "mlaad": {
        "real": "mlaad_english_real.json",
        "fake": "mlaad_english_fake.json",
    },
    "audeter": {
        "real": "../msclap_optc/audeter_real.json",
        "fake": "../msclap_optc/audeter_fake.json",
    },
}

# Ensemble weights from experiment log
ENSEMBLE_WEIGHTS = {
    "ds": 0.30,
    "sv": 0.10,
    "sv_ds": 0.20,
    "mlaad": 0.10,
    "audeter": 0.30,
}

ENSEMBLE_THRESHOLD = 0.30


def load_features(filepath: Path) -> pd.DataFrame:
    """Load features from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    records = []
    for item in data:
        if "features" in item and item["features"]:
            record = item["features"].copy()
            record["file"] = item.get("file", item.get("file_path", "unknown"))
            records.append(record)

    return pd.DataFrame(records)


def load_training_data(config: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and prepare training data for an expert."""
    real_path = TRAIN_FEATURES / config["real"]
    fake_path = TRAIN_FEATURES / config["fake"]

    if not real_path.exists() or not fake_path.exists():
        return None, None

    real_df = load_features(real_path)
    fake_df = load_features(fake_path)

    real_df["label"] = 0
    fake_df["label"] = 1

    # Combine and balance
    combined = pd.concat([real_df, fake_df], ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ["file", "label"]]

    # Balance classes
    min_count = min(len(real_df), len(fake_df))
    real_sample = combined[combined["label"] == 0].sample(n=min_count, random_state=42)
    fake_sample = combined[combined["label"] == 1].sample(n=min_count, random_state=42)
    balanced = pd.concat([real_sample, fake_sample], ignore_index=True)

    X = balanced[feature_cols]
    y = balanced["label"].values

    return X, y


def load_eval_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load In-the-Wild evaluation data."""
    real_df = load_features(EVAL_FEATURES / "real.json")
    fake_df = load_features(EVAL_FEATURES / "fake.json")

    real_df["label"] = 0
    fake_df["label"] = 1

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ["file", "label"]]

    X = combined[feature_cols]
    y = combined["label"].values

    return X, y


def rank_features_by_auc(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Rank features by individual AUC."""
    rankings = {}
    for col in X.columns:
        try:
            auc = roc_auc_score(y, X[col].values)
            auc = max(auc, 1 - auc)
            rankings[col] = auc
        except:
            rankings[col] = 0.5
    return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))


def greedy_forward_selection(X: pd.DataFrame, y: np.ndarray, max_features: int = 8) -> list:
    """Greedy forward feature selection."""
    rankings = rank_features_by_auc(X, y)
    ranked_features = list(rankings.keys())

    selected = [ranked_features[0]]
    best_score = 0

    clf = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=100,
        random_state=42, verbosity=0
    )

    for feature in ranked_features[1:]:
        if len(selected) >= max_features:
            break

        candidate = selected + [feature]
        scores = cross_val_score(clf, X[candidate], y, cv=5, scoring='f1')
        mean_score = scores.mean()

        if mean_score > best_score + 0.005:
            selected.append(feature)
            best_score = mean_score

    while len(selected) < 8:
        for f in ranked_features:
            if f not in selected:
                selected.append(f)
                break

    return selected


def train_expert(X: pd.DataFrame, y: np.ndarray, features: list) -> Tuple:
    """Train an expert model on selected features."""
    X_subset = X[features]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_subset)

    clf = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=100,
        random_state=42, verbosity=0
    )
    clf.fit(X_scaled, y)

    return clf, scaler, features


def predict_expert(clf, scaler, features, X_eval: pd.DataFrame) -> np.ndarray:
    """Get predictions from an expert."""
    available_features = [f for f in features if f in X_eval.columns]
    X_subset = X_eval[available_features]
    X_scaled = scaler.transform(X_subset)
    return clf.predict_proba(X_scaled)[:, 1]


def evaluate_ensemble(expert_probs: Dict[str, np.ndarray], y_true: np.ndarray,
                      weights: dict, threshold: float) -> dict:
    """Evaluate weighted ensemble."""
    # Compute weighted average
    p_fake = np.zeros(len(y_true))
    for name, probs in expert_probs.items():
        if name in weights:
            p_fake += weights[name] * probs
        elif name == "sv_ds":
            p_fake += weights.get("sv_ds", 0.20) * probs

    # Apply threshold
    y_pred = (p_fake > threshold).astype(int)

    # Compute metrics
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, p_fake)
    acc = accuracy_score(y_true, y_pred)

    # Detection rates
    fake_mask = y_true == 1
    real_mask = y_true == 0
    fake_det = y_pred[fake_mask].mean() if fake_mask.sum() > 0 else 0
    real_det = (1 - y_pred[real_mask]).mean() if real_mask.sum() > 0 else 0

    return {
        "f1": f1,
        "auc": auc,
        "accuracy": acc,
        "fake_detection": fake_det,
        "real_detection": real_det,
    }


def run_experiment(feature_method: str, n_features: int = None):
    """Run full experiment with specified feature selection method."""
    print(f"\n{'='*60}")
    print(f"Feature Method: {feature_method}" + (f" (N={n_features})" if n_features else ""))
    print(f"{'='*60}")

    # Load eval data
    X_eval, y_eval = load_eval_data()
    print(f"Evaluation data: {len(X_eval)} samples")

    # Train each expert
    expert_models = {}
    expert_probs = {}

    for expert_name, config in EXPERT_CONFIGS.items():
        print(f"\nTraining {expert_name}...")
        X_train, y_train = load_training_data(config)

        if X_train is None:
            print(f"  Skipping {expert_name} - data not found")
            continue

        print(f"  Training samples: {len(X_train)}")

        # Select features based on method
        common_features = [c for c in X_train.columns if c in X_eval.columns]
        X_train = X_train[common_features]

        if feature_method == "greedy":
            features = greedy_forward_selection(X_train, y_train, max_features=8)
        elif feature_method == "top_n":
            rankings = rank_features_by_auc(X_train, y_train)
            features = list(rankings.keys())[:n_features]
        elif feature_method == "all":
            features = common_features
        else:
            raise ValueError(f"Unknown method: {feature_method}")

        print(f"  Selected {len(features)} features: {features[:5]}...")

        # Train
        clf, scaler, features = train_expert(X_train, y_train, features)
        expert_models[expert_name] = (clf, scaler, features)

        # Get predictions
        probs = predict_expert(clf, scaler, features, X_eval)
        expert_probs[expert_name] = probs

        # Individual performance
        y_pred = (probs > 0.5).astype(int)
        ind_f1 = f1_score(y_eval, y_pred)
        ind_auc = roc_auc_score(y_eval, probs)
        print(f"  Individual F1: {ind_f1:.3f}, AUC: {ind_auc:.3f}")

    # Also create sv_ds combined expert
    if "sv" in expert_models and "ds" in expert_models:
        print(f"\nTraining sv_ds (combined)...")
        # Load both datasets
        X_sv, y_sv = load_training_data(EXPERT_CONFIGS["sv"])
        X_ds, y_ds = load_training_data(EXPERT_CONFIGS["ds"])

        if X_sv is not None and X_ds is not None:
            # Combine
            common_cols = list(set(X_sv.columns) & set(X_ds.columns) & set(X_eval.columns))
            X_combined = pd.concat([X_sv[common_cols], X_ds[common_cols]], ignore_index=True)
            y_combined = np.concatenate([y_sv, y_ds])

            # Balance
            min_count = min((y_combined == 0).sum(), (y_combined == 1).sum())
            idx_real = np.where(y_combined == 0)[0]
            idx_fake = np.where(y_combined == 1)[0]
            np.random.seed(42)
            idx_real = np.random.choice(idx_real, min_count, replace=False)
            idx_fake = np.random.choice(idx_fake, min_count, replace=False)
            idx = np.concatenate([idx_real, idx_fake])
            X_combined = X_combined.iloc[idx]
            y_combined = y_combined[idx]

            print(f"  Training samples: {len(X_combined)}")

            if feature_method == "greedy":
                features = greedy_forward_selection(X_combined, y_combined, max_features=8)
            elif feature_method == "top_n":
                rankings = rank_features_by_auc(X_combined, y_combined)
                features = list(rankings.keys())[:n_features]
            elif feature_method == "all":
                features = common_cols

            print(f"  Selected {len(features)} features")

            clf, scaler, features = train_expert(X_combined, y_combined, features)
            expert_models["sv_ds"] = (clf, scaler, features)

            probs = predict_expert(clf, scaler, features, X_eval)
            expert_probs["sv_ds"] = probs

            y_pred = (probs > 0.5).astype(int)
            ind_f1 = f1_score(y_eval, y_pred)
            ind_auc = roc_auc_score(y_eval, probs)
            print(f"  Individual F1: {ind_f1:.3f}, AUC: {ind_auc:.3f}")

    # Evaluate ensemble
    print(f"\n{'='*60}")
    print("5-Expert Ensemble Results")
    print(f"{'='*60}")

    results = evaluate_ensemble(expert_probs, y_eval, ENSEMBLE_WEIGHTS, ENSEMBLE_THRESHOLD)

    print(f"F1 Score: {results['f1']:.4f}")
    print(f"AUC-ROC: {results['auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Fake Detection: {results['fake_detection']*100:.1f}%")
    print(f"Real Detection: {results['real_detection']*100:.1f}%")

    return results


def main():
    print("=" * 70)
    print("Ensemble Feature Selection Ablation Study")
    print("=" * 70)

    all_results = []

    # Original greedy approach
    results = run_experiment("greedy")
    results["method"] = "Greedy 8"
    all_results.append(results)

    # Top 4 by AUC
    results = run_experiment("top_n", n_features=4)
    results["method"] = "Top 4"
    all_results.append(results)

    # Top 8 by AUC
    results = run_experiment("top_n", n_features=8)
    results["method"] = "Top 8"
    all_results.append(results)

    # Top 16 by AUC
    results = run_experiment("top_n", n_features=16)
    results["method"] = "Top 16"
    all_results.append(results)

    # All features
    results = run_experiment("all")
    results["method"] = "All 30"
    all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: 5-Expert Ensemble Comparison")
    print("=" * 70)
    print(f"{'Method':<15} {'F1':<8} {'AUC':<8} {'Fake Det':<10} {'Real Det':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['method']:<15} {r['f1']:.4f}   {r['auc']:.4f}   {r['fake_detection']*100:>5.1f}%     {r['real_detection']*100:>5.1f}%")

    # Best result
    best = max(all_results, key=lambda x: x['f1'])
    print(f"\nBest by F1: {best['method']} (F1={best['f1']:.4f})")

    best_auc = max(all_results, key=lambda x: x['auc'])
    print(f"Best by AUC: {best_auc['method']} (AUC={best_auc['auc']:.4f})")

    # LaTeX table
    print("\n" + "=" * 70)
    print("LaTeX Table")
    print("=" * 70)
    print(r"""
\begin{table}[t]
\centering
\caption{Effect of feature selection method on 5-expert ensemble performance (In-the-Wild evaluation).}
\label{tab:ablation_ensemble_features}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{F1} & \textbf{AUC} & \textbf{Fake Det.} & \textbf{Real Det.} \\
\midrule""")
    for r in all_results:
        print(f"{r['method']} & {r['f1']:.3f} & {r['auc']:.3f} & {r['fake_detection']*100:.1f}\\% & {r['real_detection']*100:.1f}\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    main()
