#!/usr/bin/env python3
"""TreeSHAP interpretability analysis for the 5-Expert MoE ensemble.

Generates per-expert SHAP beeswarm plots, bar importance plots, force plots,
cross-expert comparison figures, dependence plots, and a JSON report.

Uses the same trained models and eval data as the paper's MoE pipeline.

Usage:
    python scripts/run_treeshap_analysis.py
    python scripts/run_treeshap_analysis.py --max-samples 500
"""

import json
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.data_loader import ALL_FEATURES

# Configuration
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUT_DIR = PROJECT_ROOT / "results" / "shap_analysis"

MOE_EXPERTS = ['ds_msclap', 'audeter_msclap', 'sv_ds_msclap', 'sv_msclap', 'mlaad_msclap']
MOE_WEIGHTS = [0.30, 0.30, 0.20, 0.10, 0.10]

# Explanation dataset
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "eval_features" / "msclap" / "in_the_wild" / "features_combined.json"

# Per-model scaler feature sets (from run_paper_moe_crosseval.py)
FEATURES_27 = [f for f in ALL_FEATURES if f not in ['shapiro_p', 'normaltest_p']]
FEATURES_29 = list(ALL_FEATURES)
MLAAD_FEATURES = [
    'mean', 'std', 'variance', 'min', 'max', 'peak_to_peak', 'skewness', 'kurtosis',
    'bimodality_coefficient', 'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr',
    'tail_weight_ratio', 'trimmed_mean', 'entropy', 'gini_coefficient',
    'coefficient_of_variation', 'variance_mean_ratio', 'kurtosis_variance_ratio',
    'skewness_kurtosis_ratio', 'iqr_range_ratio', 'median_mean_diff', 'n_samples',
    'shapiro_p', 'normaltest_p',
]

SCALER_FEATURES = {
    'ds_msclap': FEATURES_29,
    'audeter_msclap': FEATURES_29,
    'sv_ds_msclap': FEATURES_27,
    'sv_msclap': FEATURES_29,
    'mlaad_msclap': MLAAD_FEATURES,
}


def load_eval_data(max_samples=1000):
    """Load In-the-Wild eval data for SHAP explanation."""
    with open(EVAL_DATA_PATH, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    X = df[ALL_FEATURES].copy().fillna(df[ALL_FEATURES].median())
    y = df['label'].values

    # Subsample for SHAP (stratified)
    if len(X) > max_samples:
        np.random.seed(42)
        idx_real = np.where(y == 0)[0]
        idx_fake = np.where(y == 1)[0]
        n_per_class = max_samples // 2
        sel_real = np.random.choice(idx_real, min(n_per_class, len(idx_real)), replace=False)
        sel_fake = np.random.choice(idx_fake, min(n_per_class, len(idx_fake)), replace=False)
        sel = np.concatenate([sel_real, sel_fake])
        X = X.iloc[sel].reset_index(drop=True)
        y = y[sel]

    return X, y


def load_expert(expert_name):
    """Load a single expert model."""
    model_path = MODELS_DIR / f"{expert_name}_model.pkl"
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return {
        'name': expert_name,
        'model': data['model'],
        'features': data['features'],
        'scaler': data.get('scaler'),
    }


def prepare_expert_data(expert, X_all):
    """Prepare scaled feature matrix for a specific expert."""
    scaler_feats = SCALER_FEATURES.get(expert['name'])

    if expert['scaler'] is not None and scaler_feats is not None:
        available = [f for f in scaler_feats if f in X_all.columns]
        X_for_scaler = X_all[available].copy().fillna(0)
        for f in scaler_feats:
            if f not in X_for_scaler.columns:
                X_for_scaler[f] = 0.0
        X_for_scaler = X_for_scaler[scaler_feats]
        X_scaled = expert['scaler'].transform(X_for_scaler.values)
        X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_feats, index=X_all.index)
        return X_scaled_df[expert['features']]
    else:
        return X_all[expert['features']].copy().fillna(0)


def run_shap_analysis(expert, X_expert, y, output_dir, max_display=15):
    """Run full SHAP analysis for one expert."""
    import shap

    name = expert['name']
    expert_dir = output_dir / name
    expert_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Computing SHAP values for {name}...")
    explainer = shap.TreeExplainer(expert['model'])
    shap_values = explainer.shap_values(X_expert)

    feature_names = list(X_expert.columns)

    # Mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # 1. Beeswarm plot
    print(f"    Generating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_expert,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Beeswarm: {name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(expert_dir / "beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Bar importance plot
    print(f"    Generating bar importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_expert,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance: {name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(expert_dir / "bar_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Force plots for example predictions
    print(f"    Generating force plots...")
    y_pred_proba = expert['model'].predict_proba(X_expert)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    examples = {}
    # Correct real prediction
    correct_real = np.where((y == 0) & (y_pred == 0))[0]
    if len(correct_real) > 0:
        examples['correct_real'] = correct_real[0]
    # Correct fake prediction
    correct_fake = np.where((y == 1) & (y_pred == 1))[0]
    if len(correct_fake) > 0:
        examples['correct_fake'] = correct_fake[0]
    # Misclassified
    misclassified = np.where(y != y_pred)[0]
    if len(misclassified) > 0:
        examples['misclassified'] = misclassified[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]

    for ex_name, idx in examples.items():
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            expected_value,
            shap_values[idx],
            X_expert.iloc[idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.title(f"Force Plot: {name} - {ex_name} (true={y[idx]}, pred={y_pred_proba[idx]:.3f})")
        plt.tight_layout()
        plt.savefig(expert_dir / f"force_{ex_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Dependence plots for top-3 features
    print(f"    Generating dependence plots...")
    top_features = list(importance.keys())[:3]
    for feat in top_features:
        feat_idx = feature_names.index(feat)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feat_idx, shap_values, X_expert,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"Dependence: {name} - {feat}", fontsize=12)
        plt.tight_layout()
        safe_feat = feat.replace("/", "_")
        plt.savefig(expert_dir / f"dependence_{safe_feat}.png", dpi=150, bbox_inches='tight')
        plt.close()

    return {
        'name': name,
        'feature_importance': importance,
        'shap_values': shap_values,
        'feature_names': feature_names,
        'expected_value': float(expected_value),
        'n_samples': len(X_expert),
    }


def create_cross_expert_comparison(all_results, output_dir):
    """Create cross-expert comparison figure showing top-10 features by mean |SHAP|."""
    print("\n  Creating cross-expert comparison...")

    # Collect all importance scores
    all_features = set()
    for result in all_results:
        all_features.update(result['feature_importance'].keys())

    # Build comparison matrix
    feature_scores = {}
    for feat in all_features:
        scores = []
        for result in all_results:
            scores.append(result['feature_importance'].get(feat, 0))
        feature_scores[feat] = np.mean(scores)

    # Top 10 by average importance
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_feature_names = [f[0] for f in top_features]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_feature_names))
    width = 0.15
    colors = ['#2166ac', '#d6604d', '#4dac26', '#984ea3', '#ff7f00']

    for i, result in enumerate(all_results):
        values = [result['feature_importance'].get(f, 0) for f in top_feature_names]
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=result['name'],
                      color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Cross-Expert Feature Importance Comparison (Top 10)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "cross_expert_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    return top_feature_names, feature_scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TreeSHAP analysis for MoE experts")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples for SHAP explanation (default: 1000)")
    parser.add_argument("--max-display", type=int, default=15,
                        help="Max features to display in plots (default: 15)")
    args = parser.parse_args()

    print("=" * 70)
    print("TreeSHAP Interpretability Analysis: 5-Expert MoE Ensemble")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval data
    print(f"\nLoading eval data from {EVAL_DATA_PATH}...")
    X_all, y = load_eval_data(max_samples=args.max_samples)
    print(f"  Samples: {len(y)} (real={np.sum(y==0)}, fake={np.sum(y==1)})")

    # Load and analyze each expert
    all_results = []
    for i, expert_name in enumerate(MOE_EXPERTS):
        print(f"\n{'─' * 50}")
        print(f"Expert {i+1}/{len(MOE_EXPERTS)}: {expert_name} (weight={MOE_WEIGHTS[i]:.2f})")
        print(f"{'─' * 50}")

        expert = load_expert(expert_name)
        print(f"  Features: {len(expert['features'])}")

        X_expert = prepare_expert_data(expert, X_all)
        result = run_shap_analysis(expert, X_expert, y, OUTPUT_DIR, max_display=args.max_display)
        all_results.append(result)

    # Cross-expert comparison
    top_features, avg_scores = create_cross_expert_comparison(all_results, OUTPUT_DIR)

    # Save JSON report
    print("\n  Saving JSON report...")
    report = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'eval_dataset': str(EVAL_DATA_PATH),
        'n_samples': len(y),
        'n_real': int(np.sum(y == 0)),
        'n_fake': int(np.sum(y == 1)),
        'experts': [],
        'cross_expert_top10': {f: float(avg_scores[f]) for f in top_features},
    }

    for result in all_results:
        expert_report = {
            'name': result['name'],
            'n_features': len(result['feature_names']),
            'features': result['feature_names'],
            'expected_value': result['expected_value'],
            'feature_importance': {k: float(v) for k, v in result['feature_importance'].items()},
            'top_5_features': list(result['feature_importance'].keys())[:5],
        }
        report['experts'].append(expert_report)

    report_path = OUTPUT_DIR / "shap_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTop 10 features by average |SHAP| across all experts:")
    for i, feat in enumerate(top_features):
        print(f"  {i+1:2d}. {feat:<30s} {avg_scores[feat]:.4f}")

    print(f"\nPer-expert top 3:")
    for result in all_results:
        top3 = list(result['feature_importance'].keys())[:3]
        print(f"  {result['name']}: {', '.join(top3)}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"  - Per-expert plots: {OUTPUT_DIR}/{{expert_name}}/")
    print(f"  - Cross-expert comparison: {OUTPUT_DIR}/cross_expert_comparison.png")
    print(f"  - JSON report: {report_path}")


if __name__ == '__main__':
    main()
