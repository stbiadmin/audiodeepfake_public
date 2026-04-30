"""Analyze augmented vs original real samples for music datasets.

Compares feature distributions between:
- Original real samples (144)
- Augmented real samples (4,896)
- Fake samples (5,521)

Generates visualizations to verify augmentation quality.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import argparse


AUDIO_TYPES = ['music_instrumental', 'music_with_vocals']
EMBEDDING_MODELS = ['laion_clap', 'msclap']

EMBEDDING_NAMES = {
    'laion_clap': 'LAION-CLAP',
    'msclap': 'MS-CLAP',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def to_dataframe(data_list):
    """Convert list of feature dicts to DataFrame."""
    rows = []
    for item in data_list:
        row = {
            'label': item.get('label', 0),
            'augmented': item.get('augmented', False),
        }
        row.update(item.get('features', {}))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_augmentation_comparison(original_df, augmented_df, fake_df,
                                  audio_type, embedding_model, output_dir):
    """Create comparison plots for original vs augmented vs fake."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_name = EMBEDDING_NAMES.get(embedding_model, embedding_model)
    audio_title = audio_type.replace('_', ' ').title()

    # Get feature columns
    feature_cols = [c for c in original_df.columns if c not in ['label', 'augmented', 'n_samples']]

    # 1. Three-way distribution comparison for top features
    # Compute which features best separate real from fake
    feature_aucs = []
    for feat in feature_cols:
        real_vals = pd.concat([original_df[feat], augmented_df[feat]]).dropna()
        fake_vals = fake_df[feat].dropna()
        if len(real_vals) > 0 and len(fake_vals) > 0:
            all_vals = np.concatenate([real_vals, fake_vals])
            all_labels = np.concatenate([np.zeros(len(real_vals)), np.ones(len(fake_vals))])
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(all_labels, all_vals)
                auc = max(auc, 1 - auc)
                feature_aucs.append((feat, auc))
            except:
                pass

    feature_aucs.sort(key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in feature_aucs[:6]]

    # Plot top 6 features with three distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]

        orig_vals = original_df[feat].dropna()
        aug_vals = augmented_df[feat].dropna()
        fake_vals = fake_df[feat].dropna()

        # Determine bin range
        all_vals = np.concatenate([orig_vals, aug_vals, fake_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 40)

        ax.hist(orig_vals, bins=bins, density=True, alpha=0.6, color='#2166ac',
                label=f'Original Real (n={len(orig_vals)})')
        ax.hist(aug_vals, bins=bins, density=True, alpha=0.6, color='#4393c3',
                label=f'Augmented Real (n={len(aug_vals)})')
        ax.hist(fake_vals, bins=bins, density=True, alpha=0.6, color='#b2182b',
                label=f'Fake (n={len(fake_vals)})')

        ax.set_xlabel(feat)
        ax.set_ylabel('Density' if i % 3 == 0 else '')
        auc = next((a for f, a in feature_aucs if f == feat), 0)
        ax.set_title(f'{feat} (AUC={auc:.3f})')

        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f'{audio_title}: Original vs Augmented vs Fake\n[{embedding_name}]',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{audio_type}_augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Statistical comparison: Original vs Augmented
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    comparison_results = []

    for i, feat in enumerate(top_features):
        ax = axes[i]

        orig_vals = original_df[feat].dropna()
        aug_vals = augmented_df[feat].dropna()

        # Box plot comparison
        bp = ax.boxplot([orig_vals, aug_vals], labels=['Original', 'Augmented'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#2166ac')
        bp['boxes'][1].set_facecolor('#4393c3')
        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax.set_ylabel(feat)

        # Statistical test
        t_stat, p_val = stats.ttest_ind(orig_vals, aug_vals)
        ks_stat, ks_pval = stats.ks_2samp(orig_vals, aug_vals)

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.set_title(f'{feat}\nt-test p={p_val:.3f} ({sig})')

        comparison_results.append({
            'feature': feat,
            'original_mean': orig_vals.mean(),
            'augmented_mean': aug_vals.mean(),
            'mean_diff_pct': (aug_vals.mean() - orig_vals.mean()) / orig_vals.mean() * 100,
            't_pvalue': p_val,
            'ks_pvalue': ks_pval,
        })

    fig.suptitle(f'{audio_title}: Original vs Augmented Real Samples\n[{embedding_name}]',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{audio_type}_original_vs_augmented.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save comparison stats
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_dir / f'{audio_type}_augmentation_stats.csv', index=False)

    # 3. Similarity distribution comparison (if available)
    def get_similarities(data_list):
        sims = []
        for item in data_list:
            if 'similarities' in item and item['similarities']:
                sims.extend(item['similarities'])
        return np.array(sims)

    # Need raw data for similarities
    return comparison_df


def analyze_audio_type(audio_type, embedding_model, data_dir, output_dir):
    """Analyze augmentation for one audio type and embedding model."""
    data_dir = Path(data_dir)

    print(f"\n{'='*60}")
    print(f"Analyzing: {audio_type} [{embedding_model}]")
    print(f"{'='*60}")

    # Load data
    original_path = data_dir / embedding_model / f'{audio_type}_real.json'
    augmented_path = data_dir / embedding_model / f'{audio_type}_real_augmented.json'
    fake_path = data_dir / embedding_model / f'{audio_type}_fake.json'

    if not all(p.exists() for p in [original_path, augmented_path, fake_path]):
        print(f"  Skipping: missing files")
        return None

    original_data = load_json(original_path)
    augmented_data = load_json(augmented_path)
    fake_data = load_json(fake_path)

    print(f"  Original real: {len(original_data)}")
    print(f"  Augmented real: {len(augmented_data)}")
    print(f"  Fake: {len(fake_data)}")

    # Convert to DataFrames
    original_df = to_dataframe(original_data)
    augmented_df = to_dataframe(augmented_data)
    fake_df = to_dataframe(fake_data)

    # Create visualizations
    comparison_df = plot_augmentation_comparison(
        original_df, augmented_df, fake_df,
        audio_type, embedding_model, output_dir
    )

    print(f"\n  Feature comparison (Original vs Augmented):")
    for _, row in comparison_df.iterrows():
        sig = '***' if row['t_pvalue'] < 0.001 else '**' if row['t_pvalue'] < 0.01 else '*' if row['t_pvalue'] < 0.05 else 'ns'
        print(f"    {row['feature']:<25} diff={row['mean_diff_pct']:+.1f}% {sig}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Analyze augmented vs original samples')
    parser.add_argument('--data-dir', '-d', default='data/features',
                        help='Directory containing feature JSON files')
    parser.add_argument('--output-dir', '-o', default='data/analysis_scientific/augmentation',
                        help='Output directory for analysis results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print("# Augmentation Analysis: Original vs Augmented Real Samples")
    print(f"{'#'*60}")

    all_results = []

    for embedding_model in EMBEDDING_MODELS:
        for audio_type in AUDIO_TYPES:
            result = analyze_audio_type(
                audio_type, embedding_model,
                args.data_dir, output_dir / embedding_model
            )
            if result is not None:
                result['audio_type'] = audio_type
                result['embedding_model'] = embedding_model
                all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
