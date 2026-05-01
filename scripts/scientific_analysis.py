"""Scientific analysis of audio deepfake detection features.

Follows the methodology from the video deepfake paper (Norman & Farid, 2025),
adapted for audio using CLAP embeddings instead of ArcFace.

Key visualizations:
1. Individual similarity distributions (like paper Figure 2)
2. Real vs Fake overlay comparisons per audio type
3. Feature distribution comparisons
4. Feature importance analysis
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

#
# Dataset and Embedding Configuration
#
# Dataset names for each audio type (for plot labels)
DATASET_INFO = {
    'single_voice': {
        'real': {'name': 'LibriSpeech', 'description': 'LibriSpeech clean-100 subset'},
        'fake': {'name': 'ASVspoof 2019', 'description': 'ASVspoof 2019 LA (TTS/VC)'},
    },
    'music_instrumental': {
        'real': {'name': 'MUSDB18', 'description': 'MUSDB18 instrumental stems'},
        'fake': {'name': 'FakeMusicCaps', 'description': 'AI-generated music (5 models)'},
    },
    'music_with_vocals': {
        'real': {'name': 'MUSDB18', 'description': 'MUSDB18 full mixture'},
        'fake': {'name': 'FakeMusicCaps', 'description': 'AI-generated music (5 models)'},
    },
    'deepspeak_v2_train': {
        'real': {'name': 'DeepSpeak v2', 'description': 'Authentic speaker talking head audio'},
        'fake': {'name': 'DeepSpeak v2 (VC)', 'description': 'Voice-cloned audio (ElevenLabs, PlayHT, Speechify)'},
    },
}

# Embedding model display names
EMBEDDING_NAMES = {
    'laion_clap': 'LAION-CLAP',
    'msclap': 'MS-CLAP',
}

# Set publication-quality defaults
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


def load_features_with_similarities(json_path):
    """Load features and raw similarities from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def plot_similarity_distributions_individual(data_list, titles, output_path,
                                              color='blue', n_examples=4):
    """Plot individual similarity distributions like paper Figure 2.

    Shows n_examples individual distributions to demonstrate variability.
    """
    fig, axes = plt.subplots(1, n_examples, figsize=(12, 3))

    # Sample n_examples from data
    indices = np.random.choice(len(data_list), min(n_examples, len(data_list)), replace=False)

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        item = data_list[idx]
        if 'similarities' in item and item['similarities']:
            sims = np.array(item['similarities'])
            ax.hist(sims, bins=30, density=True, alpha=0.7, color=color, edgecolor='white')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Frequency' if i == 0 else '')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 12)
            ax.set_title(f'Sample {i+1}', fontsize=10)

    fig.suptitle(titles, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_real_vs_fake_comparison(real_data, fake_data, audio_type, output_dir,
                                  embedding_model='laion_clap'):
    """Create Figure 2-style comparison: real distributions (blue) vs fake (red).

    Following the paper's Figure 2 style:
    - Each subplot shows pairwise similarity distribution for ONE audio file
    - Real samples shown in blue (top row), fake in red (bottom row)
    - Using frequency counts (not density) with appropriate bin widths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset info for labels
    dataset_info = DATASET_INFO.get(audio_type, {})
    real_dataset = dataset_info.get('real', {}).get('name', 'Real')
    fake_dataset = dataset_info.get('fake', {}).get('name', 'Fake')
    embedding_name = EMBEDDING_NAMES.get(embedding_model, embedding_model)

    # Filter to samples with similarity data (min_segments=3 gives C(3,2)=3 pairs minimum)
    real_with_sims = [d for d in real_data if 'similarities' in d and len(d.get('similarities', [])) >= 3]
    fake_with_sims = [d for d in fake_data if 'similarities' in d and len(d.get('similarities', [])) >= 3]

    if not real_with_sims or not fake_with_sims:
        print(f"  Skipping {audio_type}: insufficient similarity data")
        return

    # 1. Individual examples comparison (like paper Figure 2)
    # Paper shows 2 real + 2 fake, we'll show 4 of each for more comprehensive view
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    # Fixed bins for consistent comparison across all plots
    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
    bin_width = bins[1] - bins[0]  # ~0.033

    # Sample 4 real examples - prefer samples with more similarity pairs
    real_sorted = sorted(real_with_sims, key=lambda x: len(x.get('similarities', [])), reverse=True)
    real_samples = real_sorted[:min(4, len(real_sorted))]

    for i, sample in enumerate(real_samples):
        sims = np.array(sample['similarities'])
        n_pairs = len(sims)

        # Use frequency counts, normalize by dividing by total to get proportions
        counts, _ = np.histogram(sims, bins=bins)
        # Scale to reasonable y-axis (like paper's 0-12 range)
        counts_scaled = counts / counts.max() * 10 if counts.max() > 0 else counts

        axes[0, i].bar(bins[:-1], counts_scaled, width=bin_width * 0.9, align='edge',
                       color='#2166ac', alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 12)
        axes[0, i].set_xlabel('Cosine Similarity', fontsize=10)
        axes[0, i].set_title(f'n={n_pairs} pairs', fontsize=9, style='italic')
        # Y-axis label only on first column
        if i == 0:
            axes[0, i].set_ylabel(f'{real_dataset}\nFrequency', fontsize=10, color='#2166ac')

    # Sample 4 fake examples
    fake_sorted = sorted(fake_with_sims, key=lambda x: len(x.get('similarities', [])), reverse=True)
    fake_samples = fake_sorted[:min(4, len(fake_sorted))]

    for i, sample in enumerate(fake_samples):
        sims = np.array(sample['similarities'])
        n_pairs = len(sims)

        counts, _ = np.histogram(sims, bins=bins)
        counts_scaled = counts / counts.max() * 10 if counts.max() > 0 else counts

        axes[1, i].bar(bins[:-1], counts_scaled, width=bin_width * 0.9, align='edge',
                       color='#b2182b', alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 12)
        axes[1, i].set_xlabel('Cosine Similarity', fontsize=10)
        axes[1, i].set_title(f'n={n_pairs} pairs', fontsize=9, style='italic')
        # Y-axis label only on first column
        if i == 0:
            axes[1, i].set_ylabel(f'{fake_dataset}\nFrequency', fontsize=10, color='#b2182b')

    # Main title with embedding model
    audio_type_title = audio_type.replace("_", " ").title()
    title = f'{audio_type_title}: Pairwise Similarity Distributions\n[{embedding_name} Embeddings]'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Add caption explaining what's shown (clearer explanation)
    fig.text(0.5, -0.04,
             f'Each panel = one audio file segmented into overlapping windows. n = number of pairwise similarities between segments.\n'
             f'Real: {real_dataset} (blue) vs Fake: {fake_dataset} (red). Showing 4 example files per class.',
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / f'{audio_type}_individual_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Aggregated overlay comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate all similarities
    all_real_sims = np.concatenate([np.array(d['similarities']) for d in real_with_sims])
    all_fake_sims = np.concatenate([np.array(d['similarities']) for d in fake_with_sims])

    # Plot overlaid histograms with dataset names in legend
    bins = np.linspace(0, 1, 50)
    ax.hist(all_real_sims, bins=bins, density=True, alpha=0.6, color='#2166ac',
            label=f'{real_dataset} (n={len(real_with_sims)})', edgecolor='white')
    ax.hist(all_fake_sims, bins=bins, density=True, alpha=0.6, color='#b2182b',
            label=f'{fake_dataset} (n={len(fake_with_sims)})', edgecolor='white')

    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{audio_type_title}: Aggregated Similarity Distributions\n[{embedding_name} Embeddings]',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.set_xlim(0, 1)

    # Add statistics with dataset names
    real_mean, real_std = np.mean(all_real_sims), np.std(all_real_sims)
    fake_mean, fake_std = np.mean(all_fake_sims), np.std(all_fake_sims)
    stats_text = f'{real_dataset}: μ={real_mean:.3f}, σ={real_std:.3f}\n{fake_dataset}: μ={fake_mean:.3f}, σ={fake_std:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / f'{audio_type}_aggregated_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Per-file statistics distribution (shows ALL files, not just 4 examples)
    # This proves the pattern holds across the entire dataset
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Compute per-file statistics
    real_file_means = [np.mean(d['similarities']) for d in real_with_sims]
    fake_file_means = [np.mean(d['similarities']) for d in fake_with_sims]
    real_file_stds = [np.std(d['similarities']) for d in real_with_sims]
    fake_file_stds = [np.std(d['similarities']) for d in fake_with_sims]
    real_file_ranges = [np.ptp(d['similarities']) for d in real_with_sims]
    fake_file_ranges = [np.ptp(d['similarities']) for d in fake_with_sims]

    # Plot 1: Per-file mean similarity
    bins_mean = np.linspace(0.3, 1.0, 30)
    axes[0].hist(real_file_means, bins=bins_mean, density=True, alpha=0.6, color='#2166ac',
                 label=f'{real_dataset} (n={len(real_file_means)})')
    axes[0].hist(fake_file_means, bins=bins_mean, density=True, alpha=0.6, color='#b2182b',
                 label=f'{fake_dataset} (n={len(fake_file_means)})')
    axes[0].set_xlabel('Mean Similarity per File', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Per-File Mean Similarity', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Plot 2: Per-file std
    bins_std = np.linspace(0, 0.3, 30)
    axes[1].hist(real_file_stds, bins=bins_std, density=True, alpha=0.6, color='#2166ac',
                 label=f'{real_dataset}')
    axes[1].hist(fake_file_stds, bins=bins_std, density=True, alpha=0.6, color='#b2182b',
                 label=f'{fake_dataset}')
    axes[1].set_xlabel('Std Dev per File', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Per-File Similarity Variance', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)

    # Plot 3: Per-file range (peak-to-peak)
    bins_range = np.linspace(0, 1.0, 30)
    axes[2].hist(real_file_ranges, bins=bins_range, density=True, alpha=0.6, color='#2166ac',
                 label=f'{real_dataset}')
    axes[2].hist(fake_file_ranges, bins=bins_range, density=True, alpha=0.6, color='#b2182b',
                 label=f'{fake_dataset}')
    axes[2].set_xlabel('Range (max-min) per File', fontsize=11)
    axes[2].set_ylabel('Density', fontsize=11)
    axes[2].set_title('Per-File Similarity Range', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)

    fig.suptitle(f'{audio_type_title}: Per-File Statistics Across ALL Files\n[{embedding_name} Embeddings]',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{audio_type}_perfile_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'real_mean': real_mean, 'real_std': real_std,
        'fake_mean': fake_mean, 'fake_std': fake_std,
        'n_real': len(real_with_sims), 'n_fake': len(fake_with_sims)
    }


def plot_feature_comparison(real_df, fake_df, feature_name, output_path):
    """Create clean box plot + histogram comparison for a single feature."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    real_vals = real_df[feature_name].dropna()
    fake_vals = fake_df[feature_name].dropna()

    # Box plot
    bp = ax1.boxplot([real_vals, fake_vals], labels=['Real', 'Fake'],
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2166ac')
    bp['boxes'][1].set_facecolor('#b2182b')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax1.set_ylabel(feature_name)
    ax1.set_title('Distribution Comparison')

    # Histogram overlay
    bins = np.linspace(min(real_vals.min(), fake_vals.min()),
                       max(real_vals.max(), fake_vals.max()), 40)
    ax2.hist(real_vals, bins=bins, density=True, alpha=0.6, color='#2166ac', label='Real')
    ax2.hist(fake_vals, bins=bins, density=True, alpha=0.6, color='#b2182b', label='Fake')
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.set_title('Density Overlay')

    fig.suptitle(f'Feature: {feature_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_features_grid(real_df, fake_df, top_features, output_path):
    """Create a grid of the top N most discriminative features."""
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, (feature, auc) in enumerate(top_features):
        ax = axes[i]
        real_vals = real_df[feature].dropna()
        fake_vals = fake_df[feature].dropna()

        bins = np.linspace(min(real_vals.min(), fake_vals.min()),
                           max(real_vals.max(), fake_vals.max()), 30)
        ax.hist(real_vals, bins=bins, density=True, alpha=0.6, color='#2166ac', label='Real')
        ax.hist(fake_vals, bins=bins, density=True, alpha=0.6, color='#b2182b', label='Fake')
        ax.set_title(f'{feature}\nAUC={auc:.3f}', fontsize=10)
        ax.set_xlabel('')
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Top Discriminative Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_correlation_clean(df, features, output_path, max_features=15):
    """Create a readable correlation heatmap with only key features."""
    # Select subset of features
    features = features[:max_features]
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a diverging colormap
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(features, fontsize=9)

    # Add correlation values
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=7)

    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_feature_importance(real_df, fake_df, features):
    """Compute AUC-ROC for each feature (single-feature classification)."""
    results = []

    for feature in features:
        real_vals = real_df[feature].dropna()
        fake_vals = fake_df[feature].dropna()

        if len(real_vals) < 2 or len(fake_vals) < 2:
            continue

        # Combine and compute AUC
        all_vals = np.concatenate([real_vals, fake_vals])
        all_labels = np.concatenate([np.zeros(len(real_vals)), np.ones(len(fake_vals))])

        try:
            auc = roc_auc_score(all_labels, all_vals)
            direction = 'fake_higher' if auc >= 0.5 else 'real_higher'
            auc = max(auc, 1 - auc)  # Ensure AUC >= 0.5

            # Cohen's d
            pooled_std = np.sqrt(((len(real_vals)-1)*np.var(real_vals) +
                                  (len(fake_vals)-1)*np.var(fake_vals)) /
                                 (len(real_vals) + len(fake_vals) - 2))
            cohens_d = (np.mean(real_vals) - np.mean(fake_vals)) / (pooled_std + 1e-10)

            # Statistical tests
            t_stat, t_pval = stats.ttest_ind(real_vals, fake_vals)

            results.append({
                'feature': feature,
                'auc': auc,
                'direction': direction,
                'cohens_d': cohens_d,
                't_pvalue': t_pval,
                'real_mean': np.mean(real_vals),
                'fake_mean': np.mean(fake_vals),
            })
        except:
            pass

    return pd.DataFrame(results).sort_values('auc', ascending=False)


def create_summary_figure(all_results, output_path):
    """Create publication-ready summary figure across all audio types."""
    audio_types = list(all_results.keys())
    n_types = len(audio_types)

    fig = plt.figure(figsize=(14, 4*n_types))
    gs = gridspec.GridSpec(n_types, 3, width_ratios=[1, 1, 1.2])

    for i, audio_type in enumerate(audio_types):
        results = all_results[audio_type]

        if 'sim_stats' not in results or results['sim_stats'] is None:
            continue

        # 1. Similarity distribution overlay
        ax1 = fig.add_subplot(gs[i, 0])
        # Would need raw similarities here - skip for now
        ax1.text(0.5, 0.5, f'{audio_type}\nSimilarity\nDistribution',
                ha='center', va='center', fontsize=11)
        ax1.set_title('Similarity Distribution')

        # 2. Top features bar chart
        ax2 = fig.add_subplot(gs[i, 1])
        if 'feature_importance' in results:
            top_5 = results['feature_importance'].head(5)
            bars = ax2.barh(range(len(top_5)), top_5['auc'].values, color='#4393c3')
            ax2.set_yticks(range(len(top_5)))
            ax2.set_yticklabels(top_5['feature'].values)
            ax2.set_xlabel('AUC-ROC')
            ax2.set_xlim(0.5, 1.0)
            ax2.invert_yaxis()
        ax2.set_title('Top 5 Features')

        # 3. Summary statistics table
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.axis('off')
        stats = results['sim_stats']
        table_data = [
            ['Metric', 'Real', 'Fake'],
            ['N samples', f"{stats['n_real']}", f"{stats['n_fake']}"],
            ['Mean sim.', f"{stats['real_mean']:.3f}", f"{stats['fake_mean']:.3f}"],
            ['Std sim.', f"{stats['real_std']:.3f}", f"{stats['fake_std']:.3f}"],
        ]
        table = ax3.table(cellText=table_data, loc='center', cellLoc='center',
                          colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title(f'{audio_type.replace("_", " ").title()}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_audio_type(audio_type, real_path, fake_path, output_dir, embedding_model='laion_clap'):
    """Complete analysis for one audio type."""
    embedding_name = EMBEDDING_NAMES.get(embedding_model, embedding_model)
    dataset_info = DATASET_INFO.get(audio_type, {})
    real_dataset = dataset_info.get('real', {}).get('name', 'Real')
    fake_dataset = dataset_info.get('fake', {}).get('name', 'Fake')

    print(f"\n{'='*60}")
    print(f"Analyzing: {audio_type}")
    print(f"  Embedding: {embedding_name}")
    print(f"  Real data: {real_dataset}")
    print(f"  Fake data: {fake_dataset}")
    print(f"{'='*60}")

    output_dir = Path(output_dir) / audio_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    real_data = load_features_with_similarities(real_path)
    fake_data = load_features_with_similarities(fake_path)

    print(f"  Real samples: {len(real_data)}")
    print(f"  Fake samples: {len(fake_data)}")

    # 1. Similarity distribution plots
    print("  Creating similarity distribution plots...")
    sim_stats = plot_real_vs_fake_comparison(real_data, fake_data, audio_type, output_dir,
                                              embedding_model=embedding_model)

    # 2. Convert to DataFrames for feature analysis
    def to_dataframe(data_list):
        rows = []
        for item in data_list:
            row = {'label': item.get('label', 0)}
            row.update(item.get('features', {}))
            rows.append(row)
        return pd.DataFrame(rows)

    real_df = to_dataframe(real_data)
    fake_df = to_dataframe(fake_data)

    # Get feature columns
    feature_cols = [c for c in real_df.columns if c not in ['label', 'n_samples']]

    # 3. Feature importance analysis
    print("  Computing feature importance...")
    importance_df = compute_feature_importance(real_df, fake_df, feature_cols)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # 4. Top features visualization
    print("  Creating feature visualizations...")
    top_features = list(zip(importance_df['feature'].head(9), importance_df['auc'].head(9)))
    plot_top_features_grid(real_df, fake_df, top_features, output_dir / 'top_features_grid.png')

    # 5. Feature correlation (readable version)
    combined_df = pd.concat([real_df, fake_df])
    top_feature_names = importance_df['feature'].head(12).tolist()
    plot_feature_correlation_clean(combined_df, top_feature_names,
                                   output_dir / 'feature_correlation.png')

    # 6. Individual feature plots for top 5
    for feature, auc in top_features[:5]:
        plot_feature_comparison(real_df, fake_df, feature,
                               output_dir / f'feature_{feature}.png')

    # Print summary
    print("\n  Top 5 discriminative features:")
    for _, row in importance_df.head(5).iterrows():
        sig = '***' if row['t_pvalue'] < 0.001 else '**' if row['t_pvalue'] < 0.01 else '*'
        print(f"    {row['feature']:<25} AUC={row['auc']:.3f}  d={row['cohens_d']:+.2f} {sig}")

    return {
        'sim_stats': sim_stats,
        'feature_importance': importance_df,
        'n_real': len(real_data),
        'n_fake': len(fake_data)
    }


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description='Scientific analysis of audio deepfake features')
    parser.add_argument('--embedding-model', '-m', default='laion_clap',
                        choices=['laion_clap', 'msclap'],
                        help='Embedding model used for features (default: laion_clap)')
    parser.add_argument('--data-dir', '-d', default='data/features',
                        help='Directory containing feature JSON files')
    parser.add_argument('--output-dir', '-o', default='data/analysis_scientific',
                        help='Base output directory for analysis results')
    args = parser.parse_args()

    embedding_model = args.embedding_model
    embedding_name = EMBEDDING_NAMES.get(embedding_model, embedding_model)
    data_dir = Path(args.data_dir)

    # Organize output by embedding type
    output_dir = Path(args.output_dir) / embedding_model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Scientific Analysis - {embedding_name} Embeddings")
    print(f"# Data: {data_dir}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")

    audio_types = ['single_voice', 'music_instrumental', 'music_with_vocals', 'deepspeak_v2_train']
    all_results = {}

    for audio_type in audio_types:
        real_path = data_dir / f'{audio_type}_real.json'
        fake_path = data_dir / f'{audio_type}_fake.json'

        if real_path.exists() and fake_path.exists():
            results = analyze_audio_type(audio_type, real_path, fake_path, output_dir,
                                         embedding_model=embedding_model)
            all_results[audio_type] = results
        else:
            print(f"\nSkipping {audio_type}: missing feature files")
            if not real_path.exists():
                print(f"  Missing: {real_path}")
            if not fake_path.exists():
                print(f"  Missing: {fake_path}")

    # Cross-type comparison
    print(f"\n{'='*60}")
    print("Cross-Audio-Type Comparison")
    print(f"{'='*60}")

    # Collect all feature importances
    all_importance = []
    for audio_type, results in all_results.items():
        if 'feature_importance' in results:
            df = results['feature_importance'].copy()
            df['audio_type'] = audio_type
            all_importance.append(df)

    if all_importance:
        combined_importance = pd.concat(all_importance)

        # Pivot for cross-comparison
        pivot = combined_importance.pivot(index='feature', columns='audio_type', values='auc')
        pivot['mean_auc'] = pivot.mean(axis=1)
        pivot['min_auc'] = pivot.min(axis=1)
        pivot = pivot.sort_values('mean_auc', ascending=False)
        pivot.to_csv(output_dir / 'cross_type_comparison.csv')

        print("\nTop 10 features across all audio types:")
        print(pivot.head(10).to_string())

        # Consistently strong features
        consistent = pivot[pivot['min_auc'] > 0.6]
        print(f"\nFeatures with AUC > 0.6 across ALL types: {len(consistent)}")
        for feat in consistent.index[:10]:
            print(f"  {feat}: mean={pivot.loc[feat, 'mean_auc']:.3f}, min={pivot.loc[feat, 'min_auc']:.3f}")

    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
