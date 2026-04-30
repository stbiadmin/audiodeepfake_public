"""Compare LAION-CLAP vs MS-CLAP embedding performance across audio types.

Generates:
- embedding_comparison_summary.csv
- auc_scatter_comparison.png
- laion_vs_msclap_feature_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# Audio types to compare
AUDIO_TYPES = ['single_voice', 'music_instrumental', 'music_with_vocals', 'deepspeak_v2_train']

# Display names
AUDIO_TYPE_NAMES = {
    'single_voice': 'Single Voice',
    'music_instrumental': 'Music (Instrumental)',
    'music_with_vocals': 'Music (Vocals)',
    'deepspeak_v2_train': 'DeepSpeak v2',
}


def load_feature_importance(analysis_dir: Path, embedding_model: str, audio_type: str) -> pd.DataFrame:
    """Load feature importance CSV for a specific embedding model and audio type."""
    csv_path = analysis_dir / embedding_model / audio_type / 'feature_importance.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def create_comparison_summary(analysis_dir: Path, output_dir: Path):
    """Create summary CSV comparing best features across embedding models."""
    rows = []

    for audio_type in AUDIO_TYPES:
        laion_df = load_feature_importance(analysis_dir, 'laion_clap', audio_type)
        msclap_df = load_feature_importance(analysis_dir, 'msclap', audio_type)

        if laion_df is None or msclap_df is None:
            print(f"  Skipping {audio_type}: missing data")
            continue

        # Get best feature for each
        laion_best = laion_df.iloc[0]
        msclap_best = msclap_df.iloc[0]

        # Determine which is better
        if laion_best['auc'] > msclap_best['auc']:
            better = 'LAION-CLAP'
            diff = laion_best['auc'] - msclap_best['auc']
        else:
            better = 'MS-CLAP'
            diff = msclap_best['auc'] - laion_best['auc']

        rows.append({
            'Audio Type': AUDIO_TYPE_NAMES.get(audio_type, audio_type),
            'LAION-CLAP Best AUC': round(laion_best['auc'], 3),
            'LAION-CLAP Best Feature': laion_best['feature'],
            'MS-CLAP Best AUC': round(msclap_best['auc'], 3),
            'MS-CLAP Best Feature': msclap_best['feature'],
            'Better Embedding': better,
            'AUC Difference': round(diff, 3),
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / 'embedding_comparison_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


def create_auc_scatter(analysis_dir: Path, output_dir: Path):
    """Create scatter plot comparing AUC scores between embedding models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(AUDIO_TYPES)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    all_laion = []
    all_msclap = []

    for i, audio_type in enumerate(AUDIO_TYPES):
        laion_df = load_feature_importance(analysis_dir, 'laion_clap', audio_type)
        msclap_df = load_feature_importance(analysis_dir, 'msclap', audio_type)

        if laion_df is None or msclap_df is None:
            continue

        # Merge on feature name
        merged = laion_df.merge(msclap_df, on='feature', suffixes=('_laion', '_msclap'))

        laion_aucs = merged['auc_laion'].values
        msclap_aucs = merged['auc_msclap'].values

        all_laion.extend(laion_aucs)
        all_msclap.extend(msclap_aucs)

        ax.scatter(laion_aucs, msclap_aucs,
                   c=[colors[i]], marker=markers[i % len(markers)],
                   label=AUDIO_TYPE_NAMES.get(audio_type, audio_type),
                   alpha=0.7, s=50)

    # Add diagonal line
    lims = [0.5, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal Performance')

    ax.set_xlabel('LAION-CLAP AUC', fontsize=12)
    ax.set_ylabel('MS-CLAP AUC', fontsize=12)
    ax.set_title('Feature AUC Comparison: LAION-CLAP vs MS-CLAP', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add correlation
    if all_laion and all_msclap:
        corr = np.corrcoef(all_laion, all_msclap)[0, 1]
        ax.text(0.52, 0.95, f'r = {corr:.3f}', fontsize=11,
                transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    output_path = output_dir / 'auc_scatter_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_feature_comparison_plot(analysis_dir: Path, output_dir: Path):
    """Create bar chart comparing top features between embedding models."""
    n_types = len([t for t in AUDIO_TYPES
                   if load_feature_importance(analysis_dir, 'laion_clap', t) is not None])

    if n_types == 0:
        print("  No data available for feature comparison plot")
        return

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 6))
    if n_types == 1:
        axes = [axes]

    plot_idx = 0
    for audio_type in AUDIO_TYPES:
        laion_df = load_feature_importance(analysis_dir, 'laion_clap', audio_type)
        msclap_df = load_feature_importance(analysis_dir, 'msclap', audio_type)

        if laion_df is None or msclap_df is None:
            continue

        ax = axes[plot_idx]

        # Get top 10 features from each
        top_features = set(laion_df.head(10)['feature'].tolist() +
                          msclap_df.head(10)['feature'].tolist())
        top_features = sorted(top_features)[:12]  # Limit to 12

        # Get AUCs for these features
        laion_aucs = []
        msclap_aucs = []
        for feat in top_features:
            laion_row = laion_df[laion_df['feature'] == feat]
            msclap_row = msclap_df[msclap_df['feature'] == feat]
            laion_aucs.append(laion_row['auc'].values[0] if len(laion_row) > 0 else 0.5)
            msclap_aucs.append(msclap_row['auc'].values[0] if len(msclap_row) > 0 else 0.5)

        x = np.arange(len(top_features))
        width = 0.35

        bars1 = ax.barh(x - width/2, laion_aucs, width, label='LAION-CLAP', color='#2166ac', alpha=0.8)
        bars2 = ax.barh(x + width/2, msclap_aucs, width, label='MS-CLAP', color='#b2182b', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('AUC-ROC')
        ax.set_xlim(0.5, 1.0)
        ax.set_title(AUDIO_TYPE_NAMES.get(audio_type, audio_type), fontsize=12, fontweight='bold')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        if plot_idx == 0:
            ax.legend(loc='lower right')

        plot_idx += 1

    fig.suptitle('Feature Performance: LAION-CLAP vs MS-CLAP', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'laion_vs_msclap_feature_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare LAION-CLAP vs MS-CLAP embedding performance')
    parser.add_argument('--analysis-dir', '-a', default='data/analysis_scientific',
                        help='Directory containing analysis results')
    parser.add_argument('--output-dir', '-o', default='data/analysis_scientific/comparison',
                        help='Output directory for comparison results')
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nComparing LAION-CLAP vs MS-CLAP")
    print(f"  Analysis dir: {analysis_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Audio types: {AUDIO_TYPES}")
    print()

    print("Creating comparison summary...")
    summary_df = create_comparison_summary(analysis_dir, output_dir)
    print()
    print(summary_df.to_string(index=False))
    print()

    print("Creating AUC scatter plot...")
    create_auc_scatter(analysis_dir, output_dir)

    print("Creating feature comparison plot...")
    create_feature_comparison_plot(analysis_dir, output_dir)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
