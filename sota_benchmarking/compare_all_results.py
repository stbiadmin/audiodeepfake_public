#!/usr/bin/env python3
"""Unified comparison of all experiment results.

Produces three summary tables:
    Table A: Direct classification (off-the-shelf detectors vs our MoE)
    Table B: Temporal coherence pipeline with different embeddings
    Table C: AT-ADD cross-type evaluation (avg EER)

Usage:
    python sota_benchmarking/compare_all_results.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'sota_benchmarking' / 'results'
OUTPUT_DIR = RESULTS_DIR / 'comparison'


def load_csv_safe(path):
    """Load CSV if it exists, else return empty DataFrame."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_table_a():
    """Table A: Direct Classification - Off-the-shelf detectors."""
    print("\n" + "=" * 70)
    print("TABLE A: Direct Classification (Off-the-Shelf Detectors)")
    print("=" * 70)

    rows = []

    # AASIST baselines
    aasist_df = load_csv_safe(RESULTS_DIR / 'aasist_baseline' / 'aasist_baseline_results.csv')
    if not aasist_df.empty:
        for _, r in aasist_df.iterrows():
            rows.append({
                'model': r['model'],
                'dataset': r['dataset'],
                'n_samples': r.get('n_samples', 0),
                'f1': r['f1'],
                'auc': r['auc'],
                'eer': r['eer'],
            })

    # SSL baselines
    ssl_df = load_csv_safe(RESULTS_DIR / 'ssl_baselines' / 'ssl_baseline_results.csv')
    if not ssl_df.empty:
        for _, r in ssl_df.iterrows():
            rows.append({
                'model': r['model'],
                'dataset': r['dataset'],
                'n_samples': r.get('n_samples', 0),
                'f1': r['f1'],
                'auc': r['auc'],
                'eer': r['eer'],
            })

    # Paper MoE results
    moe_df = load_csv_safe(RESULTS_DIR / 'paper_moe' / 'moe_crosseval_results.csv')
    if not moe_df.empty:
        for _, r in moe_df.iterrows():
            if r['dataset'] in ['in_the_wild', 'asvspoof5']:
                rows.append({
                    'model': 'Our MoE (CLAP temporal)',
                    'dataset': r['dataset'],
                    'n_samples': r.get('n_samples', 0),
                    'f1': r['f1'],
                    'auc': r['auc'],
                    'eer': r['eer'],
                })

    if not rows:
        print("  No results found yet.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Pivot for display
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset].sort_values('eer')
        print(f"\n  Dataset: {dataset}")
        print(f"  {'Model':<35} {'N':>6} {'F1':>7} {'AUC':>7} {'EER':>7}")
        print("  " + "-" * 65)
        for _, r in subset.iterrows():
            print(f"  {r['model']:<35} {int(r['n_samples']):>6} "
                  f"{r['f1']:>7.3f} {r['auc']:>7.3f} {r['eer']:>7.3f}")

    return df


def build_table_b():
    """Table B: Temporal Coherence Pipeline with Different Embeddings."""
    print("\n" + "=" * 70)
    print("TABLE B: Temporal Coherence with Different Embeddings")
    print("=" * 70)

    rows = []

    # SSL temporal comparison
    ssl_temporal_df = load_csv_safe(RESULTS_DIR / 'ssl_temporal' / 'ssl_temporal_comparison.csv')
    if not ssl_temporal_df.empty:
        for _, r in ssl_temporal_df.iterrows():
            rows.append({
                'embedding': r['embedding_model'],
                'dataset': r['eval_dataset'],
                'f1': r['f1'],
                'auc': r['auc'],
                'eer': r['eer'],
            })

    # Paper MoE (CLAP temporal) - from crosseval
    moe_df = load_csv_safe(RESULTS_DIR / 'paper_moe' / 'moe_crosseval_results.csv')
    if not moe_df.empty:
        for _, r in moe_df.iterrows():
            rows.append({
                'embedding': 'msclap (MoE)',
                'dataset': r['dataset'],
                'f1': r['f1'],
                'auc': r['auc'],
                'eer': r['eer'],
            })

    if not rows:
        print("  No results found yet.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Print comparison
    datasets = sorted(df['dataset'].unique())
    embeddings = sorted(df['embedding'].unique())

    header = f"  {'Embedding':<20}"
    for ds in datasets:
        header += f" {ds+' EER':>15} {ds+' AUC':>15}"
    print(header)
    print("  " + "-" * (20 + 30 * len(datasets)))

    for emb in embeddings:
        line = f"  {emb:<20}"
        for ds in datasets:
            match = df[(df['embedding'] == emb) & (df['dataset'] == ds)]
            if not match.empty:
                r = match.iloc[0]
                line += f" {r['eer']:>15.3f} {r['auc']:>15.3f}"
            else:
                line += f" {'N/A':>15} {'N/A':>15}"
        print(line)

    return df


def build_table_c():
    """Table C: AT-ADD Cross-Type Average EER."""
    print("\n" + "=" * 70)
    print("TABLE C: AT-ADD Cross-Type Evaluation (Average EER)")
    print("=" * 70)

    atadd_df = load_csv_safe(RESULTS_DIR / 'atadd' / 'atadd_results.csv')
    if atadd_df.empty:
        print("  No AT-ADD results found yet.")
        return pd.DataFrame()

    # Focus on all_type protocol
    all_type = atadd_df[atadd_df['protocol'] == 'all_type']
    if not all_type.empty:
        print("\n  All-Type Training (combined):")
        for emb in all_type['embedding'].unique():
            emb_results = all_type[all_type['embedding'] == emb]
            print(f"\n  Embedding: {emb}")
            print(f"  {'Eval Type':<15} {'EER':>10} {'F1':>10} {'AUC':>10}")
            print("  " + "-" * 45)
            for _, r in emb_results.iterrows():
                print(f"  {r['eval_type']:<15} {r['eer']:>10.4f} {r['f1']:>10.3f} {r['auc']:>10.3f}")
            avg_eer = emb_results['eer'].mean()
            print(f"  {'Average':<15} {avg_eer:>10.4f}")

    # Reference
    print("\n  Reference: WPT-XLSR-AASIST (AT-ADD paper) = 3.58% avg EER")

    return atadd_df


def main():
    print("=" * 70)
    print("Unified Experiment Results Comparison")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_a = build_table_a()
    table_b = build_table_b()
    table_c = build_table_c()

    # Save unified report
    report_path = OUTPUT_DIR / 'UNIFIED_COMPARISON.txt'
    with open(report_path, 'w') as f:
        f.write("Unified Experiment Results Comparison\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("TABLE A: Direct Classification (Off-the-Shelf vs Our MoE)\n")
        f.write("-" * 70 + "\n")
        if not table_a.empty:
            f.write(table_a.to_string(index=False))
        else:
            f.write("No results available.\n")
        f.write("\n\n")

        f.write("TABLE B: Temporal Coherence with Different Embeddings\n")
        f.write("-" * 70 + "\n")
        if not table_b.empty:
            f.write(table_b.to_string(index=False))
        else:
            f.write("No results available.\n")
        f.write("\n\n")

        f.write("TABLE C: AT-ADD Cross-Type Evaluation\n")
        f.write("-" * 70 + "\n")
        if not table_c.empty:
            f.write(table_c.to_string(index=False))
        else:
            f.write("No results available.\n")

    print(f"\nUnified report saved to: {report_path}")

    # Save individual CSVs
    if not table_a.empty:
        table_a.to_csv(OUTPUT_DIR / 'table_a_direct_classification.csv', index=False)
    if not table_b.empty:
        table_b.to_csv(OUTPUT_DIR / 'table_b_temporal_embeddings.csv', index=False)
    if not table_c.empty:
        table_c.to_csv(OUTPUT_DIR / 'table_c_atadd_crosstype.csv', index=False)

    print("Individual CSVs saved to comparison directory.")


if __name__ == '__main__':
    main()
