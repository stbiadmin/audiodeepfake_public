#!/usr/bin/env python3
"""Evaluate ensemble classifiers on generalization datasets.

Usage:
    python scripts/evaluate_ensembles.py --models-dir data/training_results/v2.2_full/models

    # Specific ensemble
    python scripts/evaluate_ensembles.py --models-dir data/training_results/v2.2_full/models \
        --ensemble speech_ensemble

    # List available ensembles
    python scripts/evaluate_ensembles.py --list
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ensemble import (
    EnsembleClassifier,
    ENSEMBLE_CONFIGS,
    create_ensemble,
    list_ensembles,
)
from training.data_loader import ALL_FEATURES


# Evaluation datasets for speech models
EVAL_DATASETS = [
    {
        'name': 'in_the_wild',
        'features_path': 'data/eval_features/msclap/in_the_wild/features_combined.json',
        'audio_type': 'single_voice',
    },
    {
        'name': 'asvspoof5',
        'features_path': 'data/eval_features/msclap/asvspoof5/features_combined.json',
        'audio_type': 'single_voice',
    },
]


def load_features(features_path: Path) -> pd.DataFrame:
    """Load features from JSON file."""
    with open(features_path, 'r') as f:
        data = json.load(f)

    # Features are stored directly in each item (not nested under 'features')
    df = pd.DataFrame(data)
    return df


def evaluate_ensemble_on_dataset(
    ensemble: EnsembleClassifier,
    dataset_config: Dict,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate an ensemble on a single dataset.

    Args:
        ensemble: Ensemble classifier
        dataset_config: Dataset configuration dict
        verbose: Print progress

    Returns:
        Evaluation results dictionary
    """
    features_path = Path(dataset_config['features_path'])

    if not features_path.exists():
        if verbose:
            print(f"  Features not found: {features_path}")
        return None

    # Load features
    df = load_features(features_path)

    # Prepare features (fill NaN with median)
    X = df[ALL_FEATURES].copy()
    X = X.fillna(X.median())
    y = df['label'].values

    # Evaluate
    metrics = ensemble.evaluate(X, y)

    # Add metadata
    result = {
        'ensemble_name': ensemble.config.name,
        'dataset_name': dataset_config['name'],
        'n_samples': len(y),
        'n_real': int(np.sum(y == 0)),
        'n_fake': int(np.sum(y == 1)),
        'metrics': metrics,
        'model_weights': dict(zip(
            ensemble.config.model_names,
            ensemble.config.weights
        )),
        'timestamp': datetime.now().isoformat(),
    }

    if verbose:
        print(f"\n  Dataset: {dataset_config['name']}")
        print(f"  Samples: {result['n_samples']} (Real: {result['n_real']}, Fake: {result['n_fake']})")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['roc_auc']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ensemble classifiers on generalization datasets'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('data/training_results/v2.2_full/models'),
        help='Directory containing trained model files'
    )
    parser.add_argument(
        '--ensemble', '-e',
        type=str,
        default='all',
        help='Ensemble to evaluate (or "all")'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/eval_results/v2.3_ensemble'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available ensemble configurations'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output'
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable ensemble configurations:")
        for name, config in ENSEMBLE_CONFIGS.items():
            print(f"\n  {name}:")
            print(f"    Models: {config.model_names}")
            print(f"    Weights: {config.weights}")
        return

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Determine which ensembles to evaluate
    if args.ensemble == 'all':
        ensemble_names = list(ENSEMBLE_CONFIGS.keys())
    else:
        ensemble_names = [args.ensemble]

    print(f"\nEvaluating {len(ensemble_names)} ensemble(s):")
    for name in ensemble_names:
        print(f"  - {name}")

    # Evaluate each ensemble
    all_results = []

    for ensemble_name in ensemble_names:
        print(f"\n{'='*60}")
        print(f"Ensemble: {ensemble_name}")
        print(f"{'='*60}")

        try:
            ensemble = create_ensemble(
                args.models_dir,
                ensemble_name,
                verbose=not args.quiet
            )

            for dataset_config in EVAL_DATASETS:
                result = evaluate_ensemble_on_dataset(
                    ensemble,
                    dataset_config,
                    verbose=not args.quiet
                )
                if result:
                    all_results.append(result)

        except Exception as e:
            print(f"Error evaluating {ensemble_name}: {e}")
            continue

    # Save results
    results_path = args.output_dir / 'ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Generate summary report
    report_path = args.output_dir / 'ENSEMBLE_REPORT.md'
    with open(report_path, 'w') as f:
        f.write("# Ensemble Evaluation Results\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        f.write("## Summary\n\n")
        f.write("| Ensemble | Dataset | Samples | Accuracy | F1 | AUC-ROC |\n")
        f.write("|----------|---------|---------|----------|----|---------|\n")

        for result in sorted(all_results, key=lambda x: -x['metrics']['f1']):
            f.write(f"| {result['ensemble_name']} | {result['dataset_name']} | ")
            f.write(f"{result['n_samples']:,} | ")
            f.write(f"{result['metrics']['accuracy']:.3f} | ")
            f.write(f"{result['metrics']['f1']:.3f} | ")
            f.write(f"{result['metrics']['roc_auc']:.3f} |\n")

        f.write("\n## Ensemble Configurations\n\n")
        for name, config in ENSEMBLE_CONFIGS.items():
            f.write(f"### {name}\n\n")
            f.write("| Model | Weight |\n")
            f.write("|-------|--------|\n")
            for model, weight in zip(config.model_names, config.weights):
                f.write(f"| {model} | {weight:.2f} |\n")
            f.write("\n")

    print(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print("\n| Ensemble | Dataset | F1 | AUC-ROC |")
    print("|----------|---------|----|---------|")
    for result in sorted(all_results, key=lambda x: -x['metrics']['f1']):
        print(f"| {result['ensemble_name']} | {result['dataset_name']} | "
              f"{result['metrics']['f1']:.3f} | {result['metrics']['roc_auc']:.3f} |")


if __name__ == '__main__':
    main()
