#!/usr/bin/env python3
"""Run generalization evaluation on unseen datasets.

Evaluates trained audio deepfake detection models on external evaluation
datasets to test real-world generalizability.

Usage:
    python scripts/run_generalization_eval.py --dataset asvspoof5
    python scripts/run_generalization_eval.py --all
    python scripts/run_generalization_eval.py --list

Examples:
    # Evaluate universal model on all available datasets
    python scripts/run_generalization_eval.py --all

    # Evaluate specific model on specific dataset
    python scripts/run_generalization_eval.py --model universal_msclap --dataset sonics

    # List available configurations
    python scripts/run_generalization_eval.py --list
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    GeneralizationTester,
    EvaluationResult,
    load_training_results,
    generate_evaluation_report,
)


# Evaluation configurations
# Maps trained models to the datasets they should be evaluated on
EVAL_CONFIGS = [
    # Speech evaluations with single_voice model
    {
        'model': 'sv_msclap',
        'dataset': 'asvspoof5',
        'audio_type': 'single_voice',
        'description': 'Speech model on ASVspoof5 (2024 TTS)',
    },
    {
        'model': 'sv_msclap',
        'dataset': 'deepfake_eval_2024',
        'audio_type': 'single_voice',
        'description': 'Speech model on in-the-wild deepfakes',
    },
    {
        'model': 'sv_msclap',
        'dataset': 'mlaad',
        'audio_type': 'single_voice',
        'description': 'Speech model on multilingual data',
    },
    {
        'model': 'sv_msclap',
        'dataset': 'in_the_wild',
        'audio_type': 'single_voice',
        'description': 'Speech model on celebrity deepfakes',
    },
    {
        'model': 'ds_msclap',
        'dataset': 'in_the_wild',
        'audio_type': 'single_voice',
        'description': 'DeepSpeak model on celebrity deepfakes',
    },
    {
        'model': 'ds_msclap',
        'dataset': 'asvspoof5',
        'audio_type': 'single_voice',
        'description': 'DeepSpeak model on ASVspoof5',
    },
    # Combined single-voice models (SV + DS data)
    {
        'model': 'sv_ds_msclap',
        'dataset': 'in_the_wild',
        'audio_type': 'single_voice',
        'description': 'Combined SV+DS model on celebrity deepfakes',
    },
    {
        'model': 'sv_ds_msclap',
        'dataset': 'asvspoof5',
        'audio_type': 'single_voice',
        'description': 'Combined SV+DS model on ASVspoof5',
    },
    {
        'model': 'universal_msclap',
        'dataset': 'in_the_wild',
        'audio_type': 'single_voice',
        'description': 'Universal model on celebrity deepfakes',
    },

    # Music evaluations
    {
        'model': 'mi_msclap',
        'dataset': 'sonics',
        'audio_type': 'music_instrumental',
        'description': 'Instrumental music model on SONICS',
    },
    {
        'model': 'mv_msclap',
        'dataset': 'sonics',
        'audio_type': 'music_with_vocals',
        'description': 'Vocals music model on SONICS',
    },

    # Universal model on all datasets
    {
        'model': 'universal_msclap',
        'dataset': 'asvspoof5',
        'audio_type': 'single_voice',
        'description': 'Universal model on ASVspoof5',
    },
    {
        'model': 'universal_msclap',
        'dataset': 'deepfake_eval_2024',
        'audio_type': 'single_voice',
        'description': 'Universal model on in-the-wild',
    },
    {
        'model': 'universal_msclap',
        'dataset': 'sonics',
        'audio_type': 'music_with_vocals',
        'description': 'Universal model on SONICS',
    },
    {
        'model': 'universal_msclap',
        'dataset': 'mlaad',
        'audio_type': 'single_voice',
        'description': 'Universal model on multilingual',
    },
]


def find_model_path(model_name: str, models_dir: Path) -> Optional[Path]:
    """Find model pickle file.

    Args:
        model_name: Model name (e.g., 'universal_msclap')
        models_dir: Directory containing model files

    Returns:
        Path to model file or None
    """
    # Try exact match
    model_file = models_dir / f"{model_name}_model.pkl"
    if model_file.exists():
        return model_file

    # Try searching in subdirectories
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            model_file = subdir / 'models' / f"{model_name}_model.pkl"
            if model_file.exists():
                return model_file

    return None


def find_features_path(dataset_name: str, features_dir: Path, embedding: str = 'msclap') -> Optional[Path]:
    """Find features JSON file for a dataset.

    Args:
        dataset_name: Dataset name
        features_dir: Base features directory
        embedding: Embedding model

    Returns:
        Path to features file or None
    """
    # Try combined file
    combined = features_dir / embedding / f"{dataset_name}_combined.json"
    if combined.exists():
        return combined

    # Try in subdirectory
    subdir = features_dir / embedding / dataset_name
    if subdir.exists():
        combined = subdir / "features_combined.json"
        if combined.exists():
            return combined
        # Try any JSON
        jsons = list(subdir.glob("*.json"))
        if jsons:
            return jsons[0]

    return None


def list_configurations():
    """Print available evaluation configurations."""
    print("\n" + "="*70)
    print("Available Evaluation Configurations")
    print("="*70)

    for i, config in enumerate(EVAL_CONFIGS, 1):
        print(f"\n{i}. {config['model']} -> {config['dataset']}")
        print(f"   Audio type: {config['audio_type']}")
        print(f"   Description: {config['description']}")

    print("\n" + "="*70)


def run_evaluation(
    config: Dict[str, str],
    models_dir: Path,
    features_dir: Path,
    training_results_path: Optional[Path] = None,
    verbose: bool = True,
) -> Optional[EvaluationResult]:
    """Run a single evaluation configuration.

    Args:
        config: Evaluation configuration dict
        models_dir: Directory with trained models
        features_dir: Directory with extracted features
        training_results_path: Path to training results for comparison
        verbose: Print progress

    Returns:
        EvaluationResult or None if failed
    """
    model_name = config['model']
    dataset_name = config['dataset']
    audio_type = config['audio_type']

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} on {dataset_name}")
        print(f"{'='*60}")

    # Find model
    model_path = find_model_path(model_name, models_dir)
    if model_path is None:
        print(f"  Model not found: {model_name}")
        print(f"  Searched in: {models_dir}")
        return None

    # Find features
    features_path = find_features_path(dataset_name, features_dir)
    if features_path is None:
        print(f"  Features not found for: {dataset_name}")
        print(f"  Searched in: {features_dir}")
        print(f"  Hint: Run feature extraction first with scripts/extract_features.py")
        return None

    # Get training F1 for comparison
    training_f1 = None
    if training_results_path and training_results_path.exists():
        training_scores = load_training_results(training_results_path)
        training_f1 = training_scores.get(model_name)

    # Run evaluation
    try:
        tester = GeneralizationTester(
            model_path=model_path,
            embedding_model='msclap',
            verbose=verbose,
        )

        result = tester.evaluate_dataset(
            features_path=features_path,
            dataset_name=dataset_name,
            audio_type=audio_type,
            training_f1=training_f1,
        )

        return result

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run generalization evaluation on unseen datasets'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Specific dataset to evaluate on'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Specific model to evaluate'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all evaluation configurations'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('data/training_results'),
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--features-dir',
        type=Path,
        default=Path('data/eval_features'),
        help='Directory containing extracted features'
    )
    parser.add_argument(
        '--training-results',
        type=Path,
        default=None,
        help='Path to training results.json for F1 comparison'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available configurations'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output'
    )
    args = parser.parse_args()

    if args.list:
        list_configurations()
        return

    if not args.all and not args.dataset and not args.model:
        parser.print_help()
        print("\nUse --list to see available configurations")
        return

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('data/eval_results') / timestamp
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Find training results for comparison
    training_results = args.training_results
    if training_results is None:
        # Try to find latest training results
        training_dir = Path('data/training_results')
        if training_dir.exists():
            subdirs = sorted([d for d in training_dir.iterdir() if d.is_dir()])
            for subdir in reversed(subdirs):
                results_file = subdir / 'results.json'
                if results_file.exists():
                    training_results = results_file
                    print(f"Using training results: {training_results}")
                    break

    # Filter configurations
    configs_to_run = EVAL_CONFIGS.copy()

    if args.dataset:
        configs_to_run = [c for c in configs_to_run if c['dataset'] == args.dataset]

    if args.model:
        configs_to_run = [c for c in configs_to_run if c['model'] == args.model]

    if not configs_to_run:
        print("No matching configurations found")
        return

    print(f"\nRunning {len(configs_to_run)} evaluations:")
    for config in configs_to_run:
        print(f"  - {config['model']} -> {config['dataset']}")

    # Run evaluations
    all_results = []
    for config in configs_to_run:
        result = run_evaluation(
            config,
            args.models_dir,
            args.features_dir,
            training_results,
            verbose=not args.quiet,
        )
        if result:
            all_results.append(result)

            # Save individual result
            result_path = output_dir / 'results' / f"{result.model_name}_{result.dataset_name}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

    # Generate report
    if all_results:
        print("\n" + "="*60)
        print("Generating Report")
        print("="*60)

        report_path = generate_evaluation_report(
            all_results,
            output_dir,
            training_results,
        )
        print(f"Report generated: {report_path}")

        # Save all results
        all_results_path = output_dir / 'all_results.json'
        with open(all_results_path, 'w') as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"Results saved: {all_results_path}")

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

    if all_results:
        print("\n| Model | Dataset | F1 | AUC-ROC | F1 Drop |")
        print("|-------|---------|----|---------|---------|")
        for r in sorted(all_results, key=lambda x: x.metrics['f1'], reverse=True):
            drop = f"{r.f1_drop_pct:.1f}%" if r.f1_drop_pct else "N/A"
            print(f"| {r.model_name} | {r.dataset_name} | "
                  f"{r.metrics['f1']:.3f} | {r.metrics.get('auc_roc', 0):.3f} | {drop} |")
    else:
        print("\nNo evaluations completed successfully.")
        print("Make sure features have been extracted for the evaluation datasets.")
        print("\nTo extract features, run:")
        print("  python scripts/extract_features.py <dataset_dir> ...")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
