"""Evaluation utilities for gated ensemble classifiers.

Tests gated ensembles on evaluation datasets and generates comparison reports.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_binary_metrics, compute_confusion_matrix
from training.data_loader import ALL_FEATURES
from training.gated_ensemble import (
    GatedEnsembleClassifier,
    create_gated_ensemble,
    list_gated_ensembles,
)


@dataclass
class GatedEvaluationResult:
    """Results from evaluating a gated ensemble on a dataset."""
    ensemble_name: str
    dataset_name: str
    gating_method: str

    # Sample counts
    n_samples: int
    n_real: int
    n_fake: int

    # Metrics
    metrics: Dict[str, float]
    confusion_matrix: Dict[str, Any]

    # Routing statistics
    routing_stats: Dict[str, Any]

    # Per-expert breakdown
    expert_metrics: Optional[Dict[str, Dict[str, float]]] = None

    # Comparison to baseline
    baseline_f1: Optional[float] = None
    f1_improvement: Optional[float] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ensemble_name': self.ensemble_name,
            'dataset_name': self.dataset_name,
            'gating_method': self.gating_method,
            'n_samples': self.n_samples,
            'n_real': self.n_real,
            'n_fake': self.n_fake,
            'metrics': self.metrics,
            'confusion_matrix': self.confusion_matrix,
            'routing_stats': self.routing_stats,
            'expert_metrics': self.expert_metrics,
            'baseline_f1': self.baseline_f1,
            'f1_improvement': self.f1_improvement,
            'timestamp': self.timestamp,
        }


class GatedEnsembleEvaluator:
    """Evaluates gated ensembles on test datasets."""

    def __init__(
        self,
        models_dir: Path,
        verbose: bool = True,
    ):
        """Initialize evaluator.

        Args:
            models_dir: Directory containing expert model files
            verbose: Print progress
        """
        self.models_dir = Path(models_dir)
        self.verbose = verbose

    def evaluate_ensemble(
        self,
        ensemble: GatedEnsembleClassifier,
        features_path: Path,
        dataset_name: str,
        baseline_f1: Optional[float] = None,
        threshold: float = 0.5,
    ) -> GatedEvaluationResult:
        """Evaluate a gated ensemble on a dataset.

        Args:
            ensemble: Gated ensemble to evaluate
            features_path: Path to JSON features file
            dataset_name: Name of the dataset
            baseline_f1: Baseline F1 to compare against
            threshold: Classification threshold

        Returns:
            GatedEvaluationResult with all metrics
        """
        if self.verbose:
            print(f"\nEvaluating {ensemble.config.name} on {dataset_name}")
            print(f"  Gating method: {ensemble.config.gating_method}")

        # Load features
        with open(features_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        y = df['label'].values

        # Prepare features (fill NaN with median)
        X = df[ALL_FEATURES].copy()
        X = X.fillna(X.median())

        if self.verbose:
            print(f"  Samples: {len(y)} (Real: {np.sum(y==0)}, Fake: {np.sum(y==1)})")

        # Get predictions with routing info
        y_proba, routing = ensemble.predict_proba(X, return_routing=True)
        y_pred = (y_proba >= threshold).astype(int)

        # Compute metrics
        metrics = compute_binary_metrics(y, y_pred, y_proba)
        cm = compute_confusion_matrix(y, y_pred)

        if self.verbose:
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")

        # Routing statistics
        routing_stats = self._analyze_routing(ensemble, routing, y, y_pred, y_proba)

        if self.verbose:
            print("  Routing distribution:")
            for expert_name, count in routing_stats['expert_counts'].items():
                pct = count / len(y) * 100
                print(f"    {expert_name}: {count} ({pct:.1f}%)")

        # Per-expert metrics (for samples routed to each expert)
        expert_metrics = self._compute_expert_metrics(
            ensemble, routing, y, y_pred, y_proba, df
        )

        # Comparison to baseline
        f1_improvement = None
        if baseline_f1 is not None:
            f1_improvement = metrics['f1'] - baseline_f1
            if self.verbose:
                sign = '+' if f1_improvement >= 0 else ''
                print(f"  F1 vs baseline: {sign}{f1_improvement:.4f}")

        return GatedEvaluationResult(
            ensemble_name=ensemble.config.name,
            dataset_name=dataset_name,
            gating_method=ensemble.config.gating_method,
            n_samples=len(y),
            n_real=int(np.sum(y == 0)),
            n_fake=int(np.sum(y == 1)),
            metrics=metrics,
            confusion_matrix=cm,
            routing_stats=routing_stats,
            expert_metrics=expert_metrics,
            baseline_f1=baseline_f1,
            f1_improvement=f1_improvement,
        )

    def _analyze_routing(
        self,
        ensemble: GatedEnsembleClassifier,
        routing: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze routing decisions.

        Args:
            ensemble: The gated ensemble
            routing: Expert indices for each sample
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of routing statistics
        """
        stats = {}

        # Count per expert
        expert_counts = {}
        for i, name in enumerate(ensemble.config.model_names):
            expert_counts[name] = int(np.sum(routing == i))
        stats['expert_counts'] = expert_counts

        # Per-class routing
        real_routing = {}
        fake_routing = {}
        for i, name in enumerate(ensemble.config.model_names):
            real_mask = (y_true == 0) & (routing == i)
            fake_mask = (y_true == 1) & (routing == i)
            real_routing[name] = int(np.sum(real_mask))
            fake_routing[name] = int(np.sum(fake_mask))
        stats['real_routing'] = real_routing
        stats['fake_routing'] = fake_routing

        # Routing effectiveness: accuracy per expert
        expert_accuracy = {}
        for i, name in enumerate(ensemble.config.model_names):
            mask = routing == i
            if mask.any():
                expert_accuracy[name] = float(np.mean(y_true[mask] == y_pred[mask]))
            else:
                expert_accuracy[name] = None
        stats['expert_accuracy'] = expert_accuracy

        # Confidence by routing
        expert_confidence = {}
        for i, name in enumerate(ensemble.config.model_names):
            mask = routing == i
            if mask.any():
                # Confidence = distance from 0.5
                conf = np.abs(y_proba[mask] - 0.5) * 2
                expert_confidence[name] = {
                    'mean': float(np.mean(conf)),
                    'std': float(np.std(conf)),
                    'high_conf_pct': float(np.mean(conf > 0.8)),
                }
            else:
                expert_confidence[name] = None
        stats['expert_confidence'] = expert_confidence

        return stats

    def _compute_expert_metrics(
        self,
        ensemble: GatedEnsembleClassifier,
        routing: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Compute detailed metrics for samples routed to each expert.

        Args:
            ensemble: The gated ensemble
            routing: Expert indices
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            df: Full DataFrame with metadata

        Returns:
            Dictionary mapping expert name to metrics
        """
        expert_metrics = {}

        for i, name in enumerate(ensemble.config.model_names):
            mask = routing == i
            if not mask.any():
                continue

            y_true_expert = y_true[mask]
            y_pred_expert = y_pred[mask]
            y_proba_expert = y_proba[mask]

            metrics = compute_binary_metrics(y_true_expert, y_pred_expert, y_proba_expert)
            metrics['n_samples'] = int(mask.sum())
            metrics['n_real'] = int(np.sum(y_true_expert == 0))
            metrics['n_fake'] = int(np.sum(y_true_expert == 1))

            # Feature statistics for routed samples
            X_routed = df[ALL_FEATURES].iloc[mask]
            metrics['mean_similarity_mean'] = float(X_routed['mean'].mean())
            metrics['mean_similarity_std'] = float(X_routed['mean'].std())

            expert_metrics[name] = metrics

        return expert_metrics

    def compare_ensembles(
        self,
        features_path: Path,
        dataset_name: str,
        config_names: Optional[List[str]] = None,
        fit_gating: bool = True,
        fit_data_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Compare multiple gated ensemble configurations on a dataset.

        Args:
            features_path: Path to evaluation features
            dataset_name: Name of dataset
            config_names: List of config names to compare (None = all)
            fit_gating: Whether to fit gating networks before evaluation
            fit_data_path: Path to training data for fitting gating (if different from eval)

        Returns:
            DataFrame comparing all ensembles
        """
        if config_names is None:
            config_names = list_gated_ensembles()

        results = []

        for config_name in config_names:
            try:
                ensemble = create_gated_ensemble(
                    self.models_dir, config_name, verbose=self.verbose
                )

                # Fit gating if needed
                if fit_gating and ensemble.config.gating_method in ['learned', 'feature_profile']:
                    fit_path = fit_data_path or features_path
                    with open(fit_path, 'r') as f:
                        fit_data = json.load(f)
                    fit_df = pd.DataFrame(fit_data)
                    X_fit = fit_df[ALL_FEATURES].fillna(fit_df[ALL_FEATURES].median())
                    y_fit = fit_df['label'].values
                    ensemble.fit_gating(X_fit, y_fit, method='oracle')

                # Evaluate
                result = self.evaluate_ensemble(ensemble, features_path, dataset_name)
                results.append({
                    'config_name': config_name,
                    'gating_method': result.gating_method,
                    'f1': result.metrics['f1'],
                    'accuracy': result.metrics['accuracy'],
                    'auc_roc': result.metrics.get('auc_roc', 0),
                    'f1_real': result.metrics['f1_real'],
                    'f1_fake': result.metrics['f1_fake'],
                    'n_samples': result.n_samples,
                })

            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating {config_name}: {e}")
                continue

        return pd.DataFrame(results).sort_values('f1', ascending=False)


def generate_gated_ensemble_report(
    results: List[GatedEvaluationResult],
    output_path: Path,
    baseline_results: Optional[Dict[str, float]] = None,
) -> str:
    """Generate markdown report for gated ensemble evaluation.

    Args:
        results: List of evaluation results
        output_path: Directory to save report
        baseline_results: Dict mapping model name to baseline F1

    Returns:
        Path to generated report
    """
    lines = [
        "# Gated Ensemble Evaluation Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        "",
    ]

    if results:
        best = max(results, key=lambda r: r.metrics['f1'])
        lines.extend([
            f"- **Best ensemble:** {best.ensemble_name} (F1={best.metrics['f1']:.4f})",
            f"- **Gating method:** {best.gating_method}",
            f"- **Total ensembles evaluated:** {len(results)}",
        ])

        if best.f1_improvement is not None:
            sign = '+' if best.f1_improvement >= 0 else ''
            lines.append(f"- **F1 improvement over baseline:** {sign}{best.f1_improvement:.4f}")

    lines.extend([
        "",
        "---",
        "",
        "## Results Comparison",
        "",
        "| Ensemble | Dataset | Method | F1 | Accuracy | AUC | F1 Improve |",
        "|----------|---------|--------|----|---------|----|------------|",
    ])

    for r in sorted(results, key=lambda x: x.metrics['f1'], reverse=True):
        improve_str = f"{r.f1_improvement:+.4f}" if r.f1_improvement is not None else "N/A"
        lines.append(
            f"| {r.ensemble_name} | {r.dataset_name} | {r.gating_method} | "
            f"{r.metrics['f1']:.4f} | {r.metrics['accuracy']:.4f} | "
            f"{r.metrics.get('auc_roc', 0):.4f} | {improve_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Routing Analysis",
        "",
    ])

    for r in results:
        lines.extend([
            f"### {r.ensemble_name}",
            "",
            "**Expert routing distribution:**",
            "",
            "| Expert | Samples | Real | Fake | Accuracy |",
            "|--------|---------|------|------|----------|",
        ])

        for expert_name, count in r.routing_stats['expert_counts'].items():
            real_count = r.routing_stats['real_routing'].get(expert_name, 0)
            fake_count = r.routing_stats['fake_routing'].get(expert_name, 0)
            acc = r.routing_stats['expert_accuracy'].get(expert_name)
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            lines.append(f"| {expert_name} | {count} | {real_count} | {fake_count} | {acc_str} |")

        lines.append("")

    # Per-expert metrics
    lines.extend([
        "---",
        "",
        "## Per-Expert Performance",
        "",
    ])

    for r in results:
        if r.expert_metrics:
            lines.extend([
                f"### {r.ensemble_name}",
                "",
                "| Expert | F1 | Precision | Recall | Mean Sim |",
                "|--------|----|-----------|--------|----------|",
            ])

            for expert_name, metrics in r.expert_metrics.items():
                mean_sim = metrics.get('mean_similarity_mean', 0)
                lines.append(
                    f"| {expert_name} | {metrics['f1']:.4f} | "
                    f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{mean_sim:.4f} |"
                )

            lines.append("")

    lines.extend([
        "---",
        "",
        "## Recommendations",
        "",
    ])

    if results:
        best = max(results, key=lambda r: r.metrics['f1'])
        if best.f1_improvement is not None and best.f1_improvement > 0.02:
            lines.extend([
                f"- **Significant improvement** with {best.ensemble_name}",
                "- The gated ensemble successfully adapts to domain shift",
                "- Consider deploying this ensemble for production use",
            ])
        elif best.f1_improvement is not None and best.f1_improvement > 0:
            lines.extend([
                f"- **Modest improvement** with {best.ensemble_name}",
                "- Gating provides some benefit, consider tuning further",
            ])
        else:
            lines.extend([
                "- Gated ensembles did not improve over baseline",
                "- Consider: different expert combinations, more training data for gating",
            ])

    # Write report
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "GATED_ENSEMBLE_REPORT.md"

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return str(report_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate gated ensembles')
    parser.add_argument('--models-dir', type=str, default='models/trained',
                       help='Directory containing trained models')
    parser.add_argument('--eval-data', type=str, required=True,
                       help='Path to evaluation features JSON')
    parser.add_argument('--dataset-name', type=str, default='evaluation',
                       help='Name of evaluation dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Specific ensemble config to evaluate (default: all)')
    parser.add_argument('--output-dir', type=str, default='data/eval_results',
                       help='Directory to save results')
    parser.add_argument('--baseline-f1', type=float, default=None,
                       help='Baseline F1 for comparison')

    args = parser.parse_args()

    evaluator = GatedEnsembleEvaluator(
        models_dir=Path(args.models_dir),
        verbose=True,
    )

    if args.config:
        # Evaluate single config
        ensemble = create_gated_ensemble(
            Path(args.models_dir), args.config, verbose=True
        )

        # Fit gating on eval data (oracle mode)
        with open(args.eval_data, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        X = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
        y = df['label'].values
        ensemble.fit_gating(X, y, method='oracle')

        result = evaluator.evaluate_ensemble(
            ensemble,
            Path(args.eval_data),
            args.dataset_name,
            baseline_f1=args.baseline_f1,
        )
        print(f"\nFinal F1: {result.metrics['f1']:.4f}")

    else:
        # Compare all configs
        comparison = evaluator.compare_ensembles(
            Path(args.eval_data),
            args.dataset_name,
        )
        print("\nComparison Results:")
        print(comparison.to_string())
