"""Generalization testing framework for audio deepfake detection.

Tests trained models on unseen evaluation datasets to measure
real-world generalizability and robustness.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pickle
import sys

# Add training module to path for ALL_FEATURES
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.data_loader import ALL_FEATURES

from .metrics import compute_binary_metrics, compute_confusion_matrix


@dataclass
class EvaluationResult:
    """Results from evaluating a model on a dataset."""
    model_name: str
    dataset_name: str
    embedding_model: str
    audio_type: str

    # Sample counts
    n_samples: int
    n_real: int
    n_fake: int

    # Metrics
    metrics: Dict[str, float]
    confusion_matrix: Dict[str, Any]

    # Training comparison
    training_f1: Optional[float] = None
    f1_drop: Optional[float] = None
    f1_drop_pct: Optional[float] = None

    # Confidence analysis
    confidence_stats: Optional[Dict[str, float]] = None

    # Error analysis
    error_analysis: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'embedding_model': self.embedding_model,
            'audio_type': self.audio_type,
            'n_samples': self.n_samples,
            'n_real': self.n_real,
            'n_fake': self.n_fake,
            'metrics': self.metrics,
            'confusion_matrix': self.confusion_matrix,
            'training_f1': self.training_f1,
            'f1_drop': self.f1_drop,
            'f1_drop_pct': self.f1_drop_pct,
            'confidence_stats': self.confidence_stats,
            'error_analysis': self.error_analysis,
            'timestamp': self.timestamp,
        }


class GeneralizationTester:
    """Test trained models on unseen evaluation datasets."""

    def __init__(
        self,
        model_path: Path,
        embedding_model: str = 'msclap',
        verbose: bool = True,
    ):
        """Initialize tester.

        Args:
            model_path: Path to trained model pickle file
            embedding_model: Embedding model used (for matching)
            verbose: Print progress
        """
        self.model_path = Path(model_path)
        self.embedding_model = embedding_model
        self.verbose = verbose

        # Load model
        self.model = None
        self.features = None
        self.config = None
        self.scaler = None  # v2+: RobustScaler for normalization
        self._load_model()

    def _load_model(self):
        """Load trained model from pickle file."""
        if self.verbose:
            print(f"Loading model from: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.features = data['features']
        self.config = data.get('config', {})
        self.scaler = data.get('scaler', None)  # v2+: Load scaler if present

        if self.verbose:
            print(f"  Model loaded with {len(self.features)} features: {self.features}")
            if self.scaler is not None:
                print(f"  Scaler loaded: RobustScaler (v2+ model)")

    def evaluate_dataset(
        self,
        features_path: Path,
        dataset_name: str,
        audio_type: str = 'single_voice',
        training_f1: Optional[float] = None,
    ) -> EvaluationResult:
        """Evaluate model on a dataset.

        Args:
            features_path: Path to JSON file with extracted features
            dataset_name: Name of the evaluation dataset
            audio_type: Audio type for this dataset
            training_f1: F1 score from training (for comparison)

        Returns:
            EvaluationResult with all metrics
        """
        if self.verbose:
            print(f"\nEvaluating on: {dataset_name}")
            print(f"Features file: {features_path}")

        # Load features
        with open(features_path, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        y = df['label'].values

        # Apply scaler if present (v2+ models)
        # Scaler was fitted on ALL_FEATURES, so we need to transform all features first
        if self.scaler is not None:
            # Extract all features the scaler was trained on
            X_all = df[ALL_FEATURES].copy()
            X_all = X_all.fillna(X_all.median())

            # Apply scaler to all features
            X_all_scaled = self.scaler.transform(X_all.values)
            X_all_df = pd.DataFrame(X_all_scaled, columns=ALL_FEATURES, index=df.index)

            # Select only the model's features
            X = X_all_df[self.features].copy()
            if self.verbose:
                print(f"  Applied RobustScaler normalization")
        else:
            # No scaler - just use model's features directly
            X = df[self.features].copy()
            X = X.fillna(X.median())

        if self.verbose:
            print(f"  Samples: {len(y)} (Real: {np.sum(y==0)}, Fake: {np.sum(y==1)})")

        # Predict
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        # Compute metrics
        metrics = compute_binary_metrics(y, y_pred, y_proba)
        cm = compute_confusion_matrix(y, y_pred)

        if self.verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  AUC-ROC: {metrics.get('auc_roc', 0):.4f}")

        # Confidence analysis
        confidence_stats = self._analyze_confidence(y, y_pred, y_proba)

        # Error analysis
        error_analysis = self._analyze_errors(y, y_pred, y_proba, df)

        # Compare to training
        f1_drop = None
        f1_drop_pct = None
        if training_f1 is not None:
            f1_drop = training_f1 - metrics['f1']
            f1_drop_pct = (f1_drop / training_f1) * 100 if training_f1 > 0 else 0
            if self.verbose:
                print(f"  F1 drop from training: {f1_drop:.4f} ({f1_drop_pct:.1f}%)")

        return EvaluationResult(
            model_name=self.model_path.stem,
            dataset_name=dataset_name,
            embedding_model=self.embedding_model,
            audio_type=audio_type,
            n_samples=len(y),
            n_real=int(np.sum(y == 0)),
            n_fake=int(np.sum(y == 1)),
            metrics=metrics,
            confusion_matrix=cm,
            training_f1=training_f1,
            f1_drop=f1_drop,
            f1_drop_pct=f1_drop_pct,
            confidence_stats=confidence_stats,
            error_analysis=error_analysis,
        )

    def _analyze_confidence(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Analyze prediction confidence.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of confidence statistics
        """
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        stats = {
            'mean_confidence': float(np.mean(np.maximum(y_proba, 1 - y_proba))),
            'mean_confidence_correct': float(np.mean(np.maximum(y_proba[correct_mask], 1 - y_proba[correct_mask]))) if correct_mask.any() else 0,
            'mean_confidence_incorrect': float(np.mean(np.maximum(y_proba[incorrect_mask], 1 - y_proba[incorrect_mask]))) if incorrect_mask.any() else 0,
        }

        # Confidence distribution
        stats['pct_high_confidence'] = float(np.mean(np.maximum(y_proba, 1 - y_proba) > 0.9))
        stats['pct_low_confidence'] = float(np.mean(np.maximum(y_proba, 1 - y_proba) < 0.6))

        # Calibration: Are high-confidence predictions more accurate?
        high_conf_mask = np.maximum(y_proba, 1 - y_proba) > 0.9
        if high_conf_mask.any():
            stats['accuracy_high_confidence'] = float(np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask]))
        else:
            stats['accuracy_high_confidence'] = 0

        return stats

    def _analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Analyze misclassifications.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            df: Full DataFrame with metadata

        Returns:
            Dictionary of error analysis
        """
        # Basic error counts
        fp_mask = (y_true == 0) & (y_pred == 1)  # Real classified as fake
        fn_mask = (y_true == 1) & (y_pred == 0)  # Fake classified as real

        analysis = {
            'n_false_positives': int(fp_mask.sum()),
            'n_false_negatives': int(fn_mask.sum()),
            'false_positive_rate': float(fp_mask.sum() / (y_true == 0).sum()) if (y_true == 0).any() else 0,
            'false_negative_rate': float(fn_mask.sum() / (y_true == 1).sum()) if (y_true == 1).any() else 0,
        }

        # Confidence of errors
        if fp_mask.any():
            analysis['fp_mean_confidence'] = float(np.mean(y_proba[fp_mask]))
        if fn_mask.any():
            analysis['fn_mean_confidence'] = float(np.mean(1 - y_proba[fn_mask]))

        # If metadata available, analyze by category
        if 'generator' in df.columns:
            analysis['errors_by_generator'] = self._count_errors_by_group(
                df, fn_mask, 'generator'
            )

        if 'language' in df.columns:
            analysis['errors_by_language'] = self._count_errors_by_group(
                df, fn_mask | fp_mask, 'language'
            )

        if 'source' in df.columns:
            analysis['errors_by_source'] = self._count_errors_by_group(
                df, fn_mask | fp_mask, 'source'
            )

        return analysis

    def _count_errors_by_group(
        self,
        df: pd.DataFrame,
        error_mask: np.ndarray,
        group_col: str,
    ) -> Dict[str, int]:
        """Count errors by group category."""
        if group_col not in df.columns:
            return {}

        error_df = df[error_mask]
        return error_df[group_col].value_counts().to_dict()

    def evaluate_by_subset(
        self,
        features_path: Path,
        group_by: str,
        dataset_name: str = 'unknown',
    ) -> pd.DataFrame:
        """Evaluate model with breakdown by subset.

        Args:
            features_path: Path to features JSON
            group_by: Column to group by ('generator', 'language', 'source')
            dataset_name: Name of dataset

        Returns:
            DataFrame with metrics per group
        """
        if self.verbose:
            print(f"\nEvaluating by subset: {group_by}")

        # Load features
        with open(features_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        if group_by not in df.columns:
            print(f"Warning: '{group_by}' not in dataset columns")
            return pd.DataFrame()

        # Evaluate each group
        results = []
        for group_name, group_df in df.groupby(group_by):
            if len(group_df) < 10:  # Skip small groups
                continue

            y = group_df['label'].values

            # Apply scaler if present (v2+ models)
            if self.scaler is not None:
                X_all = group_df[ALL_FEATURES].copy().fillna(group_df[ALL_FEATURES].median())
                X_all_scaled = self.scaler.transform(X_all.values)
                X_all_df = pd.DataFrame(X_all_scaled, columns=ALL_FEATURES, index=group_df.index)
                X = X_all_df[self.features].copy()
            else:
                X = group_df[self.features].copy().fillna(group_df[self.features].median())

            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)[:, 1]

            metrics = compute_binary_metrics(y, y_pred, y_proba)
            metrics['group'] = group_name
            metrics['n_samples'] = len(y)
            metrics['n_real'] = int(np.sum(y == 0))
            metrics['n_fake'] = int(np.sum(y == 1))
            results.append(metrics)

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('f1', ascending=False)

        return results_df


def load_training_results(results_path: Path) -> Dict[str, float]:
    """Load training results to get baseline F1 scores.

    Args:
        results_path: Path to training results.json

    Returns:
        Dictionary mapping model names to their test F1 scores
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    f1_scores = {}
    for model_name, result in results.get('training_results', {}).items():
        test_scores = result.get('test_scores', {})
        f1_scores[model_name] = test_scores.get('f1', 0)

    return f1_scores


def generate_evaluation_report(
    results: List[EvaluationResult],
    output_path: Path,
    training_results_path: Optional[Path] = None,
) -> str:
    """Generate markdown evaluation report.

    Args:
        results: List of evaluation results
        output_path: Path to save report
        training_results_path: Path to training results for comparison

    Returns:
        Path to generated report
    """
    # Load training results for comparison
    training_f1 = {}
    if training_results_path and training_results_path.exists():
        training_f1 = load_training_results(training_results_path)

    lines = [
        "# Audio Deepfake Detection: Generalization Evaluation",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Executive Summary",
        "",
    ]

    # Summary stats
    if results:
        avg_f1 = np.mean([r.metrics['f1'] for r in results])
        avg_auc = np.mean([r.metrics.get('auc_roc', 0) for r in results])
        lines.append(f"- **Models tested:** {len(set(r.model_name for r in results))}")
        lines.append(f"- **Datasets evaluated:** {len(set(r.dataset_name for r in results))}")
        lines.append(f"- **Average F1:** {avg_f1:.4f}")
        lines.append(f"- **Average AUC-ROC:** {avg_auc:.4f}")

        # Find biggest performance drop
        drops = [(r.dataset_name, r.f1_drop_pct) for r in results if r.f1_drop_pct is not None]
        if drops:
            worst = max(drops, key=lambda x: x[1] if x[1] else 0)
            lines.append(f"- **Largest F1 drop:** {worst[1]:.1f}% on {worst[0]}")

    lines.extend([
        "",
        "---",
        "",
        "## 1. Overall Results",
        "",
        "| Model | Dataset | Samples | Accuracy | F1 | AUC-ROC | F1 Drop |",
        "|-------|---------|---------|----------|----|---------|---------| ",
    ])

    for r in sorted(results, key=lambda x: x.metrics['f1'], reverse=True):
        drop_str = f"{r.f1_drop_pct:.1f}%" if r.f1_drop_pct is not None else "N/A"
        lines.append(
            f"| {r.model_name} | {r.dataset_name} | {r.n_samples:,} | "
            f"{r.metrics['accuracy']:.3f} | {r.metrics['f1']:.3f} | "
            f"{r.metrics.get('auc_roc', 0):.3f} | {drop_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Class Performance",
        "",
        "| Model | Dataset | Real F1 | Fake F1 | Balance | FPR | FNR |",
        "|-------|---------|---------|---------|---------|-----|-----|",
    ])

    for r in results:
        fpr = r.error_analysis.get('false_positive_rate', 0) if r.error_analysis else 0
        fnr = r.error_analysis.get('false_negative_rate', 0) if r.error_analysis else 0
        lines.append(
            f"| {r.model_name} | {r.dataset_name} | "
            f"{r.metrics['f1_real']:.3f} | {r.metrics['f1_fake']:.3f} | "
            f"{r.metrics['class_balance_ratio']:.3f} | {fpr:.3f} | {fnr:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confidence Analysis",
        "",
        "| Model | Dataset | Mean Conf | Conf (Correct) | Conf (Wrong) | High Conf % |",
        "|-------|---------|-----------|----------------|--------------|-------------|",
    ])

    for r in results:
        if r.confidence_stats:
            cs = r.confidence_stats
            lines.append(
                f"| {r.model_name} | {r.dataset_name} | "
                f"{cs['mean_confidence']:.3f} | {cs['mean_confidence_correct']:.3f} | "
                f"{cs['mean_confidence_incorrect']:.3f} | {cs['pct_high_confidence']*100:.1f}% |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Confusion Matrices",
        "",
    ])

    for r in results:
        cm = r.confusion_matrix
        lines.extend([
            f"### {r.model_name} on {r.dataset_name}",
            "",
            "```",
            f"               Predicted",
            f"             Real    Fake",
            f"Actual Real  {cm['tn']:5d}   {cm['fp']:5d}",
            f"       Fake  {cm['fn']:5d}   {cm['tp']:5d}",
            "```",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 5. Recommendations",
        "",
    ])

    # Generate recommendations based on results
    if results:
        avg_drop = np.mean([r.f1_drop_pct for r in results if r.f1_drop_pct is not None])
        if avg_drop and avg_drop > 20:
            lines.append("- **Warning:** Significant performance drop on unseen data (>20%)")
            lines.append("- Consider: More diverse training data, domain adaptation techniques")
        elif avg_drop and avg_drop > 10:
            lines.append("- **Moderate generalization gap** (~10-20% drop)")
            lines.append("- Models may need fine-tuning for specific deployment domains")
        else:
            lines.append("- **Good generalization** (<10% drop)")
            lines.append("- Models appear robust to distribution shift")

    # Write report
    report_path = output_path / "GENERALIZATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return str(report_path)
