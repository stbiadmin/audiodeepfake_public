"""Evaluation metrics for audio deepfake detection.

Computes comprehensive classification metrics including per-class performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive binary classification metrics.

    Args:
        y_true: True labels (0=real, 1=fake)
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Weighted F1
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class metrics (real = 0, fake = 1)
    metrics['precision_real'] = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics['recall_real'] = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics['f1_real'] = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    metrics['precision_fake'] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics['recall_fake'] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics['f1_fake'] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Class balance ratio
    if metrics['f1_real'] > 0 and metrics['f1_fake'] > 0:
        metrics['class_balance_ratio'] = min(metrics['f1_real'], metrics['f1_fake']) / \
                                          max(metrics['f1_real'], metrics['f1_fake'])
    else:
        metrics['class_balance_ratio'] = 0.0

    # Probability-based metrics
    if y_proba is not None:
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics['auc_roc'] = 0.5

        try:
            metrics['auc_pr'] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics['auc_pr'] = 0.5

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute confusion matrix with named values.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with confusion matrix and named components
    """
    cm = confusion_matrix(y_true, y_pred)

    # For binary classification: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    return {
        'matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total': int(len(y_true)),
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute ROC curve data.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary with fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }


def compute_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute Precision-Recall curve data.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary with precision, recall, thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist(),
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Get formatted classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for classes (default: ['real', 'fake'])

    Returns:
        Formatted classification report string
    """
    if target_names is None:
        target_names = ['real', 'fake']

    return classification_report(y_true, y_pred, target_names=target_names)


def aggregate_cv_metrics(
    cv_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across cross-validation folds.

    Args:
        cv_metrics: List of metric dictionaries from each fold

    Returns:
        Dictionary with mean and std for each metric
    """
    if not cv_metrics:
        return {}

    metric_names = cv_metrics[0].keys()
    aggregated = {}

    for metric in metric_names:
        values = [m[metric] for m in cv_metrics]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }

    return aggregated


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    metric_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Format results as a comparison table.

    Args:
        results: Dictionary mapping config name to metrics
        metric_order: Order of metrics to display

    Returns:
        DataFrame with configs as rows and metrics as columns
    """
    if metric_order is None:
        metric_order = [
            'accuracy', 'precision', 'recall', 'f1',
            'auc_roc', 'f1_real', 'f1_fake', 'class_balance_ratio'
        ]

    rows = []
    for config_name, metrics in results.items():
        row = {'config': config_name}
        for metric in metric_order:
            if metric in metrics:
                row[metric] = metrics[metric]
        rows.append(row)

    return pd.DataFrame(rows).set_index('config')


class MetricsTracker:
    """Track metrics across multiple training runs."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def add_result(
        self,
        config_name: str,
        metrics: Dict[str, float],
        confusion_matrix: Optional[Dict[str, Any]] = None,
        cv_metrics: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Add results for a training configuration.

        Args:
            config_name: Name of the configuration
            metrics: Test set metrics
            confusion_matrix: Confusion matrix data
            cv_metrics: Cross-validation metrics per fold
        """
        self.results[config_name] = {
            'test_metrics': metrics,
            'confusion_matrix': confusion_matrix,
            'cv_metrics': cv_metrics,
            'cv_aggregated': aggregate_cv_metrics(cv_metrics) if cv_metrics else None,
        }

    def get_best_config(self, metric: str = 'f1') -> Tuple[str, float]:
        """Get the best configuration by a metric.

        Args:
            metric: Metric to optimize

        Returns:
            Tuple of (config_name, metric_value)
        """
        best_config = None
        best_value = -float('inf')

        for config_name, result in self.results.items():
            value = result['test_metrics'].get(metric, 0)
            if value > best_value:
                best_value = value
                best_config = config_name

        return best_config, best_value

    def get_summary_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Get summary table of all results.

        Args:
            metrics: Metrics to include

        Returns:
            DataFrame with results
        """
        if metrics is None:
            metrics = ['accuracy', 'f1', 'auc_roc', 'class_balance_ratio']

        rows = []
        for config_name, result in self.results.items():
            row = {'config': config_name}
            for metric in metrics:
                row[metric] = result['test_metrics'].get(metric, np.nan)
            rows.append(row)

        return pd.DataFrame(rows).set_index('config').round(4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all results to dictionary."""
        return self.results.copy()
