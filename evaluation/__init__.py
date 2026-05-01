"""Evaluation module for audio deepfake detection."""

from .ablation import (
    FEATURE_SETS,
    AblationStudy,
    feature_group_ablation,
    feature_set_comparison,
    incremental_addition_ablation,
    leave_one_out_ablation,
)
from .feature_importance import (
    FeatureImportanceAnalyzer,
    get_combined_importance,
    get_permutation_importance,
    get_shap_importance,
    get_xgboost_importance,
)
from .generalization_test import (
    EvaluationResult,
    GeneralizationTester,
    generate_evaluation_report,
    load_training_results,
)
from .metrics import (
    MetricsTracker,
    aggregate_cv_metrics,
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_pr_curve,
    compute_roc_curve,
    format_metrics_table,
    get_classification_report,
)
from .report_generator import (
    ReportGenerator,
    generate_classification_report,
)

__all__ = [
    # Metrics
    'compute_binary_metrics',
    'compute_confusion_matrix',
    'compute_roc_curve',
    'compute_pr_curve',
    'get_classification_report',
    'aggregate_cv_metrics',
    'format_metrics_table',
    'MetricsTracker',
    # Feature importance
    'get_xgboost_importance',
    'get_shap_importance',
    'get_permutation_importance',
    'get_combined_importance',
    'FeatureImportanceAnalyzer',
    # Ablation
    'FEATURE_SETS',
    'leave_one_out_ablation',
    'incremental_addition_ablation',
    'feature_set_comparison',
    'feature_group_ablation',
    'AblationStudy',
    # Report
    'generate_classification_report',
    'ReportGenerator',
    # Generalization testing
    'EvaluationResult',
    'GeneralizationTester',
    'load_training_results',
    'generate_evaluation_report',
]
