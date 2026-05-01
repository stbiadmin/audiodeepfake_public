"""Training module for audio deepfake detection."""

from .data_loader import (
    ALL_FEATURES,
    AUDIO_TYPES,
    EMBEDDING_MODELS,
    balance_classes,
    get_dataset_stats,
    get_feature_names,
    load_all_audio_types,
    load_and_prepare,
    load_combined_features,
    prepare_feature_matrix,
)
from .feature_selector import (
    GreedyFeatureSelector,
    compute_feature_stability,
    select_features,
)
from .trainer import (
    V2_2_CONFIG,
    Trainer,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    'AUDIO_TYPES',
    'EMBEDDING_MODELS',
    'ALL_FEATURES',
    'get_feature_names',
    'load_combined_features',
    'load_all_audio_types',
    'prepare_feature_matrix',
    'balance_classes',
    'load_and_prepare',
    'get_dataset_stats',
    'GreedyFeatureSelector',
    'select_features',
    'compute_feature_stability',
    'TrainingConfig',
    'TrainingResult',
    'Trainer',
    'V2_2_CONFIG',
]
