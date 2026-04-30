"""Training module for audio deepfake detection."""

from .data_loader import (
    AUDIO_TYPES,
    EMBEDDING_MODELS,
    ALL_FEATURES,
    get_feature_names,
    load_combined_features,
    load_all_audio_types,
    prepare_feature_matrix,
    balance_classes,
    load_and_prepare,
    get_dataset_stats,
)
from .feature_selector import (
    GreedyFeatureSelector,
    select_features,
    compute_feature_stability,
)
from .trainer import (
    TrainingConfig,
    TrainingResult,
    Trainer,
    V2_2_CONFIG,
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
