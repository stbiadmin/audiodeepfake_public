"""Model module for audio deepfake detection."""

from .ensemble import (
    EnsembleConfig,
    EnsembleClassifier,
    ENSEMBLE_CONFIGS,
    create_ensemble,
    list_ensembles,
)

__all__ = [
    'EnsembleConfig',
    'EnsembleClassifier',
    'ENSEMBLE_CONFIGS',
    'create_ensemble',
    'list_ensembles',
]
