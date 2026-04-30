"""Classifier models for audio deepfake detection."""

from .xgboost_classifier import (
    XGBoostHyperparameters,
    AudioDeepfakeClassifier,
)

__all__ = [
    'XGBoostHyperparameters',
    'AudioDeepfakeClassifier',
]
