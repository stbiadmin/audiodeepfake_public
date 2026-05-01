"""Classifier models for audio deepfake detection."""

from .xgboost_classifier import (
    AudioDeepfakeClassifier,
    XGBoostHyperparameters,
)

__all__ = [
    'XGBoostHyperparameters',
    'AudioDeepfakeClassifier',
]
