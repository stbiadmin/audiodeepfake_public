"""Classifier wrappers for speech and music deepfake detection.

Provides unified interfaces for:
- Single speech model (XGBoost)
- 5-expert speech ensemble
- Music adaptive classifier (threshold-based)
- Music XGBoost classifier
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .model_registry import FEATURE_SETS, SPEECH_ENSEMBLE_CONFIG, get_registry

# Feature orders for different scaler versions
# 27 features (excludes shapiro_p, normaltest_p - used by sv_ds_msclap)
FEATURES_27 = [
    'mean', 'std', 'variance', 'min', 'max', 'peak_to_peak',
    'skewness', 'kurtosis', 'bimodality_coefficient',
    'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr',
    'tail_weight_ratio', 'trimmed_mean', 'entropy', 'gini_coefficient',
    'coefficient_of_variation', 'variance_mean_ratio', 'kurtosis_variance_ratio',
    'skewness_kurtosis_ratio', 'iqr_range_ratio', 'median_mean_diff',
]

# 29 features (standard training/data_loader.py ALL_FEATURES)
FEATURES_29 = FEATURES_27 + ['shapiro_p', 'normaltest_p']

# 30 features (includes n_samples, used by mlaad)
FEATURES_30 = FEATURES_27 + ['n_samples', 'shapiro_p', 'normaltest_p']

# Default to 30 features to support all models (including mlaad which needs n_samples)
ALL_FEATURES_ORDER = FEATURES_30

# Median values for features that may be NaN (from in-the-wild evaluation data)
# Used as defaults when features are missing or cannot be computed
FEATURE_MEDIANS = {
    'shapiro_p': 0.209,
    'normaltest_p': 0.304,
    'bimodality_coefficient': 0.379,
}


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""

    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Make prediction.

        Args:
            features: Dictionary of computed features

        Returns:
            Tuple of (label, confidence) where label is 'real' or 'fake'
        """
        pass

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Get list of features required by this classifier."""
        pass


class SpeechClassifier(BaseClassifier):
    """XGBoost speech classifier wrapper."""

    def __init__(self, model_name: str = 'ds_msclap'):
        """Initialize speech classifier.

        Args:
            model_name: Name of the model (e.g., 'ds_msclap', 'sv_msclap')
        """
        self.model_name = model_name
        self._model_data = None

    def _ensure_loaded(self) -> None:
        """Lazy load model from registry."""
        if self._model_data is None:
            registry = get_registry()
            self._model_data = registry.get_classifier(self.model_name)

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Make prediction using XGBoost model.

        Args:
            features: Dictionary of computed features

        Returns:
            Tuple of ('real'/'fake', confidence)
        """
        self._ensure_loaded()

        model = self._model_data['model']
        scaler = self._model_data.get('scaler')
        model_features = self._model_data['features']

        # Determine which feature order to use based on scaler dimensions
        if scaler is not None:
            if scaler.n_features_in_ == 30:
                feature_order = FEATURES_30
            elif scaler.n_features_in_ == 29:
                feature_order = FEATURES_29
            elif scaler.n_features_in_ == 27:
                feature_order = FEATURES_27
            else:
                # Unknown scaler size - skip scaling
                feature_order = FEATURES_29
                scaler = None
        else:
            feature_order = FEATURES_29

        # Build feature vector in correct order
        # Use median defaults for features that may be NaN
        def get_feature_value(f):
            val = features.get(f)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return FEATURE_MEDIANS.get(f, 0.0)
            return val
        all_feature_values = [get_feature_value(f) for f in feature_order]
        X_all = pd.DataFrame([all_feature_values], columns=feature_order)

        # Scale if scaler exists
        if scaler is not None:
            X_scaled = pd.DataFrame(
                scaler.transform(X_all),
                columns=feature_order
            )
        else:
            X_scaled = X_all

        # Select only the features this model needs
        X_model = X_scaled[model_features]

        # Get prediction and probability
        proba = model.predict_proba(X_model)[0]
        pred = int(model.predict(X_model)[0])

        # XGBoost: class 0 = real, class 1 = fake
        label = 'fake' if pred == 1 else 'real'
        confidence = float(proba[1] if pred == 1 else proba[0])

        return label, confidence

    def get_required_features(self) -> List[str]:
        """Get required features for this model (all 29 for proper scaling)."""
        return ALL_FEATURES_ORDER.copy()

    def warmup(self) -> None:
        """Pre-load model."""
        self._ensure_loaded()


class SpeechEnsembleClassifier(BaseClassifier):
    """5-expert weighted ensemble classifier for speech."""

    def __init__(
        self,
        experts: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        threshold: float = 0.30,
    ):
        """Initialize ensemble classifier.

        Args:
            experts: List of expert model names
            weights: List of weights for each expert (must sum to 1)
            threshold: Decision threshold for fake classification
        """
        self.experts = experts or SPEECH_ENSEMBLE_CONFIG['experts']
        self.weights = weights or SPEECH_ENSEMBLE_CONFIG['weights']
        self.threshold = threshold

        if len(self.experts) != len(self.weights):
            raise ValueError("Number of experts must match number of weights")

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        self._classifiers: List[SpeechClassifier] = []

    def _ensure_loaded(self) -> None:
        """Lazy load all expert classifiers."""
        if not self._classifiers:
            self._classifiers = [
                SpeechClassifier(name) for name in self.experts
            ]
            for clf in self._classifiers:
                clf._ensure_loaded()

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Make weighted ensemble prediction.

        Args:
            features: Dictionary of computed features

        Returns:
            Tuple of ('real'/'fake', confidence)
        """
        self._ensure_loaded()

        # Get predictions from all experts
        weighted_prob = 0.0

        for clf, weight in zip(self._classifiers, self.weights):
            label, conf = clf.predict(features)
            # Convert to fake probability
            fake_prob = conf if label == 'fake' else (1 - conf)
            weighted_prob += weight * fake_prob

        # Apply threshold
        label = 'fake' if weighted_prob >= self.threshold else 'real'
        confidence = weighted_prob if label == 'fake' else (1 - weighted_prob)

        return label, confidence

    def get_required_features(self) -> List[str]:
        """Get required features (all 29 for proper scaling)."""
        return ALL_FEATURES_ORDER.copy()

    def warmup(self) -> None:
        """Pre-load all expert models."""
        self._ensure_loaded()

    def predict_all_experts(self, features: Dict[str, float]) -> Dict[str, Tuple[str, float]]:
        """Get predictions from all experts (for debugging).

        Args:
            features: Dictionary of computed features

        Returns:
            Dictionary mapping expert name to (label, confidence)
        """
        self._ensure_loaded()

        results = {}
        for clf in self._classifiers:
            label, conf = clf.predict(features)
            results[clf.model_name] = (label, conf)
        return results


class MusicAdaptiveClassifier(BaseClassifier):
    """Domain-adaptive music classifier using percentile threshold.

    This classifier uses a simple threshold on mean similarity,
    which is robust to domain shift.
    """

    def __init__(self, threshold: float = 0.8004):
        """Initialize adaptive classifier.

        Args:
            threshold: Threshold on mean similarity. Above = real, below = fake.
        """
        self.threshold = threshold
        self.feature = 'mean'

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Make prediction based on mean similarity threshold.

        Args:
            features: Dictionary with at least 'mean' feature

        Returns:
            Tuple of ('real'/'fake', confidence)
        """
        mean_sim = features.get(self.feature, 0.5)

        # Above threshold = real, below = fake
        if mean_sim >= self.threshold:
            label = 'real'
            # Confidence based on distance from threshold
            confidence = min(1.0, 0.5 + (mean_sim - self.threshold) * 2)
        else:
            label = 'fake'
            confidence = min(1.0, 0.5 + (self.threshold - mean_sim) * 2)

        return label, confidence

    def get_required_features(self) -> List[str]:
        """Get required features."""
        return [self.feature]

    def warmup(self) -> None:
        """No warmup needed for threshold classifier."""
        pass


class MusicXGBoostClassifier(BaseClassifier):
    """XGBoost music classifier (for in-distribution use)."""

    def __init__(self, model_name: str = 'mi_msclap'):
        """Initialize music XGBoost classifier.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self._model_data = None

    def _ensure_loaded(self) -> None:
        """Lazy load model from registry."""
        if self._model_data is None:
            registry = get_registry()
            self._model_data = registry.get_classifier(self.model_name)

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Make prediction using XGBoost model.

        Args:
            features: Dictionary of computed features (must include all 29 features)

        Returns:
            Tuple of ('real'/'fake', confidence)
        """
        self._ensure_loaded()

        model = self._model_data['model']
        scaler = self._model_data.get('scaler')
        model_features = self._model_data['features']

        # Build full feature vector in correct order for scaler
        all_feature_values = [features.get(f, 0.0) for f in ALL_FEATURES_ORDER]
        X_all = pd.DataFrame([all_feature_values], columns=ALL_FEATURES_ORDER)

        # Scale if scaler exists AND dimensions match
        if scaler is not None and scaler.n_features_in_ == len(ALL_FEATURES_ORDER):
            X_scaled = pd.DataFrame(
                scaler.transform(X_all),
                columns=ALL_FEATURES_ORDER
            )
        else:
            X_scaled = X_all

        # Select only the features this model needs
        X_model = X_scaled[model_features]

        # Get prediction and probability
        proba = model.predict_proba(X_model)[0]
        pred = int(model.predict(X_model)[0])

        label = 'fake' if pred == 1 else 'real'
        confidence = float(proba[1] if pred == 1 else proba[0])

        return label, confidence

    def get_required_features(self) -> List[str]:
        """Get required features (all 29 for proper scaling)."""
        return ALL_FEATURES_ORDER.copy()

    def warmup(self) -> None:
        """Pre-load model."""
        self._ensure_loaded()


def get_classifier(model: str) -> BaseClassifier:
    """Factory function to get classifier by model name.

    Args:
        model: Model name or type:
            - 'speech' or 'ds_msclap': Best single speech model
            - 'speech_ensemble': 5-expert ensemble
            - 'music' or 'mi_adaptive': Domain-robust music classifier
            - 'music_xgb' or 'mi_msclap': In-distribution music classifier
            - Other speech model names: sv_msclap, sv_ds_msclap, etc.

    Returns:
        Appropriate classifier instance
    """
    model_lower = model.lower()

    if model_lower in ('speech', 'ds_msclap'):
        return SpeechClassifier('ds_msclap')
    elif model_lower == 'speech_ensemble':
        return SpeechEnsembleClassifier()
    elif model_lower in ('music', 'mi_adaptive'):
        return MusicAdaptiveClassifier()
    elif model_lower in ('music_xgb', 'mi_msclap'):
        return MusicXGBoostClassifier('mi_msclap')
    elif model_lower in FEATURE_SETS:
        # Specific speech model
        return SpeechClassifier(model_lower)
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Valid options: speech, speech_ensemble, music, music_xgb, "
            f"or specific model names: {list(FEATURE_SETS.keys())}"
        )
