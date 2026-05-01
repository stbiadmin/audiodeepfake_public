"""Ensemble classifiers for audio deepfake detection.

Combines multiple trained models for improved generalization.

Based on research showing ensemble methods improve OOD performance:
- Averaging predictions reduces variance
- Different models capture complementary patterns
- Weighted ensembles can emphasize better-generalizing models
"""

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loader import ALL_FEATURES


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble."""
    name: str
    model_names: List[str]
    weights: Optional[List[float]] = None  # None = equal weights

    def __post_init__(self):
        if self.weights is None:
            self.weights = [1.0 / len(self.model_names)] * len(self.model_names)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]


# Pre-defined ensemble configurations
ENSEMBLE_CONFIGS = {
    'speech_ensemble': EnsembleConfig(
        name='speech_ensemble',
        model_names=['sv_msclap_model', 'ds_msclap_model'],
        weights=[0.5, 0.5],
    ),
    'full_ensemble': EnsembleConfig(
        name='full_ensemble',
        model_names=['sv_msclap_model', 'ds_msclap_model', 'universal_msclap_model'],
        weights=[0.4, 0.35, 0.25],
    ),
    'weighted_by_itw': EnsembleConfig(
        name='weighted_by_itw',
        model_names=['sv_msclap_model', 'ds_msclap_model', 'universal_msclap_model'],
        # Weight by In-the-Wild F1 performance (ds_msclap best)
        weights=[0.25, 0.50, 0.25],
    ),
}


class EnsembleClassifier:
    """Ensemble of multiple trained classifiers.

    Combines predictions from multiple models using weighted averaging.
    Handles models with different features and scalers.
    """

    def __init__(
        self,
        models_dir: Path,
        config: EnsembleConfig,
        verbose: bool = True,
    ):
        """Initialize ensemble from saved models.

        Args:
            models_dir: Directory containing .pkl model files
            config: Ensemble configuration
            verbose: Print loading progress
        """
        self.models_dir = Path(models_dir)
        self.config = config
        self.verbose = verbose

        self.models_: List[Dict[str, Any]] = []
        self._load_models()

    def _load_models(self) -> None:
        """Load all models in the ensemble."""
        for i, model_name in enumerate(self.config.model_names):
            model_path = self.models_dir / f"{model_name}.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            model_info = {
                'name': model_name,
                'model': data['model'],
                'features': data['features'],
                'scaler': data.get('scaler'),
                'weight': self.config.weights[i],
            }

            self.models_.append(model_info)

            if self.verbose:
                print(f"Loaded {model_name}: {len(data['features'])} features, "
                      f"weight={self.config.weights[i]:.2f}, "
                      f"scaler={'yes' if data.get('scaler') else 'no'}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get weighted average prediction probabilities.

        Args:
            X: Feature DataFrame with ALL_FEATURES columns

        Returns:
            Array of probabilities for fake class
        """
        weighted_probas = np.zeros(len(X))

        for model_info in self.models_:
            # Apply scaler if present (to all features first)
            if model_info['scaler'] is not None:
                X_scaled = model_info['scaler'].transform(X[ALL_FEATURES].values)
                X_scaled_df = pd.DataFrame(X_scaled, columns=ALL_FEATURES, index=X.index)
                X_model = X_scaled_df[model_info['features']]
            else:
                X_model = X[model_info['features']]

            # Get predictions
            probas = model_info['model'].predict_proba(X_model)[:, 1]

            # Add weighted contribution
            weighted_probas += model_info['weight'] * probas

        return weighted_probas

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions.

        Args:
            X: Feature DataFrame
            threshold: Classification threshold

        Returns:
            Binary predictions (0=real, 1=fake)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Evaluate ensemble on a dataset.

        Args:
            X: Feature DataFrame
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, y_proba)),
        }

        # Per-class metrics
        metrics['f1_real'] = float(f1_score(y, y_pred, pos_label=0, zero_division=0))
        metrics['f1_fake'] = float(f1_score(y, y_pred, pos_label=1, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

        return metrics


def create_ensemble(
    models_dir: Path,
    ensemble_name: str,
    verbose: bool = True,
) -> EnsembleClassifier:
    """Create an ensemble from a pre-defined configuration.

    Args:
        models_dir: Directory containing model files
        ensemble_name: Name of ensemble config (from ENSEMBLE_CONFIGS)
        verbose: Print progress

    Returns:
        Configured EnsembleClassifier
    """
    if ensemble_name not in ENSEMBLE_CONFIGS:
        available = list(ENSEMBLE_CONFIGS.keys())
        raise ValueError(f"Unknown ensemble: {ensemble_name}. Available: {available}")

    config = ENSEMBLE_CONFIGS[ensemble_name]
    return EnsembleClassifier(models_dir, config, verbose=verbose)


def list_ensembles() -> List[str]:
    """List available ensemble configurations."""
    return list(ENSEMBLE_CONFIGS.keys())
