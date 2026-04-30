"""Thread-safe singleton model registry for lazy loading and caching.

Provides centralized access to:
- MS-CLAP embedding model (loaded once, reused)
- Trained classifiers (XGBoost models with scalers)
"""

import pickle
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch

# Model directory (relative to project root)
MODELS_DIR = Path(__file__).parent.parent / "models" / "trained"

# Feature sets for each classifier
FEATURE_SETS: Dict[str, List[str]] = {
    'ds_msclap': ['q25', 'entropy', 'trimmed_mean', 'mean', 'q10', 'q50', 'coefficient_of_variation', 'std'],
    'sv_msclap': ['q95', 'trimmed_mean', 'entropy', 'median_mean_diff', 'q90', 'q75', 'q50', 'iqr_range_ratio'],
    'sv_ds_msclap': ['trimmed_mean', 'bimodality_coefficient', 'skewness', 'q5', 'mean', 'q50', 'q75', 'q90'],
    'mlaad_msclap': ['mean', 'std', 'variance', 'min', 'max', 'peak_to_peak', 'skewness', 'kurtosis',
                     'bimodality_coefficient', 'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr',
                     'tail_weight_ratio', 'trimmed_mean', 'entropy', 'gini_coefficient',
                     'coefficient_of_variation', 'variance_mean_ratio', 'kurtosis_variance_ratio',
                     'skewness_kurtosis_ratio', 'iqr_range_ratio', 'median_mean_diff', 'n_samples',
                     'shapiro_p', 'normaltest_p'],
    'audeter_msclap': ['shapiro_p', 'max', 'q10', 'entropy', 'normaltest_p', 'bimodality_coefficient', 'iqr', 'peak_to_peak'],
    'favc_msclap': ['bimodality_coefficient', 'q90', 'mean', 'kurtosis_variance_ratio', 'entropy', 'min', 'peak_to_peak', 'q5'],
    'spoofceleb_msclap': ['q5', 'entropy', 'max', 'q10', 'min', 'mean', 'q50', 'trimmed_mean'],
    'mi_msclap': ['entropy', 'shapiro_p', 'peak_to_peak', 'min', 'q5', 'variance_mean_ratio', 'q10', 'std'],
    'mi_adaptive': ['mean'],
}

# 5-Expert ensemble configuration
SPEECH_ENSEMBLE_CONFIG = {
    'experts': ['ds_msclap', 'audeter_msclap', 'sv_ds_msclap', 'sv_msclap', 'mlaad_msclap'],
    'weights': [0.30, 0.30, 0.20, 0.10, 0.10],
    'threshold': 0.30,
}


class ModelRegistry:
    """Thread-safe singleton registry for model caching."""

    _instance: Optional['ModelRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'ModelRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._embedding_model = None
        self._embedding_device = None
        self._classifiers: Dict[str, Dict[str, Any]] = {}
        self._model_lock = threading.Lock()
        self._initialized = True

    def get_embedding_model(self) -> Tuple[Any, torch.device]:
        """Get the MS-CLAP embedding model (lazy loaded).

        Returns:
            Tuple of (model, device)
        """
        if self._embedding_model is None:
            with self._model_lock:
                if self._embedding_model is None:
                    self._embedding_model, self._embedding_device = self._load_embedding_model()
        return self._embedding_model, self._embedding_device

    def _load_embedding_model(self) -> Tuple[Any, torch.device]:
        """Load MS-CLAP model with MPS acceleration if available."""
        from msclap import CLAP

        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Load model
        model = CLAP(version='2023', use_cuda=False)

        # Move to MPS if available and patch the embeddings method
        if device.type == 'mps':
            model.clap = model.clap.to(device)

            def patched_get_audio_embeddings(preprocessed_audio):
                with torch.no_grad():
                    preprocessed_audio = preprocessed_audio.reshape(
                        preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                    preprocessed_audio = preprocessed_audio.to(device)
                    return model.clap.audio_encoder(preprocessed_audio)[0]

            model._get_audio_embeddings = patched_get_audio_embeddings

        return model, device

    def get_classifier(self, name: str) -> Dict[str, Any]:
        """Get a trained classifier by name (lazy loaded).

        Args:
            name: Classifier name (e.g., 'ds_msclap', 'mi_adaptive')

        Returns:
            Dict with 'model', 'scaler' (if applicable), 'features', etc.

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if name not in self._classifiers:
            with self._model_lock:
                if name not in self._classifiers:
                    self._classifiers[name] = self._load_classifier(name)
        return self._classifiers[name]

    def _load_classifier(self, name: str) -> Dict[str, Any]:
        """Load a classifier from disk."""
        model_path = MODELS_DIR / f"{name}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Available models: {list(MODELS_DIR.glob('*.pkl'))}"
            )

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        # Add feature list if not present (for mi_adaptive)
        if 'features' not in data and name in FEATURE_SETS:
            data['features'] = FEATURE_SETS[name]

        data['name'] = name
        return data

    def get_ensemble_classifiers(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get multiple classifiers for ensemble.

        Args:
            names: List of classifier names. If None, uses default speech ensemble.

        Returns:
            List of classifier dicts
        """
        if names is None:
            names = SPEECH_ENSEMBLE_CONFIG['experts']

        return [self.get_classifier(name) for name in names]

    def get_features_for_model(self, name: str) -> List[str]:
        """Get the feature list for a model without loading the full model.

        Args:
            name: Model name

        Returns:
            List of feature names
        """
        return FEATURE_SETS.get(name, [])

    def get_union_features(self, names: List[str]) -> List[str]:
        """Get union of features needed for multiple models.

        Args:
            names: List of model names

        Returns:
            Deduplicated list of all required features
        """
        all_features = set()
        for name in names:
            all_features.update(self.get_features_for_model(name))
        return list(all_features)

    def warmup(self) -> None:
        """Pre-load all models to eliminate cold start latency."""
        # Load embedding model
        self.get_embedding_model()

        # Load common classifiers
        for name in ['ds_msclap', 'mi_adaptive']:
            try:
                self.get_classifier(name)
            except FileNotFoundError:
                pass

    def clear_cache(self) -> None:
        """Clear all cached models (for testing/memory management)."""
        with self._model_lock:
            self._embedding_model = None
            self._embedding_device = None
            self._classifiers.clear()


# Module-level convenience function
def get_registry() -> ModelRegistry:
    """Get the singleton model registry."""
    return ModelRegistry()
