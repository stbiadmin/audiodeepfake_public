"""Data loading utilities for classification training.

Loads combined JSON feature files and prepares feature matrices for training.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.audio_types import COMBINED_CONFIGS, get_combined_audio_types


# Audio types available for training
AUDIO_TYPES = [
    'single_voice',
    'music_instrumental',
    'music_with_vocals',
    'deepspeak_v2_train',
    'mlaad_english',
    'audeter',  # AUDETER in-the-wild bona-fide + modern TTS fakes
    'fakeavceleb',  # FakeAVCeleb audio deepfakes (SV2TTS voice cloning)
    'spoofceleb',  # SpoofCeleb GradTTS+BigVGAN fakes + FakeAVCeleb real (VoxCeleb domain)
]

# Embedding models
EMBEDDING_MODELS = ['laion_clap', 'msclap', 'wavlm', 'wav2vec2']

# Base path for feature data
DEFAULT_DATA_DIR = Path('data/features')

# Features to exclude (dataset artifacts or unreliable for short clips)
# - n_samples: dataset artifact, not discriminative
# - shapiro_p, normaltest_p: require n>=10-20 samples, unreliable for short clips
EXCLUDED_FEATURES = ['n_samples', 'shapiro_p', 'normaltest_p']

# All 30 statistical features (excluding n_samples)
ALL_FEATURES = [
    # Basic statistics
    'mean', 'std', 'variance', 'min', 'max', 'peak_to_peak',
    # Distribution shape
    'skewness', 'kurtosis', 'bimodality_coefficient',
    # Percentiles
    'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr',
    # Tail & outliers
    'tail_weight_ratio', 'trimmed_mean',
    # Information & uniformity
    'entropy', 'gini_coefficient', 'coefficient_of_variation',
    # Derived ratios
    'variance_mean_ratio', 'kurtosis_variance_ratio',
    'skewness_kurtosis_ratio', 'iqr_range_ratio', 'median_mean_diff',
    # Normality tests
    'shapiro_p', 'normaltest_p',
]


def get_feature_names(exclude_artifacts: bool = True) -> List[str]:
    """Get list of feature names for training.

    Args:
        exclude_artifacts: If True, exclude n_samples and other artifacts

    Returns:
        List of feature names
    """
    if exclude_artifacts:
        return [f for f in ALL_FEATURES if f not in EXCLUDED_FEATURES]
    return ALL_FEATURES.copy()


def load_json(path: Union[str, Path]) -> List[Dict]:
    """Load JSON file containing feature data.

    Args:
        path: Path to JSON file

    Returns:
        List of sample dictionaries
    """
    with open(path) as f:
        return json.load(f)


def load_combined_features(
    audio_type: str,
    embedding_model: str,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
) -> List[Dict]:
    """Load combined features for a single audio type.

    Args:
        audio_type: One of AUDIO_TYPES
        embedding_model: One of EMBEDDING_MODELS
        data_dir: Base directory for feature data

    Returns:
        List of sample dictionaries with features and labels
    """
    data_dir = Path(data_dir)
    path = data_dir / embedding_model / 'combined' / f'{audio_type}_combined.json'

    if not path.exists():
        raise FileNotFoundError(f"Combined features not found: {path}")

    return load_json(path)


def load_all_audio_types(
    embedding_model: str,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    audio_types: Optional[List[str]] = None,
) -> List[Dict]:
    """Load combined features for all audio types.

    Args:
        embedding_model: One of EMBEDDING_MODELS
        data_dir: Base directory for feature data
        audio_types: List of audio types to load (default: all)

    Returns:
        List of sample dictionaries with audio_type field preserved
    """
    if audio_types is None:
        audio_types = AUDIO_TYPES

    all_data = []
    for audio_type in audio_types:
        try:
            data = load_combined_features(audio_type, embedding_model, data_dir)
            # Ensure audio_type is set for each sample
            for item in data:
                item['audio_type'] = audio_type
            all_data.extend(data)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    return all_data


def prepare_feature_matrix(
    data: List[Dict],
    features: Optional[List[str]] = None,
    handle_nan: str = 'fill_median',
    normalize: bool = False,
    scaler: Optional[RobustScaler] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[RobustScaler]]:
    """Extract feature matrix and labels from loaded data.

    Args:
        data: List of sample dictionaries
        features: List of feature names to extract (default: all)
        handle_nan: How to handle NaN values ('fill_median', 'fill_zero', 'drop')
        normalize: Whether to apply RobustScaler normalization
        scaler: Pre-fitted scaler to use (for inference). If None and normalize=True,
                a new scaler will be fitted.

    Returns:
        Tuple of (X DataFrame, y array, audio_type_indices array or None, scaler or None)
    """
    if features is None:
        features = get_feature_names()

    # Extract features
    rows = []
    labels = []
    audio_types = []

    for item in data:
        feat_dict = item.get('features', {})
        row = {f: feat_dict.get(f, np.nan) for f in features}
        rows.append(row)
        labels.append(item.get('label', 0))
        audio_types.append(item.get('audio_type', 'unknown'))

    X = pd.DataFrame(rows)
    y = np.array(labels)

    # Handle NaN values
    if handle_nan == 'fill_median':
        X = X.fillna(X.median())
    elif handle_nan == 'fill_zero':
        X = X.fillna(0)
    elif handle_nan == 'drop':
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        audio_types = [at for at, v in zip(audio_types, valid_mask) if v]

    # Apply normalization if requested
    fitted_scaler = None
    if normalize:
        if scaler is not None:
            # Use pre-fitted scaler (inference mode)
            X_values = scaler.transform(X.values)
            fitted_scaler = scaler
        else:
            # Fit new scaler (training mode)
            fitted_scaler = RobustScaler()
            X_values = fitted_scaler.fit_transform(X.values)
        X = pd.DataFrame(X_values, columns=X.columns, index=X.index)

    # Create audio type indices for stratification
    audio_type_indices = None
    if len(set(audio_types)) > 1:
        # Create combined stratification key: audio_type + label
        strat_keys = [f"{at}_{label}" for at, label in zip(audio_types, y)]
        unique_keys = sorted(set(strat_keys))
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        audio_type_indices = np.array([key_to_idx[k] for k in strat_keys])

    return X, y, audio_type_indices, fitted_scaler


def balance_classes(
    X: pd.DataFrame,
    y: np.ndarray,
    strategy: str = 'undersample',
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Balance classes by undersampling majority class.

    Args:
        X: Feature DataFrame
        y: Label array
        strategy: Balancing strategy ('undersample' only for now)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of balanced (X, y)
    """
    if strategy != 'undersample':
        raise ValueError(f"Unknown strategy: {strategy}")

    # Check current balance
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # If already balanced (within 10%), return as-is
    if len(unique) == 2:
        ratio = min(counts) / max(counts)
        if ratio > 0.9:
            return X, y

    # Undersample majority class
    rus = RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = rus.fit_resample(X, y)

    return pd.DataFrame(X_balanced, columns=X.columns), y_balanced


def get_dataset_stats(
    data: List[Dict],
) -> Dict[str, Dict[str, int]]:
    """Get sample counts per audio type and class.

    Args:
        data: List of sample dictionaries

    Returns:
        Dictionary mapping audio_type -> {real: count, fake: count}
    """
    stats = {}

    for item in data:
        audio_type = item.get('audio_type', 'unknown')
        label = item.get('label', 0)

        if audio_type not in stats:
            stats[audio_type] = {'real': 0, 'fake': 0}

        if label == 0:
            stats[audio_type]['real'] += 1
        else:
            stats[audio_type]['fake'] += 1

    return stats


def load_and_prepare(
    audio_type: str,
    embedding_model: str,
    features: Optional[List[str]] = None,
    balance: bool = True,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    random_state: int = 42,
    normalize: bool = False,
    scaler: Optional[RobustScaler] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[RobustScaler]]:
    """Convenience function to load and prepare data for training.

    Args:
        audio_type: Audio type to load. Can be:
            - A single audio type (e.g., 'single_voice')
            - 'universal' for all audio types
            - A combined config name from COMBINED_CONFIGS (e.g., 'sv_ds')
        embedding_model: Embedding model name
        features: Feature names to use
        balance: Whether to balance classes
        data_dir: Base data directory
        random_state: Random seed
        normalize: Whether to apply RobustScaler normalization
        scaler: Pre-fitted scaler to use (for inference)

    Returns:
        Tuple of (X, y, scaler) ready for training. Scaler is None if normalize=False.
    """
    # Load data
    if audio_type == 'universal':
        data = load_all_audio_types(embedding_model, data_dir)
    elif audio_type in COMBINED_CONFIGS:
        # Load from combined configuration (multiple audio types)
        audio_types_to_combine = get_combined_audio_types(audio_type)
        data = load_all_audio_types(embedding_model, data_dir, audio_types=audio_types_to_combine)
    else:
        data = load_combined_features(audio_type, embedding_model, data_dir)

    # Prepare feature matrix
    X, y, _, fitted_scaler = prepare_feature_matrix(
        data, features, normalize=normalize, scaler=scaler
    )

    # Balance if requested
    if balance:
        X, y = balance_classes(X, y, random_state=random_state)

    return X, y, fitted_scaler


def create_stratification_groups(
    data: List[Dict],
) -> np.ndarray:
    """Create stratification groups for train/test split.

    For universal models, stratify by audio_type + label to ensure
    each combination is represented in both train and test sets.

    Args:
        data: List of sample dictionaries

    Returns:
        Array of stratification group indices
    """
    strat_keys = []
    for item in data:
        audio_type = item.get('audio_type', 'unknown')
        label = item.get('label', 0)
        strat_keys.append(f"{audio_type}_{label}")

    unique_keys = sorted(set(strat_keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    return np.array([key_to_idx[k] for k in strat_keys])
