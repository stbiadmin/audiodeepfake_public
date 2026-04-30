"""Pairwise similarity and statistical feature computation.

Computes cosine similarities between embeddings and extracts
statistical features from the similarity distribution.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats
from typing import Dict, List, Optional, Set


# All available features
ALL_FEATURES = [
    # Basic statistics
    'mean', 'std', 'variance', 'min', 'max', 'peak_to_peak',
    # Distribution shape
    'skewness', 'kurtosis', 'bimodality_coefficient',
    # Percentiles
    'q5', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95', 'iqr',
    # Tail & outliers
    'tail_weight_ratio', 'trimmed_mean',
    # Information
    'entropy', 'gini_coefficient', 'coefficient_of_variation',
    # Derived ratios
    'variance_mean_ratio', 'kurtosis_variance_ratio', 'skewness_kurtosis_ratio',
    'iqr_range_ratio', 'median_mean_diff',
    # Additional
    'n_samples',
    # Normality tests (expensive)
    'shapiro_p', 'normaltest_p',
]

# Features that are expensive to compute
EXPENSIVE_FEATURES = {'shapiro_p', 'normaltest_p'}


def compute_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities between embeddings.

    Args:
        embeddings: Array of shape (N, embedding_dim)

    Returns:
        1D array of N(N-1)/2 pairwise similarity values
    """
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 embeddings to compute similarities")

    # pdist returns distances, convert to similarities
    distances = pdist(embeddings, metric='cosine')
    similarities = 1 - distances

    return similarities


def compute_features(
    similarities: np.ndarray,
    feature_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute statistical features from similarity distribution.

    Args:
        similarities: 1D array of pairwise similarities
        feature_list: List of features to compute. If None, computes all.

    Returns:
        Dictionary mapping feature names to values
    """
    if feature_list is None:
        feature_list = ALL_FEATURES

    # Convert to set for fast lookup
    needed: Set[str] = set(feature_list)
    features: Dict[str, float] = {}

    # Pre-compute commonly needed values
    n = len(similarities)
    mean_val = float(np.mean(similarities))
    std_val = float(np.std(similarities))
    var_val = float(np.var(similarities))

    # Basic statistics
    if 'mean' in needed:
        features['mean'] = mean_val
    if 'std' in needed:
        features['std'] = std_val
    if 'variance' in needed:
        features['variance'] = var_val
    if 'min' in needed:
        features['min'] = float(np.min(similarities))
    if 'max' in needed:
        features['max'] = float(np.max(similarities))
    if 'peak_to_peak' in needed:
        features['peak_to_peak'] = float(np.ptp(similarities))

    # Distribution shape
    if 'skewness' in needed or 'bimodality_coefficient' in needed or 'skewness_kurtosis_ratio' in needed:
        skew = float(stats.skew(similarities))
        if 'skewness' in needed:
            features['skewness'] = skew
    else:
        skew = None

    if 'kurtosis' in needed or 'bimodality_coefficient' in needed or 'kurtosis_variance_ratio' in needed or 'skewness_kurtosis_ratio' in needed:
        kurt = float(stats.kurtosis(similarities))
        if 'kurtosis' in needed:
            features['kurtosis'] = kurt
    else:
        kurt = None

    if 'bimodality_coefficient' in needed:
        if skew is None:
            skew = float(stats.skew(similarities))
        if kurt is None:
            kurt = float(stats.kurtosis(similarities))
        # Sarle's bimodality coefficient with sample size correction (matches training)
        n_samp = len(similarities)
        if n_samp > 3:
            excess_kurt_correction = 3 * ((n_samp - 1) ** 2 / ((n_samp - 2) * (n_samp - 3)))
            denominator = kurt + excess_kurt_correction
            features['bimodality_coefficient'] = (skew**2 + 1) / denominator if denominator != 0 else 0.0
        else:
            # Not enough samples for proper calculation
            features['bimodality_coefficient'] = float('nan')

    # Percentiles
    percentile_map = {
        'q5': 5, 'q10': 10, 'q25': 25, 'q50': 50,
        'q75': 75, 'q90': 90, 'q95': 95
    }
    percentiles_needed = [p for p in percentile_map.keys() if p in needed]

    if percentiles_needed or 'iqr' in needed or 'iqr_range_ratio' in needed:
        # Compute all at once for efficiency
        pct_values = percentile_map.values() if not percentiles_needed else [percentile_map[p] for p in percentiles_needed]
        if 'iqr' in needed or 'iqr_range_ratio' in needed:
            pct_values = list(set(list(pct_values) + [25, 75]))
        computed = np.percentile(similarities, list(pct_values))
        pct_dict = dict(zip(pct_values, computed))

        for name in percentiles_needed:
            features[name] = float(pct_dict[percentile_map[name]])

        if 'iqr' in needed or 'iqr_range_ratio' in needed:
            iqr_val = float(pct_dict[75] - pct_dict[25])
            if 'iqr' in needed:
                features['iqr'] = iqr_val

    # Tail weight ratio - ratio of samples in tails vs center (matches training)
    if 'tail_weight_ratio' in needed:
        q25 = np.percentile(similarities, 25)
        q75 = np.percentile(similarities, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        in_tails = np.sum((similarities < lower_bound) | (similarities > upper_bound))
        features['tail_weight_ratio'] = float(in_tails / len(similarities))

    # Trimmed mean (trim 10% from each end)
    if 'trimmed_mean' in needed:
        features['trimmed_mean'] = float(stats.trim_mean(similarities, 0.1))

    # Entropy (discretize to 20 bins - matches training)
    if 'entropy' in needed:
        hist, _ = np.histogram(similarities, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zeros
        features['entropy'] = float(stats.entropy(hist)) if len(hist) > 0 else 0.0

    # Gini coefficient
    if 'gini_coefficient' in needed:
        sorted_sim = np.sort(similarities)
        n_sim = len(sorted_sim)
        index = np.arange(1, n_sim + 1)
        features['gini_coefficient'] = float(
            (2 * np.sum(index * sorted_sim) / (n_sim * np.sum(sorted_sim))) - (n_sim + 1) / n_sim
        ) if np.sum(sorted_sim) != 0 else 0.0

    # Coefficient of variation
    if 'coefficient_of_variation' in needed:
        features['coefficient_of_variation'] = std_val / mean_val if mean_val != 0 else 0.0

    # Derived ratios
    if 'variance_mean_ratio' in needed:
        features['variance_mean_ratio'] = var_val / mean_val if mean_val != 0 else 0.0

    if 'kurtosis_variance_ratio' in needed:
        if kurt is None:
            kurt = float(stats.kurtosis(similarities))
        features['kurtosis_variance_ratio'] = kurt / var_val if var_val != 0 else 0.0

    if 'skewness_kurtosis_ratio' in needed:
        if skew is None:
            skew = float(stats.skew(similarities))
        if kurt is None:
            kurt = float(stats.kurtosis(similarities))
        features['skewness_kurtosis_ratio'] = skew / kurt if kurt != 0 else 0.0

    if 'iqr_range_ratio' in needed:
        if 'iqr' not in features:
            q25 = np.percentile(similarities, 25)
            q75 = np.percentile(similarities, 75)
            iqr_val = float(q75 - q25)
        else:
            iqr_val = features['iqr']
        range_val = features.get('peak_to_peak', float(np.ptp(similarities)))
        features['iqr_range_ratio'] = iqr_val / range_val if range_val != 0 else 0.0

    if 'median_mean_diff' in needed:
        median = features.get('q50', float(np.median(similarities)))
        features['median_mean_diff'] = median - mean_val

    # Sample count
    if 'n_samples' in needed:
        features['n_samples'] = float(n)

    # Normality tests (expensive - only compute if needed)
    if 'shapiro_p' in needed:
        # Shapiro-Wilk test (limited to 5000 samples)
        sample = similarities[:5000] if n > 5000 else similarities
        try:
            _, p = stats.shapiro(sample)
            features['shapiro_p'] = float(p)
        except Exception:
            features['shapiro_p'] = 0.0

    if 'normaltest_p' in needed:
        # D'Agostino-Pearson test
        try:
            _, p = stats.normaltest(similarities)
            features['normaltest_p'] = float(p)
        except Exception:
            features['normaltest_p'] = 0.0

    return features


def compute_all_features(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute similarities and all features from embeddings.

    Args:
        embeddings: Array of shape (N, embedding_dim)

    Returns:
        Dictionary of all features
    """
    similarities = compute_similarities(embeddings)
    return compute_features(similarities, ALL_FEATURES)


def compute_selected_features(
    embeddings: np.ndarray,
    feature_list: List[str],
) -> Dict[str, float]:
    """Compute similarities and selected features from embeddings.

    Args:
        embeddings: Array of shape (N, embedding_dim)
        feature_list: List of features to compute

    Returns:
        Dictionary of selected features
    """
    similarities = compute_similarities(embeddings)
    return compute_features(similarities, feature_list)


def get_similarity_stats(similarities: np.ndarray) -> Dict[str, float]:
    """Get basic statistics about similarity distribution.

    Args:
        similarities: 1D array of similarities

    Returns:
        Dictionary with mean, std, min, max, n_pairs
    """
    return {
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'n_pairs': len(similarities),
    }
