"""Statistical feature extraction from similarity distributions.

Expanded feature set for exploratory analysis - cast a wide net to find
which statistical properties distinguish real from AI-generated audio.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from config.base import FeatureConfig


@dataclass
class ExtractionResult:
    """Result of feature extraction."""
    features: Dict[str, float]
    n_similarities: int
    is_valid: bool
    message: str = ""


def _safe_entropy(x: np.ndarray) -> float:
    """Compute entropy of distribution using histogram."""
    hist, _ = np.histogram(x, bins=20, density=True)
    hist = hist[hist > 0]  # Remove zeros
    return float(stats.entropy(hist)) if len(hist) > 0 else 0.0


def _bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient. >0.555 suggests bimodality."""
    n = len(x)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)
    return float((skew**2 + 1) / (kurt + 3 * ((n-1)**2 / ((n-2)*(n-3)))))


def _peak_to_peak(x: np.ndarray) -> float:
    """Peak-to-peak range."""
    return float(np.max(x) - np.min(x))


def _coefficient_of_variation(x: np.ndarray) -> float:
    """CV = std/mean, measure of relative variability."""
    mean = np.mean(x)
    return float(np.std(x) / (mean + 1e-8)) if mean != 0 else 0.0


def _tail_weight_ratio(x: np.ndarray) -> float:
    """Ratio of samples in tails vs center."""
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    in_tails = np.sum((x < lower_bound) | (x > upper_bound))
    return float(in_tails / len(x))


def _gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient - measure of inequality/spread."""
    x_sorted = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x_sorted)
    return float((2 * np.sum((np.arange(1, n+1) * x_sorted))) / (n * np.sum(x_sorted)) - (n + 1) / n)


class DistributionFeatureExtractor:
    """Extract statistical features from similarity distributions.

    Expanded feature set for exploratory analysis:

    Basic Statistics:
    - mean, std, variance
    - min, max, range (peak_to_peak)

    Distribution Shape:
    - skewness, kurtosis
    - bimodality_coefficient

    Percentiles & Quartiles:
    - q5, q10, q25, q50, q75, q90, q95
    - iqr (interquartile range)

    Ratios & Derived:
    - variance_mean_ratio, kurtosis_variance_ratio
    - coefficient_of_variation
    - skewness_kurtosis_ratio

    Tail & Outlier Features:
    - tail_weight_ratio
    - trimmed_mean (10% trimmed)

    Information & Uniformity:
    - entropy
    - gini_coefficient

    Normality Tests (p-values):
    - shapiro_p (if n <= 5000)
    - normaltest_p
    """

    # Feature computation functions - expanded set
    FEATURE_FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {
        # Basic statistics
        'mean': lambda x: float(np.mean(x)),
        'std': lambda x: float(np.std(x)),
        'variance': lambda x: float(np.var(x)),
        'min': lambda x: float(np.min(x)),
        'max': lambda x: float(np.max(x)),
        'peak_to_peak': _peak_to_peak,

        # Distribution shape
        'skewness': lambda x: float(stats.skew(x)),
        'kurtosis': lambda x: float(stats.kurtosis(x)),
        'bimodality_coefficient': _bimodality_coefficient,

        # Percentiles
        'q5': lambda x: float(np.percentile(x, 5)),
        'q10': lambda x: float(np.percentile(x, 10)),
        'q25': lambda x: float(np.percentile(x, 25)),
        'q50': lambda x: float(np.percentile(x, 50)),
        'q75': lambda x: float(np.percentile(x, 75)),
        'q90': lambda x: float(np.percentile(x, 90)),
        'q95': lambda x: float(np.percentile(x, 95)),
        'iqr': lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),

        # Tail & outliers
        'tail_weight_ratio': _tail_weight_ratio,
        'trimmed_mean': lambda x: float(stats.trim_mean(x, 0.1)),

        # Information & uniformity
        'entropy': _safe_entropy,
        'gini_coefficient': _gini_coefficient,
        'coefficient_of_variation': _coefficient_of_variation,
    }

    # Derived features that depend on other features
    DERIVED_FEATURES = [
        'variance_mean_ratio',
        'kurtosis_variance_ratio',
        'skewness_kurtosis_ratio',
        'iqr_range_ratio',
        'median_mean_diff',
    ]

    # Normality test features (computed separately due to sample size constraints)
    NORMALITY_FEATURES = ['shapiro_p', 'normaltest_p']

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the feature extractor.

        Args:
            config: Feature configuration. Uses defaults if not provided.
        """
        self.config = config or FeatureConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate that all configured features are supported."""
        all_features = set(self.FEATURE_FUNCTIONS.keys()) | set(self.DERIVED_FEATURES)
        for feature in self.config.features:
            if feature not in all_features:
                raise ValueError(
                    f"Unknown feature: {feature}. "
                    f"Supported features: {sorted(all_features)}"
                )

    def extract_features(
        self,
        similarities: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Extract statistical features from a similarity distribution.

        Args:
            similarities: 1D array of similarity values
            prefix: Optional prefix for feature names (e.g., "all_")

        Returns:
            Dictionary mapping feature names to values
        """
        if len(similarities) < 2:
            raise ValueError("Need at least 2 similarity values for feature extraction")

        features = {}

        # Compute base features first
        for feature_name in self.config.features:
            if feature_name in self.FEATURE_FUNCTIONS:
                key = f"{prefix}{feature_name}" if prefix else feature_name
                features[key] = self.FEATURE_FUNCTIONS[feature_name](similarities)

        # Compute derived features
        mean_key = f"{prefix}mean" if prefix else "mean"
        var_key = f"{prefix}variance" if prefix else "variance"
        kurt_key = f"{prefix}kurtosis" if prefix else "kurtosis"

        if 'variance_mean_ratio' in self.config.features:
            key = f"{prefix}variance_mean_ratio" if prefix else "variance_mean_ratio"
            mean_val = features.get(mean_key, np.mean(similarities))
            var_val = features.get(var_key, np.var(similarities))
            # Avoid division by zero
            features[key] = var_val / (mean_val + 1e-8) if mean_val != 0 else 0.0

        if 'kurtosis_variance_ratio' in self.config.features:
            key = f"{prefix}kurtosis_variance_ratio" if prefix else "kurtosis_variance_ratio"
            kurt_val = features.get(kurt_key, stats.kurtosis(similarities))
            var_val = features.get(var_key, np.var(similarities))
            # Avoid division by zero
            features[key] = kurt_val / (var_val + 1e-8) if var_val != 0 else 0.0

        return features

    def extract_all_features(
        self,
        similarities: np.ndarray,
        prefix: str = "",
        include_normality_tests: bool = True,
    ) -> Dict[str, float]:
        """Extract ALL statistical features for exploratory analysis.

        Args:
            similarities: 1D array of similarity values
            prefix: Optional prefix for feature names
            include_normality_tests: Whether to run normality tests (slower)

        Returns:
            Dictionary with all features (~30 features)
        """
        if len(similarities) < 2:
            raise ValueError("Need at least 2 similarity values")

        features = {}
        n = len(similarities)

        # Compute all base features
        for name, func in self.FEATURE_FUNCTIONS.items():
            key = f"{prefix}{name}" if prefix else name
            try:
                features[key] = func(similarities)
            except Exception:
                features[key] = np.nan

        # Compute derived features
        mean_val = features.get(f"{prefix}mean" if prefix else "mean", np.mean(similarities))
        var_val = features.get(f"{prefix}variance" if prefix else "variance", np.var(similarities))
        kurt_val = features.get(f"{prefix}kurtosis" if prefix else "kurtosis", stats.kurtosis(similarities))
        skew_val = features.get(f"{prefix}skewness" if prefix else "skewness", stats.skew(similarities))
        iqr_val = features.get(f"{prefix}iqr" if prefix else "iqr", np.percentile(similarities, 75) - np.percentile(similarities, 25))
        range_val = features.get(f"{prefix}peak_to_peak" if prefix else "peak_to_peak", np.ptp(similarities))
        median_val = features.get(f"{prefix}q50" if prefix else "q50", np.median(similarities))

        # Derived ratios
        vmr_key = f"{prefix}variance_mean_ratio" if prefix else "variance_mean_ratio"
        kvr_key = f"{prefix}kurtosis_variance_ratio" if prefix else "kurtosis_variance_ratio"
        skr_key = f"{prefix}skewness_kurtosis_ratio" if prefix else "skewness_kurtosis_ratio"
        irr_key = f"{prefix}iqr_range_ratio" if prefix else "iqr_range_ratio"
        mmd_key = f"{prefix}median_mean_diff" if prefix else "median_mean_diff"

        features[vmr_key] = var_val / (mean_val + 1e-8) if abs(mean_val) > 1e-8 else 0.0
        features[kvr_key] = kurt_val / (var_val + 1e-8) if abs(var_val) > 1e-8 else 0.0
        features[skr_key] = skew_val / (kurt_val + 1e-8) if abs(kurt_val) > 1e-8 else 0.0
        features[irr_key] = iqr_val / (range_val + 1e-8) if abs(range_val) > 1e-8 else 0.0
        features[mmd_key] = median_val - mean_val

        # Normality tests (useful for detecting synthetic distributions)
        if include_normality_tests and n >= 8:
            # Shapiro-Wilk (only works for n <= 5000)
            shapiro_key = f"{prefix}shapiro_p" if prefix else "shapiro_p"
            if n <= 5000:
                try:
                    _, p = stats.shapiro(similarities)
                    features[shapiro_key] = float(p)
                except Exception:
                    features[shapiro_key] = np.nan
            else:
                features[shapiro_key] = np.nan

            # D'Agostino and Pearson's test (needs n >= 8)
            normaltest_key = f"{prefix}normaltest_p" if prefix else "normaltest_p"
            try:
                _, p = stats.normaltest(similarities)
                features[normaltest_key] = float(p)
            except Exception:
                features[normaltest_key] = np.nan

        # Sample count (useful metadata)
        n_key = f"{prefix}n_samples" if prefix else "n_samples"
        features[n_key] = float(n)

        return features

    def extract_with_validation(
        self,
        similarities: np.ndarray,
        min_samples: int = 10,
        prefix: str = "",
    ) -> ExtractionResult:
        """Extract features with validation and error handling.

        Args:
            similarities: 1D array of similarity values
            min_samples: Minimum required samples for valid extraction
            prefix: Optional prefix for feature names

        Returns:
            ExtractionResult with features and validation info
        """
        n_similarities = len(similarities)

        if n_similarities < min_samples:
            return ExtractionResult(
                features={},
                n_similarities=n_similarities,
                is_valid=False,
                message=f"Too few samples: {n_similarities} < {min_samples}",
            )

        try:
            features = self.extract_features(similarities, prefix)
            return ExtractionResult(
                features=features,
                n_similarities=n_similarities,
                is_valid=True,
                message="Success",
            )
        except Exception as e:
            return ExtractionResult(
                features={},
                n_similarities=n_similarities,
                is_valid=False,
                message=f"Extraction failed: {e}",
            )

    def extract_for_multiple_speakers(
        self,
        speaker_similarities: Dict[str, np.ndarray],
        aggregation: str = "weighted_mean",
    ) -> Dict[str, float]:
        """Extract and aggregate features for multiple speakers.

        Args:
            speaker_similarities: Dict mapping speaker_id -> similarities
            aggregation: How to aggregate ("weighted_mean", "mean", "concat")

        Returns:
            Aggregated feature dictionary
        """
        if not speaker_similarities:
            raise ValueError("No speaker similarities provided")

        speaker_features = {}
        speaker_weights = {}

        # Extract features for each speaker
        for speaker_id, similarities in speaker_similarities.items():
            if len(similarities) >= 2:
                speaker_features[speaker_id] = self.extract_all_features(similarities)
                speaker_weights[speaker_id] = len(similarities)

        if not speaker_features:
            raise ValueError("No speakers had enough similarities for feature extraction")

        if aggregation == "concat":
            # Concatenate all features with speaker prefix
            result = {}
            for speaker_id, features in speaker_features.items():
                for key, value in features.items():
                    result[f"{speaker_id}_{key}"] = value
            return result

        # Aggregate across speakers
        feature_names = list(next(iter(speaker_features.values())).keys())
        aggregated = {}

        for feature_name in feature_names:
            values = []
            weights = []

            for speaker_id, features in speaker_features.items():
                values.append(features[feature_name])
                weights.append(speaker_weights[speaker_id])

            if aggregation == "weighted_mean":
                total_weight = sum(weights)
                aggregated[feature_name] = sum(
                    v * w for v, w in zip(values, weights)
                ) / total_weight
            else:  # mean
                aggregated[feature_name] = np.mean(values)

        # Add speaker-level statistics
        aggregated['num_speakers'] = len(speaker_features)
        if len(speaker_features) > 1:
            mean_values = [f['mean'] for f in speaker_features.values()]
            aggregated['speaker_mean_variance'] = np.var(mean_values)

        return aggregated

    def get_feature_names(self, prefix: str = "") -> List[str]:
        """Get list of feature names that will be extracted.

        Args:
            prefix: Optional prefix for feature names

        Returns:
            List of feature names
        """
        if prefix:
            return [f"{prefix}{f}" for f in self.config.features]
        return list(self.config.features)

    @staticmethod
    def get_all_feature_names(prefix: str = "") -> List[str]:
        """Get all possible feature names from expanded feature set.

        Args:
            prefix: Optional prefix for feature names

        Returns:
            List of all ~30 feature names
        """
        all_features = [
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
            # Metadata
            'n_samples',
        ]
        if prefix:
            return [f"{prefix}{f}" for f in all_features]
        return all_features
