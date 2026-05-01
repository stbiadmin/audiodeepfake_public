"""Base configuration dataclasses for audio deepfake detection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 48000
    segment_duration: float = 2.0  # seconds
    segment_hop: float = 1.0  # seconds (overlap = segment_duration - segment_hop)
    min_segments: int = 3  # minimum segments required per file (relaxed from 5)
    normalize: bool = True

    @property
    def segment_samples(self) -> int:
        """Number of samples per segment."""
        return int(self.segment_duration * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        """Number of samples to hop between segments."""
        return int(self.segment_hop * self.sample_rate)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""
    model_type: str = "laion_clap"  # "laion_clap" or "msclap"
    msclap_version: str = "2023"  # "2022", "2023", or "clapcap" (for msclap only)
    use_cuda: bool = False
    batch_size: int = 16
    embedding_dim: int = 512


@dataclass
class FeatureConfig:
    """Configuration for statistical feature extraction."""
    features: List[str] = field(default_factory=lambda: [
        'mean',
        'variance',
        'skewness',
        'kurtosis',
        'q25',
        'q50',
        'q75',
        'variance_mean_ratio',
        'kurtosis_variance_ratio',
    ])

    def get_feature_names(self, prefix: str = "") -> List[str]:
        """Get feature names with optional prefix."""
        if prefix:
            return [f"{prefix}_{f}" for f in self.features]
        return self.features.copy()


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    audio_type: str = "single_voice"  # single_voice, multi_voice, music_instrumental, etc.
    durations: List[int] = field(default_factory=lambda: [5, 15, 30, 60])
    segment_lengths: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    cv_folds: int = 5
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # Paths
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    model_dir: Optional[Path] = None


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost classifier."""
    tree_method: str = 'hist'
    objective: str = 'binary:logistic'
    eval_metric: str = 'logloss'
    learning_rate: float = 0.1
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    num_boost_round: int = 100
    early_stopping_rounds: int = 10

    def to_dict(self) -> dict:
        """Convert to XGBoost parameter dictionary."""
        return {
            'tree_method': self.tree_method,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
        }


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)

    @classmethod
    def from_audio_type(cls, audio_type: str) -> 'PipelineConfig':
        """Create pipeline config for a specific audio type."""
        from .audio_types import get_audio_type_config

        type_config = get_audio_type_config(audio_type)

        # Get segment settings from audio type config (with defaults)
        segment_duration = type_config.get('segment_duration', 2.0)
        segment_hop = type_config.get('segment_hop', segment_duration / 2)  # Default 50% overlap
        min_segments = type_config.get('min_segments', 3)

        audio_config = AudioConfig(
            segment_duration=segment_duration,
            segment_hop=segment_hop,
            min_segments=min_segments,
        )

        experiment_config = ExperimentConfig(
            audio_type=audio_type,
        )

        return cls(
            audio=audio_config,
            experiment=experiment_config,
        )
