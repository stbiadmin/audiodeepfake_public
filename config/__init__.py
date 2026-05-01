"""Configuration module for audio deepfake detection."""

from .audio_types import AUDIO_TYPE_CONFIGS, get_audio_type_config
from .base import AudioConfig, EmbeddingConfig, ExperimentConfig, FeatureConfig

__all__ = [
    'AudioConfig',
    'EmbeddingConfig',
    'FeatureConfig',
    'ExperimentConfig',
    'AUDIO_TYPE_CONFIGS',
    'get_audio_type_config',
]
