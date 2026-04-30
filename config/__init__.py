"""Configuration module for audio deepfake detection."""

from .base import AudioConfig, EmbeddingConfig, FeatureConfig, ExperimentConfig
from .audio_types import AUDIO_TYPE_CONFIGS, get_audio_type_config

__all__ = [
    'AudioConfig',
    'EmbeddingConfig',
    'FeatureConfig',
    'ExperimentConfig',
    'AUDIO_TYPE_CONFIGS',
    'get_audio_type_config',
]
