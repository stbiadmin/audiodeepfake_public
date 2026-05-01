"""Core processing modules for audio deepfake detection."""

from .feature_extractor import DistributionFeatureExtractor
from .segmenter import AudioSegmenter
from .similarity_computer import SimilarityComputer

__all__ = [
    'AudioSegmenter',
    'SimilarityComputer',
    'DistributionFeatureExtractor',
]
