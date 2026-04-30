"""Core processing modules for audio deepfake detection."""

from .segmenter import AudioSegmenter
from .similarity_computer import SimilarityComputer
from .feature_extractor import DistributionFeatureExtractor

__all__ = [
    'AudioSegmenter',
    'SimilarityComputer',
    'DistributionFeatureExtractor',
]
