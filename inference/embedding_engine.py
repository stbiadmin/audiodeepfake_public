"""MS-CLAP embedding extraction with MPS acceleration.

Extracts 1024-dimensional embeddings from audio segments using Microsoft CLAP.
"""

import numpy as np
import torch
from typing import List, Optional

from .model_registry import get_registry

# MS-CLAP expects 7 seconds of audio at 48kHz
TARGET_DURATION = 7.0  # seconds
TARGET_SAMPLE_RATE = 48000
TARGET_LENGTH = int(TARGET_DURATION * TARGET_SAMPLE_RATE)

# Default batch size for embedding extraction
DEFAULT_BATCH_SIZE = 16


class EmbeddingEngine:
    """MS-CLAP embedding extraction with MPS acceleration."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """Initialize embedding engine.

        Args:
            batch_size: Number of segments to process at once
        """
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _ensure_model(self) -> None:
        """Lazy load the model from registry."""
        if self._model is None:
            registry = get_registry()
            self._model, self._device = registry.get_embedding_model()

    def warmup(self) -> None:
        """Pre-load model to eliminate cold start latency."""
        self._ensure_model()

    def extract(
        self,
        segments: List[np.ndarray],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> np.ndarray:
        """Extract embeddings from audio segments.

        Args:
            segments: List of audio segments (numpy arrays)
            sample_rate: Sample rate of segments (default: 48000)

        Returns:
            Embeddings array of shape (n_segments, 1024)
        """
        self._ensure_model()

        n_segments = len(segments)
        all_embeddings = []

        # Process in batches
        for i in range(0, n_segments, self.batch_size):
            batch = segments[i:i + self.batch_size]
            batch_embeddings = self._extract_batch(batch, sample_rate)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _extract_batch(
        self,
        segments: List[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        """Extract embeddings for a batch of segments.

        Args:
            segments: List of audio segments
            sample_rate: Sample rate

        Returns:
            Embeddings array of shape (batch_size, 1024)
        """
        # Prepare audio tensors
        audio_tensors = []

        for segment in segments:
            # Pad or tile to target length (MS-CLAP requires 7s)
            audio = self._prepare_segment(segment, sample_rate)
            audio_tensors.append(torch.FloatTensor(audio).reshape(1, -1))

        # Stack into batch: (N, 1, T)
        batch_tensor = torch.stack(audio_tensors, dim=0)

        # Extract embeddings
        with torch.no_grad():
            # Reshape and move to device
            batch_tensor = batch_tensor.reshape(len(segments), -1).to(self._device)
            embeddings = self._model.clap.audio_encoder(batch_tensor)[0]

        return embeddings.detach().cpu().numpy()

    def _prepare_segment(
        self,
        segment: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Prepare segment for MS-CLAP (pad/tile to 7 seconds).

        Args:
            segment: Audio segment
            sample_rate: Sample rate

        Returns:
            Prepared audio array of exactly TARGET_LENGTH samples
        """
        # Calculate target length for this sample rate
        target_length = int(TARGET_DURATION * sample_rate)

        audio = segment.flatten()

        if target_length >= len(audio):
            # Tile to fill
            repeat_factor = int(np.ceil(target_length / len(audio)))
            audio = np.tile(audio, repeat_factor)[:target_length]
        else:
            # Trim from start (deterministic)
            audio = audio[:target_length]

        return audio.astype(np.float32)

    def extract_single(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> np.ndarray:
        """Extract embedding for a single audio array.

        Args:
            audio: Full audio array (will be prepared as single segment)
            sample_rate: Sample rate

        Returns:
            Embedding array of shape (1, 1024)
        """
        return self.extract([audio], sample_rate)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return 1024

    @property
    def device(self) -> Optional[torch.device]:
        """Return current device."""
        return self._device


# Module-level singleton for convenience
_default_engine: Optional[EmbeddingEngine] = None


def get_engine() -> EmbeddingEngine:
    """Get the default embedding engine singleton."""
    global _default_engine
    if _default_engine is None:
        _default_engine = EmbeddingEngine()
    return _default_engine


def extract_embeddings(
    segments: List[np.ndarray],
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Convenience function to extract embeddings.

    Args:
        segments: List of audio segments
        sample_rate: Sample rate

    Returns:
        Embeddings array of shape (n_segments, 1024)
    """
    return get_engine().extract(segments, sample_rate)
