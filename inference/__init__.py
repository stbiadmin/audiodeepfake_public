"""Audio Deepfake Detection Inference Pipeline.

Simple API for detecting AI-generated audio in speech and music.

Quick Start:
    from inference import detect

    # Basic usage - auto-detects domain
    result = detect('audio.mp3')
    print(result)  # {'label': 'fake', 'confidence': 0.92, 'model': 'speech'}

    # Explicit model selection
    result = detect('song.wav', model='music')           # F1=0.938, fastest
    result = detect('interview.mp3', model='speech')     # F1=0.654, fast
    result = detect('interview.mp3', model='speech_ensemble')  # F1=0.675, best

Model Options:
    | Model              | Domain | F1    | Speed (30s) | Notes                    |
    |--------------------|--------|-------|-------------|--------------------------|
    | 'auto'             | Auto   | Varies| ~700ms      | Heuristic domain detect  |
    | 'music'            | Music  | 0.938 | ~600ms      | Best for songs           |
    | 'speech'           | Speech | 0.654 | ~700ms      | Single model, fast       |
    | 'speech_ensemble'  | Speech | 0.675 | ~900ms      | 5-expert, best accuracy  |

Advanced Usage:
    from inference import AudioDeepfakeDetector

    detector = AudioDeepfakeDetector(
        speech_model='ds_msclap',
        music_model='mi_adaptive',
    )
    detector.warmup()  # Pre-load models for low latency

    result = detector.detect('audio.mp3', return_metadata=True)
"""

__version__ = '1.0.0'
__all__ = ['detect', 'detect_batch', 'AudioDeepfakeDetector']

from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .detector import AudioDeepfakeDetector
from .audio_processor import (
    AudioLoadError,
    UnsupportedFormatError,
    AudioTooShortError,
)

# Module-level singleton for simple API
_default_detector: Optional[AudioDeepfakeDetector] = None


def _get_detector() -> AudioDeepfakeDetector:
    """Get the default detector singleton."""
    global _default_detector
    if _default_detector is None:
        _default_detector = AudioDeepfakeDetector()
    return _default_detector


def detect(
    audio_path: Union[str, Path],
    model: str = 'auto',
    return_metadata: bool = False,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect if audio is real or AI-generated.

    Args:
        audio_path: Path to audio file (mp3, wav, flac, etc.)
        model: Detection model to use:
            - 'auto': Auto-detect speech vs music (default)
            - 'music': Music deepfake detection (F1=0.938)
            - 'speech': Speech deepfake detection (F1=0.654)
            - 'speech_ensemble': 5-expert ensemble (F1=0.675, slower)
        return_metadata: Include detailed metadata in result
        threshold: Override decision threshold (default: model-specific)

    Returns:
        Dictionary with:
            - 'label': 'real' or 'fake'
            - 'confidence': 0.0-1.0
            - 'model': model used ('speech', 'music', or 'speech_ensemble')

        If return_metadata=True, also includes:
            - 'metadata': dict with n_segments, duration_seconds,
              mean_similarity, features

    Example:
        >>> from inference import detect
        >>> result = detect('speech.mp3')
        >>> print(result['label'], result['confidence'])
        fake 0.92

        >>> result = detect('song.wav', model='music', return_metadata=True)
        >>> print(result['metadata']['n_segments'])
        20
    """
    return _get_detector().detect(
        audio_path,
        model=model,
        return_metadata=return_metadata,
        threshold=threshold,
    )


def detect_batch(
    audio_paths: List[Union[str, Path]],
    model: str = 'auto',
    return_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """Detect multiple audio files efficiently.

    Shares model loading across all files for better performance.

    Args:
        audio_paths: List of paths to audio files
        model: Detection model (same options as detect())
        return_metadata: Include metadata in results

    Returns:
        List of result dictionaries (same format as detect())

    Example:
        >>> from inference import detect_batch
        >>> results = detect_batch(['file1.mp3', 'file2.wav'], model='speech')
        >>> for r in results:
        ...     print(r['label'], r['confidence'])
    """
    return _get_detector().detect_batch(
        audio_paths,
        model=model,
        return_metadata=return_metadata,
    )


def warmup() -> None:
    """Pre-load all models to eliminate cold start latency.

    Call this before time-critical detection. Models are loaded
    once and cached for subsequent calls.

    Example:
        >>> from inference import warmup, detect
        >>> warmup()  # Takes ~8s to load MS-CLAP
        >>> result = detect('audio.mp3')  # Now runs in <1s
    """
    _get_detector().warmup()
