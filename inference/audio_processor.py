"""Fast audio loading and segmentation.

Uses soundfile + scipy.resample_poly for ~21x faster loading than librosa.
"""

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Domain-specific segment settings (matches training config/base.py)
SEGMENT_CONFIGS = {
    'speech': {
        'segment_duration': 2.0,  # seconds (matches training)
        'segment_hop': 1.0,       # 50% overlap (matches training)
        'min_segments': 3,        # matches training
        'min_duration': 4.0,      # 3 segments * 2s - 2 overlaps = 4s
    },
    'music': {
        'segment_duration': 2.0,  # matches training
        'segment_hop': 1.0,       # matches training
        'min_segments': 3,
        'min_duration': 4.0,
    },
}

TARGET_SAMPLE_RATE = 48000  # MS-CLAP requirement

# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}


class AudioLoadError(Exception):
    """Raised when audio cannot be loaded."""
    pass


class UnsupportedFormatError(Exception):
    """Raised when audio format is not supported."""
    pass


class AudioTooShortError(Exception):
    """Raised when audio is too short for processing."""
    pass


@dataclass
class AudioInfo:
    """Information about loaded audio."""
    duration: float
    sample_rate: int
    n_samples: int
    n_segments: int
    domain: str


def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Uses soundfile for I/O and scipy.resample_poly for fast resampling.
    This is ~21x faster than librosa.load.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 48000 for MS-CLAP)

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        UnsupportedFormatError: If format not supported
        AudioLoadError: If file cannot be loaded
    """
    path = Path(path)

    # Check format
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported format: {suffix}. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    try:
        # Load with soundfile (very fast, C-based)
        audio, sr = sf.read(str(path), dtype='float32', always_2d=False)

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            audio = _resample_fast(audio, sr, target_sr)

        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return audio.astype(np.float32), target_sr

    except Exception as e:
        raise AudioLoadError(f"Failed to load audio: {path}. Error: {e}")


def _resample_fast(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy.resample_poly (polyphase filter).

    This is much faster than librosa's kaiser_best resampling.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    # Find GCD to reduce ratio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    # Use polyphase resampling
    return resample_poly(audio, up, down).astype(np.float32)


def segment_audio(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    domain: str = 'speech',
    segment_duration: Optional[float] = None,
    segment_hop: Optional[float] = None,
) -> List[np.ndarray]:
    """Segment audio into fixed-length windows.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of audio
        domain: 'speech' or 'music' (determines default settings)
        segment_duration: Override segment duration (seconds)
        segment_hop: Override segment hop (seconds)

    Returns:
        List of audio segments as numpy arrays
    """
    # Get config for domain
    config = SEGMENT_CONFIGS.get(domain, SEGMENT_CONFIGS['speech'])

    seg_dur = segment_duration or config['segment_duration']
    seg_hop = segment_hop or config['segment_hop']

    segment_samples = int(seg_dur * sample_rate)
    hop_samples = int(seg_hop * sample_rate)

    segments = []
    start = 0

    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples

    return segments


def get_audio_duration(path: Union[str, Path]) -> float:
    """Get audio duration without loading full file.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    info = sf.info(str(path))
    return info.duration


def get_segment_count(
    duration: float,
    domain: str = 'speech',
    segment_duration: Optional[float] = None,
    segment_hop: Optional[float] = None,
) -> int:
    """Calculate number of segments for a given duration.

    Args:
        duration: Audio duration in seconds
        domain: 'speech' or 'music'
        segment_duration: Override segment duration
        segment_hop: Override segment hop

    Returns:
        Number of segments
    """
    config = SEGMENT_CONFIGS.get(domain, SEGMENT_CONFIGS['speech'])

    seg_dur = segment_duration or config['segment_duration']
    seg_hop = segment_hop or config['segment_hop']

    if duration < seg_dur:
        return 0

    return int((duration - seg_dur) / seg_hop) + 1


def validate_audio(
    path: Union[str, Path],
    domain: str = 'speech',
) -> AudioInfo:
    """Validate audio file meets requirements.

    Args:
        path: Path to audio file
        domain: 'speech' or 'music'

    Returns:
        AudioInfo with file details

    Raises:
        AudioTooShortError: If audio is too short
        UnsupportedFormatError: If format not supported
    """
    path = Path(path)

    # Check format
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported format: {suffix}. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    # Get duration
    try:
        info = sf.info(str(path))
        duration = info.duration
        sample_rate = info.samplerate
    except Exception as e:
        raise AudioLoadError(f"Cannot read audio file: {e}")

    # Check segment count
    config = SEGMENT_CONFIGS.get(domain, SEGMENT_CONFIGS['speech'])
    n_segments = get_segment_count(duration, domain)

    if n_segments < config['min_segments']:
        raise AudioTooShortError(
            f"Audio too short: {duration:.1f}s yields {n_segments} segments, "
            f"need at least {config['min_segments']} for {domain}. "
            f"Minimum duration: {config['min_duration']}s"
        )

    return AudioInfo(
        duration=duration,
        sample_rate=sample_rate,
        n_samples=int(duration * sample_rate),
        n_segments=n_segments,
        domain=domain,
    )


def process_audio(
    path: Union[str, Path],
    domain: str = 'speech',
) -> Tuple[List[np.ndarray], AudioInfo]:
    """Load and segment audio file in one call.

    Args:
        path: Path to audio file
        domain: 'speech' or 'music'

    Returns:
        Tuple of (segments, audio_info)
    """
    # Validate first (fast - just reads metadata)
    info = validate_audio(path, domain)

    # Load and segment
    audio, sr = load_audio(path)
    segments = segment_audio(audio, sr, domain)

    # Update info with actual values
    info.n_samples = len(audio)
    info.duration = len(audio) / sr
    info.n_segments = len(segments)

    return segments, info
