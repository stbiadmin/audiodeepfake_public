"""Audio segmentation module for fixed-length windowing."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np

from config.base import AudioConfig


@dataclass
class SegmentInfo:
    """Information about a segment."""
    index: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration: float


class AudioSegmenter:
    """Segment audio into fixed-length windows for embedding extraction.

    This class handles loading audio files and splitting them into
    fixed-duration segments with configurable overlap.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the segmenter.

        Args:
            config: Audio configuration. Uses defaults if not provided.
        """
        self.config = config or AudioConfig()

    def load_audio(
        self,
        file_path: Union[str, Path],
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load an audio file and optionally resample.

        Args:
            file_path: Path to the audio file
            target_sr: Target sample rate. Uses config if not provided.

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        target_sr = target_sr or self.config.sample_rate

        # Load audio with librosa (handles various formats)
        audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)

        # Normalize if configured
        if self.config.normalize:
            audio = self._normalize_audio(audio)

        return audio, sr

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def segment_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_duration: Optional[float] = None,
        segment_hop: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Segment audio array into fixed-length windows.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            segment_duration: Duration of each segment in seconds
            segment_hop: Hop between segments in seconds

        Returns:
            List of audio segments as numpy arrays
        """
        segment_duration = segment_duration or self.config.segment_duration
        segment_hop = segment_hop or self.config.segment_hop

        segment_samples = int(segment_duration * sample_rate)
        hop_samples = int(segment_hop * sample_rate)

        segments = []
        start = 0

        while start + segment_samples <= len(audio):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
            start += hop_samples

        return segments

    def segment_file(
        self,
        file_path: Union[str, Path],
        segment_duration: Optional[float] = None,
        segment_hop: Optional[float] = None,
        return_info: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[SegmentInfo]]]:
        """Segment an audio file into fixed-length windows.

        Args:
            file_path: Path to the audio file
            segment_duration: Duration of each segment in seconds
            segment_hop: Hop between segments in seconds
            return_info: If True, also return segment timing information

        Returns:
            List of audio segments, optionally with segment info
        """
        audio, sr = self.load_audio(file_path)
        segments = self.segment_audio(
            audio, sr,
            segment_duration=segment_duration,
            segment_hop=segment_hop,
        )

        if return_info:
            segment_duration = segment_duration or self.config.segment_duration
            segment_hop = segment_hop or self.config.segment_hop
            segment_samples = int(segment_duration * sr)
            hop_samples = int(segment_hop * sr)

            infos = []
            for i, _ in enumerate(segments):
                start_sample = i * hop_samples
                end_sample = start_sample + segment_samples
                infos.append(SegmentInfo(
                    index=i,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time=start_sample / sr,
                    end_time=end_sample / sr,
                    duration=segment_duration,
                ))
            return segments, infos

        return segments

    def get_segment_count(
        self,
        audio_duration: float,
        segment_duration: Optional[float] = None,
        segment_hop: Optional[float] = None,
    ) -> int:
        """Calculate the number of segments for a given audio duration.

        Args:
            audio_duration: Total duration of audio in seconds
            segment_duration: Duration of each segment
            segment_hop: Hop between segments

        Returns:
            Number of segments
        """
        segment_duration = segment_duration or self.config.segment_duration
        segment_hop = segment_hop or self.config.segment_hop

        if audio_duration < segment_duration:
            return 0

        return int((audio_duration - segment_duration) / segment_hop) + 1

    def get_audio_duration(self, file_path: Union[str, Path]) -> float:
        """Get the duration of an audio file without loading it fully.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds
        """
        return librosa.get_duration(path=str(file_path))

    def validate_file(
        self,
        file_path: Union[str, Path],
        min_segments: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Validate that an audio file meets requirements.

        Args:
            file_path: Path to the audio file
            min_segments: Minimum required segments

        Returns:
            Tuple of (is_valid, message)
        """
        min_segments = min_segments or self.config.min_segments

        try:
            duration = self.get_audio_duration(file_path)
        except Exception as e:
            return False, f"Failed to read audio file: {e}"

        segment_count = self.get_segment_count(duration)

        if segment_count < min_segments:
            return False, (
                f"Audio too short: {duration:.1f}s yields {segment_count} segments, "
                f"minimum required is {min_segments}"
            )

        return True, f"Valid: {duration:.1f}s, {segment_count} segments"

    def segment_by_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speaker_segments: List[Tuple[float, float, str]],
    ) -> dict:
        """Segment audio by speaker using diarization results.

        Args:
            audio: Full audio array
            sample_rate: Sample rate
            speaker_segments: List of (start_time, end_time, speaker_id) tuples

        Returns:
            Dictionary mapping speaker_id to list of audio segments
        """
        speaker_audio = {}

        for start_time, end_time, speaker_id in speaker_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            segment_audio = audio[start_sample:end_sample]

            if speaker_id not in speaker_audio:
                speaker_audio[speaker_id] = []

            # Further segment this speaker's audio into fixed-length chunks
            speaker_segments_list = self.segment_audio(
                segment_audio,
                sample_rate,
            )
            speaker_audio[speaker_id].extend(speaker_segments_list)

        return speaker_audio
