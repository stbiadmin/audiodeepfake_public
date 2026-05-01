"""Main AudioDeepfakeDetector class.

Orchestrates the full detection pipeline:
load -> segment -> embed -> similarity -> features -> classify
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .audio_processor import (
    AudioTooShortError,
    get_audio_duration,
    load_audio,
    segment_audio,
    validate_audio,
)
from .classifiers import (
    BaseClassifier,
    SpeechClassifier,
    SpeechEnsembleClassifier,
    get_classifier,
)
from .embedding_engine import EmbeddingEngine
from .feature_computer import compute_features, compute_similarities, get_similarity_stats


class AudioDeepfakeDetector:
    """Main class for audio deepfake detection.

    Provides a simple interface for detecting AI-generated audio
    in both speech and music domains.

    Example:
        detector = AudioDeepfakeDetector()
        result = detector.detect('audio.mp3')
        print(result)  # {'label': 'fake', 'confidence': 0.92, 'model': 'speech'}
    """

    def __init__(
        self,
        speech_model: str = 'ds_msclap',
        music_model: str = 'mi_adaptive',
    ):
        """Initialize detector with specified models.

        Args:
            speech_model: Speech classifier to use:
                - 'ds_msclap': Best single model (F1=0.654)
                - 'ensemble': 5-expert ensemble (F1=0.675)
                - Other: sv_msclap, sv_ds_msclap, mlaad_msclap, audeter_msclap
            music_model: Music classifier to use:
                - 'mi_adaptive': Domain-robust (F1=0.938)
                - 'mi_msclap': In-distribution (F1=0.995)
        """
        self.speech_model_name = speech_model
        self.music_model_name = music_model

        # Lazy-loaded components
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._speech_classifier: Optional[BaseClassifier] = None
        self._music_classifier: Optional[BaseClassifier] = None

    def _get_embedding_engine(self) -> EmbeddingEngine:
        """Get or create embedding engine."""
        if self._embedding_engine is None:
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    def _get_speech_classifier(self) -> BaseClassifier:
        """Get or create speech classifier."""
        if self._speech_classifier is None:
            if self.speech_model_name == 'ensemble':
                self._speech_classifier = SpeechEnsembleClassifier()
            else:
                self._speech_classifier = SpeechClassifier(self.speech_model_name)
        return self._speech_classifier

    def _get_music_classifier(self) -> BaseClassifier:
        """Get or create music classifier."""
        if self._music_classifier is None:
            self._music_classifier = get_classifier(self.music_model_name)
        return self._music_classifier

    def warmup(self) -> None:
        """Pre-load all models to eliminate cold start latency.

        Call this before time-critical detection to ensure
        models are loaded in memory.
        """
        engine = self._get_embedding_engine()
        engine.warmup()

        self._get_speech_classifier().warmup()
        self._get_music_classifier().warmup()

    def detect(
        self,
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
                - 'speech_ensemble': 5-expert ensemble (F1=0.675)
            return_metadata: Include detailed metadata in result
            threshold: Override decision threshold (default: model-specific)

        Returns:
            Dictionary with:
                - 'label': 'real' or 'fake'
                - 'confidence': 0.0-1.0
                - 'model': model used ('speech', 'music', or 'speech_ensemble')
            If return_metadata=True, also includes:
                - 'metadata': dict with n_segments, duration, mean_similarity, features
        """
        path = Path(audio_path)

        # Determine domain
        if model == 'auto':
            domain = self._detect_domain(path)
        elif model in ('music', 'mi_adaptive', 'mi_msclap'):
            domain = 'music'
        else:
            domain = 'speech'

        # Validate audio
        try:
            info = validate_audio(path, domain)
        except AudioTooShortError as e:
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'model': model,
                'error': str(e),
            }

        # Load and segment audio
        audio, sr = load_audio(path)
        segments = segment_audio(audio, sr, domain)

        # Extract embeddings
        engine = self._get_embedding_engine()
        embeddings = engine.extract(segments, sr)

        # Compute similarities
        similarities = compute_similarities(embeddings)

        # Get classifier and required features
        if model == 'speech_ensemble':
            classifier = SpeechEnsembleClassifier()
        elif domain == 'speech':
            classifier = self._get_speech_classifier()
        else:
            classifier = self._get_music_classifier()

        required_features = classifier.get_required_features()
        features = compute_features(similarities, required_features)

        # Make prediction
        label, confidence = classifier.predict(features)

        # Apply custom threshold if provided
        if threshold is not None:
            # Reinterpret confidence as fake probability
            fake_prob = confidence if label == 'fake' else (1 - confidence)
            label = 'fake' if fake_prob >= threshold else 'real'
            confidence = fake_prob if label == 'fake' else (1 - fake_prob)

        # Build result
        result = {
            'label': label,
            'confidence': confidence,
            'model': 'speech_ensemble' if model == 'speech_ensemble' else domain,
        }

        if return_metadata:
            sim_stats = get_similarity_stats(similarities)
            result['metadata'] = {
                'n_segments': len(segments),
                'duration_seconds': info.duration,
                'mean_similarity': sim_stats['mean'],
                'features': features,
            }

        return result

    def detect_batch(
        self,
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
        """
        # Ensure models are loaded
        self.warmup()

        results = []
        for path in audio_paths:
            try:
                result = self.detect(path, model=model, return_metadata=return_metadata)
            except Exception as e:
                result = {
                    'label': 'error',
                    'confidence': 0.0,
                    'model': model,
                    'error': str(e),
                }
            results.append(result)

        return results

    def _detect_domain(self, path: Path) -> str:
        """Auto-detect whether audio is speech or music.

        Uses a simple heuristic based on audio variance and duration.
        This is a basic implementation - could be improved with
        a dedicated classifier.

        Args:
            path: Path to audio file

        Returns:
            'speech' or 'music'
        """
        try:
            # Quick heuristic: check duration
            duration = get_audio_duration(path)

            # Very short clips are likely speech
            if duration < 10:
                return 'speech'

            # Load a sample to check characteristics
            audio, sr = load_audio(path)

            # Sample middle 5 seconds
            mid = len(audio) // 2
            sample_len = min(5 * sr, len(audio))
            sample = audio[mid - sample_len//2 : mid + sample_len//2]

            # Compute variance - music typically has lower variance
            # due to more consistent energy levels
            variance = float(np.var(sample))

            # Simple threshold (could be improved)
            if variance < 0.02:
                return 'music'
            else:
                return 'speech'

        except Exception:
            # Default to speech if detection fails
            return 'speech'

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about configured models.

        Returns:
            Dictionary with model names and expected performance
        """
        return {
            'speech_model': self.speech_model_name,
            'music_model': self.music_model_name,
            'speech_features': self._get_speech_classifier().get_required_features(),
            'music_features': self._get_music_classifier().get_required_features(),
            'expected_performance': {
                'speech': {
                    'ds_msclap': {'f1': 0.654, 'note': 'Best single model, fake-biased'},
                    'ensemble': {'f1': 0.675, 'note': '5-expert ensemble, balanced'},
                },
                'music': {
                    'mi_adaptive': {'f1': 0.938, 'note': 'Domain-robust'},
                    'mi_msclap': {'f1': 0.995, 'note': 'In-distribution only'},
                },
            },
        }
