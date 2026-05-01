"""Main feature extraction pipeline for audio deepfake detection.

This script processes audio files through the complete pipeline:
1. Audio segmentation into fixed-length windows
2. CLAP embedding extraction for each segment
3. Pairwise cosine similarity computation
4. Statistical feature extraction from similarity distribution
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.audio_types import get_audio_type_config
from config.base import PipelineConfig
from core.feature_extractor import DistributionFeatureExtractor
from core.segmenter import AudioSegmenter
from core.similarity_computer import SimilarityComputer


@dataclass
class ExtractionResult:
    """Result of feature extraction for a single audio file."""
    file_path: str
    audio_type: str
    label: Optional[int]  # 0=real, 1=fake, None=unknown
    duration: float
    segment_count: int
    similarity_count: int
    features: Dict[str, float]
    embeddings: Optional[np.ndarray] = None
    similarities: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

    def to_dict(self, include_arrays: bool = False) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'file_path': self.file_path,
            'audio_type': self.audio_type,
            'label': self.label,
            'duration': self.duration,
            'segment_count': self.segment_count,
            'similarity_count': self.similarity_count,
            'features': self.features,
            'metadata': self.metadata or {},
        }
        if include_arrays:
            if self.embeddings is not None:
                result['embeddings'] = self.embeddings.tolist()
            if self.similarities is not None:
                result['similarities'] = self.similarities.tolist()
        return result


class FeatureExtractionPipeline:
    """Complete feature extraction pipeline for audio deepfake detection."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        embedding_model: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
            embedding_model: Default embedding model ("laion_clap" or "msclap").
                            If None, uses audio_type config or defaults to "laion_clap".
            verbose: Whether to print progress
        """
        self.config = config or PipelineConfig()
        self.default_embedding_model = embedding_model or "laion_clap"
        self.verbose = verbose

        # Initialize components
        self.segmenter = AudioSegmenter(self.config.audio)
        self.similarity_computer = SimilarityComputer()
        self.feature_extractor = DistributionFeatureExtractor(self.config.feature)

        # Embedding extractors cache (lazy-loaded per model type)
        self._embedding_extractors: Dict[str, any] = {}

    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    def _get_embedding_extractor(self, model_type: Optional[str] = None):
        """Lazy-load the embedding extractor for the specified model type.

        Args:
            model_type: Embedding model type. If None, uses default.
                       Supported: laion_clap, msclap, wavlm, wav2vec2

        Returns:
            The embedding extractor instance
        """
        model_type = model_type or self.default_embedding_model

        if model_type not in self._embedding_extractors:
            self._log(f"  Loading embedding model: {model_type}")
            if model_type == "laion_clap":
                from utilities.clap_inference import AudioEmbeddingExtractor
                self._embedding_extractors[model_type] = AudioEmbeddingExtractor()
            elif model_type == "msclap":
                from utilities.msclap_inference import AudioEmbeddingExtractor
                self._embedding_extractors[model_type] = AudioEmbeddingExtractor(
                    config={'version': self.config.embedding.msclap_version}
                )
            elif model_type == "wavlm":
                from utilities.wavlm_inference import AudioEmbeddingExtractor
                self._embedding_extractors[model_type] = AudioEmbeddingExtractor()
            elif model_type == "wav2vec2":
                from utilities.wav2vec2_inference import AudioEmbeddingExtractor
                self._embedding_extractors[model_type] = AudioEmbeddingExtractor()
            elif model_type == "aasist":
                from utilities.aasist_inference import AudioEmbeddingExtractor
                self._embedding_extractors[model_type] = AudioEmbeddingExtractor()
            else:
                raise ValueError(f"Unknown embedding model: {model_type}. "
                               f"Supported: laion_clap, msclap, wavlm, wav2vec2, aasist")

        return self._embedding_extractors[model_type]

    def extract_embeddings_from_segments(
        self,
        segments: List[np.ndarray],
        sample_rate: int,
        embedding_model: Optional[str] = None,
    ) -> np.ndarray:
        """Extract embeddings for a list of audio segments.

        Args:
            segments: List of audio segment arrays
            sample_rate: Sample rate of the audio
            embedding_model: Which embedding model to use (None = default)

        Returns:
            Array of embeddings with shape (n_segments, embedding_dim)
        """
        extractor = self._get_embedding_extractor(embedding_model)

        embeddings = []
        batch_size = self.config.embedding.batch_size

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            # Stack segments into array with shape (batch_size, samples)
            # Pad shorter segments to match longest in batch
            max_len = max(len(s) for s in batch)
            batch_array = np.zeros((len(batch), max_len), dtype=np.float32)
            for j, seg in enumerate(batch):
                batch_array[j, :len(seg)] = seg

            batch_embeddings = extractor.extract_from_audio_data(batch_array)

            # Handle different return types
            if isinstance(batch_embeddings, np.ndarray):
                if batch_embeddings.ndim == 1:
                    embeddings.append(batch_embeddings.reshape(1, -1))
                else:
                    embeddings.append(batch_embeddings)
            elif isinstance(batch_embeddings, list):
                embeddings.extend([np.array(e).reshape(1, -1) if np.array(e).ndim == 1 else np.array(e) for e in batch_embeddings])
            else:
                # Assume tensor-like
                embeddings.append(np.array(batch_embeddings))

        return np.vstack(embeddings)

    def process_file(
        self,
        file_path: str,
        audio_type: str = "single_voice",
        label: Optional[int] = None,
        metadata: Optional[Dict] = None,
        store_arrays: bool = False,
        embedding_model: Optional[str] = None,
    ) -> ExtractionResult:
        """Process a single audio file through the pipeline.

        Args:
            file_path: Path to the audio file
            audio_type: Type of audio content
            label: Label (0=real, 1=fake)
            metadata: Additional metadata
            store_arrays: Whether to store embeddings/similarities in result
            embedding_model: Override embedding model (None = use audio_type config)

        Returns:
            ExtractionResult with features and metadata
        """
        file_path = str(file_path)
        self._log(f"Processing: {file_path}")

        # Get audio type configuration
        type_config = get_audio_type_config(audio_type)

        # Determine embedding model: explicit override > audio_type config > default
        effective_embedding_model = (
            embedding_model or
            type_config.get('embedding_model') or
            self.default_embedding_model
        )

        # Update segment duration based on audio type
        segment_duration = type_config.get(
            'segment_duration',
            self.config.audio.segment_duration
        )

        # Step 1: Segment audio
        self._log("  Segmenting audio...")
        segments, segment_info = self.segmenter.segment_file(
            file_path,
            segment_duration=segment_duration,
            return_info=True,
        )
        duration = self.segmenter.get_audio_duration(file_path)
        self._log(f"  Duration: {duration:.1f}s, Segments: {len(segments)}")

        if len(segments) < self.config.audio.min_segments:
            raise ValueError(
                f"Not enough segments: {len(segments)} < {self.config.audio.min_segments}"
            )

        # Step 2: Extract embeddings using the determined model
        self._log(f"  Extracting embeddings with {effective_embedding_model}...")
        embeddings = self.extract_embeddings_from_segments(
            segments,
            self.config.audio.sample_rate,
            embedding_model=effective_embedding_model,
        )
        self._log(f"  Embeddings shape: {embeddings.shape}")

        # Step 3: Compute pairwise similarities
        self._log("  Computing similarities...")
        similarities = self.similarity_computer.compute_pairwise(embeddings)
        self._log(f"  Similarity pairs: {len(similarities)}")

        # Step 4: Extract statistical features
        self._log("  Extracting features...")
        features = self.feature_extractor.extract_all_features(similarities)
        self._log(f"  Features: {list(features.keys())}")

        # Include embedding model in metadata
        result_metadata = metadata.copy() if metadata else {}
        result_metadata['embedding_model'] = effective_embedding_model

        return ExtractionResult(
            file_path=file_path,
            audio_type=audio_type,
            label=label,
            duration=duration,
            segment_count=len(segments),
            similarity_count=len(similarities),
            features=features,
            embeddings=embeddings if store_arrays else None,
            similarities=similarities if store_arrays else None,
            metadata=result_metadata,
        )

    def process_directory(
        self,
        directory: str,
        audio_type: str = "single_voice",
        label: Optional[int] = None,
        pattern: str = "**/*.wav",
        output_file: Optional[str] = None,
        embedding_model: Optional[str] = None,
        store_arrays: bool = False,
        checkpoint_interval: int = 500,
        resume: bool = True,
    ) -> List[ExtractionResult]:
        """Process all audio files in a directory.

        Args:
            directory: Path to directory
            audio_type: Type of audio content
            label: Label for all files (0=real, 1=fake)
            pattern: Glob pattern for finding audio files
            output_file: Optional path to save results as JSON
            embedding_model: Override embedding model (None = use audio_type config)
            store_arrays: Whether to store embeddings/similarities in results
            checkpoint_interval: Save checkpoint every N files (default 500)
            resume: If True, resume from existing checkpoint if available

        Returns:
            List of ExtractionResult objects
        """
        directory = Path(directory)
        audio_files = list(directory.glob(pattern))

        # Also check for other common audio formats (only if using default recursive pattern)
        if pattern == "**/*.wav":
            audio_files.extend(directory.glob("**/*.mp3"))
            audio_files.extend(directory.glob("**/*.flac"))
        elif pattern == "*.wav":
            # Non-recursive: also check mp3/flac in same directory
            audio_files.extend(directory.glob("*.mp3"))
            audio_files.extend(directory.glob("*.flac"))

        self._log(f"Found {len(audio_files)} audio files in {directory}")

        # Load existing checkpoint if resuming
        results = []
        processed_files = set()
        checkpoint_file = None

        if output_file:
            checkpoint_file = Path(str(output_file) + '.checkpoint')
            if resume and checkpoint_file.exists():
                self._log(f"Loading checkpoint from {checkpoint_file}")
                try:
                    with open(checkpoint_file, 'r') as f:
                        content = f.read()
                        if not content.strip():
                            self._log("WARNING: Checkpoint file is empty, starting fresh")
                        else:
                            checkpoint_data = json.loads(content)
                            results = [ExtractionResult(**{
                                'file_path': r['file_path'],
                                'audio_type': r['audio_type'],
                                'label': r['label'],
                                'duration': r['duration'],
                                'segment_count': r['segment_count'],
                                'similarity_count': r['similarity_count'],
                                'features': r['features'],
                                'embeddings': np.array(r['embeddings']) if r.get('embeddings') else None,
                                'similarities': np.array(r['similarities']) if r.get('similarities') else None,
                                'metadata': r.get('metadata'),
                            }) for r in checkpoint_data]
                            processed_files = {r.file_path for r in results}
                            self._log(f"Resumed with {len(results)} already processed files")
                except json.JSONDecodeError as e:
                    self._log(f"WARNING: Checkpoint file corrupted ({e}), starting fresh")
                except Exception as e:
                    self._log(f"WARNING: Error loading checkpoint ({e}), starting fresh")

        # Filter out already processed files
        if processed_files:
            audio_files = [f for f in audio_files if str(f) not in processed_files]
            self._log(f"Remaining files to process: {len(audio_files)}")

        files_since_checkpoint = 0
        for file_path in audio_files:
            try:
                result = self.process_file(
                    str(file_path),
                    audio_type=audio_type,
                    label=label,
                    embedding_model=embedding_model,
                    store_arrays=store_arrays,
                )
                results.append(result)
                files_since_checkpoint += 1

                # Periodic checkpoint - use atomic write
                if checkpoint_file and files_since_checkpoint >= checkpoint_interval:
                    self._log(f"Saving checkpoint ({len(results)} files processed)...")
                    temp_checkpoint = Path(str(checkpoint_file) + '.tmp')
                    try:
                        with open(temp_checkpoint, 'w') as f:
                            json.dump(
                                [r.to_dict(include_arrays=store_arrays) for r in results],
                                f,
                            )
                        # Atomic rename - only replaces checkpoint if write succeeded
                        temp_checkpoint.rename(checkpoint_file)
                        files_since_checkpoint = 0
                    except Exception as e:
                        self._log(f"WARNING: Failed to save checkpoint ({e})")
                        # Try to clean up temp file if it exists
                        if temp_checkpoint.exists():
                            temp_checkpoint.unlink()

            except Exception as e:
                self._log(f"  Error processing {file_path}: {e}")

        if output_file:
            self._log(f"Saving final results to {output_file}")
            # Use atomic write for final output too
            output_path = Path(output_file)
            temp_output = Path(str(output_file) + '.tmp')
            try:
                with open(temp_output, 'w') as f:
                    json.dump(
                        [r.to_dict(include_arrays=store_arrays) for r in results],
                        f,
                        indent=2,
                    )
                # Atomic rename
                temp_output.rename(output_path)
                self._log(f"Successfully saved {len(results)} results to {output_file}")
                # Only remove checkpoint after final output is confirmed written
                if checkpoint_file and checkpoint_file.exists():
                    checkpoint_file.unlink()
                    self._log("Checkpoint file removed after successful completion")
            except Exception as e:
                self._log(f"ERROR: Failed to save final output ({e})")
                self._log("Checkpoint file preserved for recovery")
                # Try to clean up temp file if it exists
                if temp_output.exists():
                    temp_output.unlink()

        return results

    def process_dataset(
        self,
        real_dir: str,
        fake_dir: str,
        audio_type: str = "single_voice",
        output_file: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Tuple[List[ExtractionResult], List[ExtractionResult]]:
        """Process a dataset with real and fake audio directories.

        Args:
            real_dir: Directory containing real audio
            fake_dir: Directory containing fake audio
            audio_type: Type of audio content
            output_file: Optional path to save combined results
            embedding_model: Override embedding model (None = use audio_type config)

        Returns:
            Tuple of (real_results, fake_results)
        """
        self._log(f"Processing dataset: real={real_dir}, fake={fake_dir}")

        real_results = self.process_directory(
            real_dir,
            audio_type=audio_type,
            label=0,
            embedding_model=embedding_model,
        )
        self._log(f"Processed {len(real_results)} real files")

        fake_results = self.process_directory(
            fake_dir,
            audio_type=audio_type,
            label=1,
            embedding_model=embedding_model,
        )
        self._log(f"Processed {len(fake_results)} fake files")

        if output_file:
            all_results = real_results + fake_results
            with open(output_file, 'w') as f:
                json.dump(
                    [r.to_dict() for r in all_results],
                    f,
                    indent=2,
                )

        return real_results, fake_results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract statistical features from audio files for deepfake detection"
    )
    parser.add_argument(
        "input",
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--audio-type", "-t",
        default="single_voice",
        help="Audio type (single_voice, multi_voice, music_instrumental, etc.)"
    )
    parser.add_argument(
        "--label", "-l",
        type=int,
        choices=[0, 1],
        help="Label for files (0=real, 1=fake)"
    )
    parser.add_argument(
        "--embedding-model", "-m",
        default="laion_clap",
        choices=["laion_clap", "msclap", "wavlm", "wav2vec2", "aasist"],
        help="Embedding model to use (laion_clap, msclap, wavlm, wav2vec2, aasist)"
    )
    parser.add_argument(
        "--segment-duration", "-s",
        type=float,
        default=2.0,
        help="Segment duration in seconds"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--store-arrays",
        action="store_true",
        help="Store embeddings and similarities in output (larger files)"
    )
    parser.add_argument(
        "--pattern", "-p",
        default="**/*.wav",
        help="Glob pattern for finding audio files (default: **/*.wav, use *.flac for flat dirs)"
    )

    args = parser.parse_args()

    # Create configuration from audio type (gets segment settings from audio_types.py)
    if args.audio_type:
        config = PipelineConfig.from_audio_type(args.audio_type)
    else:
        config = PipelineConfig()

    # Override segment_duration if explicitly provided via CLI
    if args.segment_duration != 2.0:  # Only override if non-default
        config.audio.segment_duration = args.segment_duration

    # Create pipeline
    pipeline = FeatureExtractionPipeline(
        config=config,
        embedding_model=args.embedding_model,
        verbose=not args.quiet,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        result = pipeline.process_file(
            str(input_path),
            audio_type=args.audio_type,
            label=args.label,
            store_arrays=args.store_arrays,
            embedding_model=args.embedding_model,
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(include_arrays=args.store_arrays), f, indent=2)
        else:
            print(json.dumps(result.to_dict(include_arrays=args.store_arrays), indent=2))

    elif input_path.is_dir():
        # Process directory
        results = pipeline.process_directory(
            str(input_path),
            audio_type=args.audio_type,
            label=args.label,
            pattern=args.pattern,
            output_file=args.output,
            embedding_model=args.embedding_model,
            store_arrays=args.store_arrays,
        )

        if not args.output:
            print(f"Processed {len(results)} files")
            for r in results:
                print(f"  {r.file_path}: {len(r.features)} features")

    else:
        print(f"Error: {args.input} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
