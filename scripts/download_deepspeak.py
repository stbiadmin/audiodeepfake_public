"""Download and set up DeepSpeak v2 dataset for audio deepfake detection.

This script:
1. Downloads the DeepSpeak v2 dataset from HuggingFace
2. Extracts audio from the video files
3. Organizes audio by real/fake labels and identity
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_ffmpeg_path() -> str:
    """Get the path to ffmpeg, preferring local installation."""
    script_dir = Path(__file__).parent.parent
    local_ffmpeg = script_dir / "bin" / "ffmpeg"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    return "ffmpeg"  # Fall back to system ffmpeg


def extract_audio_from_video(video_path: str, audio_path: str, sample_rate: int = 48000) -> bool:
    """Extract audio from a video file using ffmpeg.

    Args:
        video_path: Path to input video
        audio_path: Path to output audio file
        sample_rate: Target sample rate

    Returns:
        True if successful, False otherwise
    """
    try:
        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path, '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            audio_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def setup_deepspeak_v2(
    output_dir: str = "data/deepspeak_v2",
    max_samples: Optional[int] = None,
    split: str = "train",
):
    """Download and set up DeepSpeak v2 dataset.

    Args:
        output_dir: Directory to save extracted audio
        max_samples: Maximum number of samples to process (None for all)
        split: Dataset split to use ("train" or "test")
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    # Organize by split to keep train/test separate
    split_dir = output_dir / split
    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DeepSpeak v2 dataset ({split} split only)...")
    # Pass split= to avoid downloading all splits
    data = load_dataset("faridlab/deepspeak_v2", split=split, trust_remote_code=True)
    print(f"Total samples in {split}: {len(data)}")

    if max_samples:
        print(f"Processing first {max_samples} samples")

    # Track metadata
    metadata = {
        'real': [],
        'fake': [],
    }

    processed = 0
    errors = 0

    for i, item in enumerate(data):
        if max_samples and processed >= max_samples:
            break

        video_path = item.get('video-file') or item.get('video_file')
        data_type = item.get('type', 'unknown')

        if not video_path or not os.path.exists(video_path):
            print(f"  Skipping {i}: video file not found")
            errors += 1
            continue

        # Determine output directory and identity
        if data_type == 'real':
            meta = item.get('metadata-real', {}) or item.get('metadata_real', {})
            identity = meta.get('identity', f'unknown_{i}')
            base_dir = real_dir
        elif data_type == 'fake':
            meta = item.get('metadata-fake', {}) or item.get('metadata_fake', {})
            identity = meta.get('identity-target', f'unknown_{i}')
            base_dir = fake_dir
        else:
            print(f"  Skipping {i}: unknown type '{data_type}'")
            continue

        # Create identity subdirectory
        identity_dir = base_dir / str(identity)
        identity_dir.mkdir(parents=True, exist_ok=True)

        # Generate audio filename
        video_name = Path(video_path).stem
        audio_path = identity_dir / f"{video_name}.wav"

        if audio_path.exists():
            print(f"  Skipping {i}: audio already exists")
            processed += 1
            continue

        print(f"  [{processed+1}] Extracting audio from: {video_path}")

        if extract_audio_from_video(str(video_path), str(audio_path)):
            processed += 1

            # Store metadata
            meta_entry = {
                'file_path': str(audio_path),
                'identity': str(identity),
                'video_source': str(video_path),
            }

            if data_type == 'fake':
                meta_entry.update({
                    'kind': meta.get('kind'),
                    'engine': meta.get('engine'),
                    'audio_config': meta.get('audio-config'),
                })

            metadata[data_type].append(meta_entry)
        else:
            print("    Failed to extract audio")
            errors += 1

    # Save metadata in the split directory
    metadata_path = split_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Real samples: {len(metadata['real'])}")
    print(f"  Fake samples: {len(metadata['fake'])}")
    print(f"  Metadata saved to: {metadata_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and set up DeepSpeak v2")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/deepspeak_v2",
        help="Output directory for extracted audio"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--split", "-s",
        default="train",
        choices=["train", "test"],
        help="Dataset split to use"
    )

    args = parser.parse_args()

    # Check for ffmpeg
    ffmpeg_path = get_ffmpeg_path()
    try:
        subprocess.run([ffmpeg_path, '-version'], capture_output=True, check=True)
        print(f"Using ffmpeg: {ffmpeg_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is required but not found. Please install it:")
        print("  brew install ffmpeg  (macOS)")
        print("  apt install ffmpeg   (Ubuntu)")
        print("  Or download to: bin/ffmpeg")
        sys.exit(1)

    setup_deepspeak_v2(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split=args.split,
    )


if __name__ == "__main__":
    main()
