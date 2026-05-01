"""Download SONICS dataset for AI-generated music detection.

SONICS (ICLR 2025) contains 97k+ songs (4,751 hours) including:
- 49k+ synthetic songs from Suno (v2, v3, v3.5) and Udio (v32, v130)
- 48k+ real songs (requires YouTube download)

Standard splits for benchmarking:
- Train: 66,709 songs
- Validation: 4,440 songs
- Test: 26,015 songs (13,237 real, 12,778 synthetic) <-- Use this for evaluation

Dataset: https://huggingface.co/datasets/awsaf49/sonics
Paper: https://arxiv.org/abs/2408.14080
GitHub: https://github.com/awsaf49/sonics

Usage:
    # Download only test split (recommended for evaluation)
    python scripts/download_sonics.py --test-only

    # Download full dataset
    python scripts/download_sonics.py --full
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HAS_HF = True
except ImportError:
    HAS_HF = False


def download_sonics_test_split(
    output_dir: str = "data/sonics",
    max_samples: int = None,
):
    """Download only the SONICS test split for evaluation.

    This is the standard benchmark split used in the ICLR 2025 paper:
    - 26,015 songs total
    - 13,237 real songs
    - 12,778 synthetic songs (Suno v2/v3/v3.5, Udio-32/130)

    Args:
        output_dir: Directory to save the dataset
        max_samples: Optional limit for testing (None = all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets soundfile")
        from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SONICS test split from HuggingFace...")
    print("This downloads only the test set (26,015 songs) for evaluation.")

    # Load only the test split
    try:
        dataset = load_dataset('awsaf49/sonics', split='test')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying to load with trust_remote_code=True...")
        dataset = load_dataset('awsaf49/sonics', split='test', trust_remote_code=True)

    print(f"\nTest set loaded: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    # Get label distribution
    if 'target' in dataset.column_names:
        labels = dataset['target']
        n_real = sum(1 for l in labels if l == 0)
        n_fake = sum(1 for l in labels if l == 1)
        print(f"Real: {n_real}, Fake: {n_fake}")

    # Limit samples if requested
    if max_samples and max_samples < len(dataset):
        print(f"\nLimiting to {max_samples} samples for testing...")
        # Balanced sampling
        real_indices = [i for i, t in enumerate(dataset['target']) if t == 0]
        fake_indices = [i for i, t in enumerate(dataset['target']) if t == 1]
        n_each = max_samples // 2
        selected = real_indices[:n_each] + fake_indices[:n_each]
        dataset = dataset.select(selected)
        print(f"Selected {len(dataset)} samples ({n_each} real, {n_each} fake)")

    # Check if audio is included
    if 'audio' in dataset.column_names:
        print("\nAudio data included! Saving to files...")
        import soundfile as sf
        from tqdm import tqdm

        # Create output directories
        real_dir = output_dir / 'test' / 'real'
        fake_dir = output_dir / 'test' / 'fake'
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        for i, sample in enumerate(tqdm(dataset, desc="Saving audio")):
            target = sample['target']
            audio = sample['audio']

            # Determine output path
            if target == 0:
                filename = f"real_{i:05d}.wav"
                out_path = real_dir / filename
            else:
                filename = f"fake_{i:05d}.wav"
                out_path = fake_dir / filename

            # Save audio
            sf.write(str(out_path), audio['array'], audio['sampling_rate'])

            # Collect metadata
            meta = {
                'filename': filename,
                'filepath': str(out_path.relative_to(output_dir)),
                'label': int(target),
                'source': sample.get('source', 'unknown'),
                'algorithm': sample.get('algorithm', 'unknown'),
                'duration': sample.get('duration', len(audio['array']) / audio['sampling_rate']),
                'genre': sample.get('genre', ''),
                'mood': sample.get('mood', ''),
            }
            metadata.append(meta)

        # Save metadata
        meta_path = output_dir / 'test_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        n_real = sum(1 for m in metadata if m['label'] == 0)
        n_fake = sum(1 for m in metadata if m['label'] == 1)
        print(f"\nSaved {len(metadata)} audio files:")
        print(f"  Real: {n_real} files in {real_dir}")
        print(f"  Fake: {n_fake} files in {fake_dir}")
        print(f"  Metadata: {meta_path}")

    else:
        print("\nAudio not directly included in this dataset version.")
        print("Saving metadata for reference...")

        # Save metadata
        meta_path = output_dir / 'test_metadata.json'
        metadata = []
        for sample in dataset:
            meta = {k: v for k, v in sample.items() if k != 'audio'}
            metadata.append(meta)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")
        print("\nTo get audio, you may need to:")
        print("1. Download fake_songs from HuggingFace: huggingface_hub.snapshot_download('awsaf49/sonics')")
        print("2. Download real songs from YouTube using the youtube_id field")

    return output_dir


def download_sonics_fake_songs(
    output_dir: str = "data/sonics",
    use_kaggle: bool = False,
):
    """Download SONICS synthetic/fake songs (full dataset).

    Args:
        output_dir: Directory to save the dataset
        use_kaggle: Use Kaggle API instead of HuggingFace
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_kaggle:
        print("Downloading SONICS from Kaggle...")
        print("Make sure kaggle API credentials are configured.")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "awsaf49/sonics-dataset",
            "--unzip",
            "-p", str(output_dir),
        ])
    else:
        if not HAS_HF:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub")
            print("Or use --kaggle flag to download via Kaggle API")
            return None

        print("Downloading SONICS from HuggingFace...")
        print("This is a large dataset (~150GB). This will take a while...")

        try:
            # Download the full dataset
            snapshot_download(
                repo_id="awsaf49/sonics",
                repo_type="dataset",
                local_dir=str(output_dir),
                resume_download=True,
            )
            print(f"Downloaded to {output_dir}")
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

    # Count files
    mp3_files = list(output_dir.rglob("*.mp3"))
    print("\nSONICS dataset (synthetic songs):")
    print(f"  Location: {output_dir}")
    print(f"  MP3 files: {len(mp3_files)}")

    return output_dir


def download_sonics_metadata(output_dir: str = "data/sonics"):
    """Download just the SONICS metadata (for real song YouTube IDs).

    Args:
        output_dir: Directory to save metadata
    """
    if not HAS_HF:
        print("huggingface_hub required. Install with: pip install huggingface_hub")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SONICS metadata...")

    # Download just CSV metadata files
    for filename in ["real_songs.csv", "fake_songs.csv"]:
        try:
            hf_hub_download(
                repo_id="awsaf49/sonics",
                repo_type="dataset",
                filename=filename,
                local_dir=str(output_dir),
            )
            print(f"  Downloaded {filename}")
        except Exception as e:
            print(f"  Could not download {filename}: {e}")

    return output_dir


def download_real_songs_from_youtube(
    metadata_path: str,
    output_dir: str = "data/sonics/real_songs",
    max_songs: int = None,
):
    """Download real songs from YouTube using SONICS metadata.

    Requires yt-dlp: pip install yt-dlp

    Args:
        metadata_path: Path to real_songs.csv from SONICS
        output_dir: Directory to save downloaded songs
        max_songs: Maximum number of songs to download (None = all)
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read YouTube IDs from metadata
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_songs:
        rows = rows[:max_songs]

    print(f"Downloading {len(rows)} songs from YouTube...")
    print("This requires yt-dlp: pip install yt-dlp")

    for i, row in enumerate(rows):
        youtube_id = row.get('youtube_id') or row.get('ytid')
        if not youtube_id:
            continue

        print(f"  [{i+1}/{len(rows)}] {youtube_id}")

        try:
            subprocess.run([
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "mp3",
                "--audio-quality", "0",  # Best quality
                "-o", str(output_dir / f"{youtube_id}.%(ext)s"),
                f"https://www.youtube.com/watch?v={youtube_id}",
            ], capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"    Error: {e}")
        except FileNotFoundError:
            print("    yt-dlp not found. Install with: pip install yt-dlp")
            break


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SONICS dataset for music deepfake evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download test split only (recommended for evaluation)
    python scripts/download_sonics.py --test-only

    # Download small sample for testing
    python scripts/download_sonics.py --test-only --max-samples 100

    # Download full dataset
    python scripts/download_sonics.py --full
        """
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/sonics",
        help="Output directory (default: data/sonics)"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Download only test split (26,015 songs) - recommended for evaluation"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum samples to download (for quick testing)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download full dataset (~150GB)"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only metadata CSV files"
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use Kaggle API instead of HuggingFace"
    )
    parser.add_argument(
        "--download-real",
        action="store_true",
        help="Download real songs from YouTube (requires yt-dlp)"
    )
    parser.add_argument(
        "--max-real-songs",
        type=int,
        default=None,
        help="Maximum number of real songs to download from YouTube"
    )

    args = parser.parse_args()

    if args.test_only:
        download_sonics_test_split(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
    elif args.metadata_only:
        download_sonics_metadata(args.output_dir)
    elif args.download_real:
        metadata_path = Path(args.output_dir) / "real_songs.csv"
        if not metadata_path.exists():
            print("Downloading metadata first...")
            download_sonics_metadata(args.output_dir)
        download_real_songs_from_youtube(
            str(metadata_path),
            str(Path(args.output_dir) / "real_songs"),
            max_songs=args.max_real_songs,
        )
    elif args.full:
        download_sonics_fake_songs(
            output_dir=args.output_dir,
            use_kaggle=args.kaggle,
        )
    else:
        print("Please specify --test-only (recommended) or --full")
        print("Use --help for more options")
        parser.print_help()


if __name__ == "__main__":
    main()
