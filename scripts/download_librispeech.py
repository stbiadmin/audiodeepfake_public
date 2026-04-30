"""Download LibriSpeech subsets for real audio samples.

LibriSpeech is a corpus of read English speech, perfect for
providing ground truth "real" audio samples.

Available subsets:
- dev-clean: ~330 MB (development, clean speech)
- test-clean: ~350 MB (test, clean speech)
- train-clean-100: ~6.3 GB (100 hours clean training)
"""

import os
import sys
import tarfile
from pathlib import Path
import urllib.request


LIBRISPEECH_URLS = {
    'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
    'test-clean': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
    'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
}


def download_with_progress(url: str, output_path: str, description: str = "Downloading"):
    """Download a file with progress indication."""
    print(f"{description}: {url}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        downloaded = count * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {percent}% ({downloaded:.1f}/{total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print()


def setup_librispeech(
    output_dir: str = "data/librispeech",
    subset: str = "dev-clean",
):
    """Download and set up LibriSpeech subset.

    Args:
        output_dir: Directory to save the dataset
        subset: Which subset to download
    """
    if subset not in LIBRISPEECH_URLS:
        print(f"Unknown subset: {subset}")
        print(f"Available: {list(LIBRISPEECH_URLS.keys())}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = LIBRISPEECH_URLS[subset]
    tar_path = output_dir / f"{subset}.tar.gz"

    # Download
    if not tar_path.exists():
        print(f"Downloading LibriSpeech {subset}...")
        download_with_progress(url, str(tar_path), f"Downloading {subset}")
    else:
        print(f"Found existing {tar_path}")

    # Extract
    extract_dir = output_dir / "LibriSpeech" / subset
    if not extract_dir.exists():
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall(output_dir)
        print("Extraction complete.")
    else:
        print(f"Already extracted to {extract_dir}")

    # Count files
    flac_files = list(extract_dir.rglob("*.flac"))
    print(f"\nLibriSpeech {subset}:")
    print(f"  Location: {extract_dir}")
    print(f"  Audio files: {len(flac_files)}")

    # List speakers
    speakers = set()
    for f in flac_files:
        # Path structure: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
        speaker_id = f.parts[-3]
        speakers.add(speaker_id)
    print(f"  Speakers: {len(speakers)}")

    return extract_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download LibriSpeech subset")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/librispeech",
        help="Output directory"
    )
    parser.add_argument(
        "--subset", "-s",
        default="dev-clean",
        choices=list(LIBRISPEECH_URLS.keys()),
        help="Which subset to download"
    )

    args = parser.parse_args()

    setup_librispeech(
        output_dir=args.output_dir,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
