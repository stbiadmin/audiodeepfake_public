"""Download FakeMusicCaps dataset for AI-generated music detection.

FakeMusicCaps contains ~27,605 AI-generated music tracks (~77 hours)
generated using 5 text-to-music models:
- MusicGen
- MusicLDM
- AudioLDM2
- Stable Audio Open
- Mustango

All audio: 16kHz mono WAV format.

Dataset: https://zenodo.org/records/13732524
Paper: https://arxiv.org/abs/2409.10684
GitHub: https://github.com/polimi-ispl/FakeMusicCaps
"""

import os
import sys
import zipfile
from pathlib import Path
import urllib.request


ZENODO_URL = "https://zenodo.org/records/13732524/files/FakeMusicCaps.zip?download=1"
EXPECTED_SIZE_GB = 4.5  # Approximate


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


def setup_fakemusiccaps(
    output_dir: str = "data/fakemusiccaps",
    download: bool = True,
):
    """Download and set up FakeMusicCaps dataset.

    Args:
        output_dir: Directory to save the dataset
        download: Whether to download (False to just organize existing files)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "FakeMusicCaps.zip"

    # Download if needed
    if download and not zip_path.exists():
        print(f"Downloading FakeMusicCaps (~{EXPECTED_SIZE_GB} GB)...")
        print("This may take a while...")
        download_with_progress(ZENODO_URL, str(zip_path), "Downloading FakeMusicCaps")
    elif zip_path.exists():
        print(f"Found existing {zip_path}")

    # Check if already extracted
    extracted_marker = output_dir / ".extracted"
    if extracted_marker.exists():
        print("Already extracted")
    elif zip_path.exists():
        # Extract
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        extracted_marker.touch()
        print("Extraction complete.")

    # Count and organize files
    wav_files = list(output_dir.rglob("*.wav"))
    print(f"\nFakeMusicCaps dataset:")
    print(f"  Location: {output_dir}")
    print(f"  Audio files: {len(wav_files)}")

    # List generators if subdirectories exist
    generators = set()
    for f in wav_files:
        # Attempt to identify generator from path
        parts = f.relative_to(output_dir).parts
        if len(parts) > 1:
            generators.add(parts[0])

    if generators:
        print(f"  Generators found: {sorted(generators)}")

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download FakeMusicCaps dataset")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/fakemusiccaps",
        help="Output directory"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download, just organize existing files"
    )

    args = parser.parse_args()

    setup_fakemusiccaps(
        output_dir=args.output_dir,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
