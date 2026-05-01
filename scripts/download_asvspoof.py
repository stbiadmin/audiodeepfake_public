"""Download and set up ASVspoof 2019 LA dataset for audio deepfake detection.

ASVspoof 2019 LA (Logical Access) is the standard benchmark for audio
spoofing/deepfake detection. It contains bonafide (real) and spoofed (fake)
audio samples.

Dataset info: https://www.asvspoof.org/index2019.html
Download: https://datashare.ed.ac.uk/handle/10283/3336
"""

import json
import sys
import urllib.request
import zipfile
from pathlib import Path


def download_with_progress(url: str, output_path: str, description: str = "Downloading"):
    """Download a file with progress indication.

    Args:
        url: URL to download
        output_path: Path to save the file
        description: Description for progress output
    """
    print(f"{description}: {url}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        downloaded = count * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {percent}% ({downloaded:.1f}/{total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print()  # Newline after progress


def parse_protocol_file(protocol_path: str) -> dict:
    """Parse ASVspoof protocol file.

    Protocol format: SPEAKER_ID AUDIO_FILE - SYSTEM_ID LABEL
    Example: LA_0079 LA_E_2834763 - A13 spoof

    Args:
        protocol_path: Path to protocol file

    Returns:
        Dict mapping filename to metadata
    """
    metadata = {}

    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id = parts[0]
                audio_file = parts[1]
                system_id = parts[3]
                label = parts[4]  # 'bonafide' or 'spoof'

                metadata[audio_file] = {
                    'speaker_id': speaker_id,
                    'system_id': system_id,
                    'label': label,
                    'is_real': label == 'bonafide',
                }

    return metadata


def setup_asvspoof_2019(
    output_dir: str = "data/asvspoof2019",
    download: bool = True,
    subset: str = "LA",
):
    """Download and set up ASVspoof 2019 dataset.

    Args:
        output_dir: Directory to save the dataset
        download: Whether to download (False to just organize existing files)
        subset: Dataset subset ("LA" for Logical Access)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / f"{subset}.zip"
    extract_dir = output_dir / subset

    # Download if needed
    if download and not extract_dir.exists():
        url = f"https://datashare.ed.ac.uk/bitstream/handle/10283/3336/{subset}.zip?sequence=3&isAllowed=y"

        if not zip_path.exists():
            print(f"Downloading ASVspoof 2019 {subset} (~7.1 GB)...")
            print("This may take a while...")
            download_with_progress(url, str(zip_path), f"Downloading {subset}.zip")
        else:
            print(f"Found existing {zip_path}")

        # Extract
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print("Extraction complete.")

    # Parse protocol files and organize
    print("Organizing dataset...")

    # Find the actual LA directory (might be nested)
    la_dir = None
    for candidate in [extract_dir, output_dir / "LA", output_dir / "ASVspoof2019_LA"]:
        if candidate.exists():
            la_dir = candidate
            break

    if la_dir is None:
        print(f"Error: Could not find extracted LA directory in {output_dir}")
        return

    # Look for protocol files
    protocol_dir = None
    for candidate in [
        la_dir / "ASVspoof2019_LA_cm_protocols",
        la_dir / "ASVspoof2019.LA.cm.train.trn.txt",
    ]:
        if candidate.exists():
            protocol_dir = candidate.parent if candidate.is_file() else candidate
            break

    if protocol_dir is None:
        # Try finding protocol files
        protocol_files = list(la_dir.rglob("*.txt"))
        if protocol_files:
            protocol_dir = protocol_files[0].parent
            print(f"Found protocol files in: {protocol_dir}")

    # Find audio directories
    audio_dirs = {
        'train': None,
        'dev': None,
        'eval': None,
    }

    for split in ['train', 'dev', 'eval']:
        for candidate in [
            la_dir / f"ASVspoof2019_LA_{split}" / "flac",
            la_dir / f"ASVspoof2019_LA_{split}",
            la_dir / split / "flac",
            la_dir / split,
        ]:
            if candidate.exists() and any(candidate.glob("*.flac")):
                audio_dirs[split] = candidate
                break

    # Print what we found
    print(f"\nDataset structure in {la_dir}:")
    for split, path in audio_dirs.items():
        if path:
            count = len(list(path.glob("*.flac")))
            print(f"  {split}: {path} ({count} files)")
        else:
            print(f"  {split}: not found")

    # Create organized structure with symlinks
    organized_dir = output_dir / "organized"
    real_dir = organized_dir / "bonafide"
    fake_dir = organized_dir / "spoof"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Parse all protocol files if available
    all_metadata = {}
    if protocol_dir:
        for protocol_file in protocol_dir.glob("*.txt"):
            print(f"Parsing {protocol_file.name}...")
            metadata = parse_protocol_file(str(protocol_file))
            all_metadata.update(metadata)
        print(f"Total entries in protocols: {len(all_metadata)}")

    # Count files by label
    real_count = sum(1 for m in all_metadata.values() if m['is_real'])
    fake_count = sum(1 for m in all_metadata.values() if not m['is_real'])
    print(f"  Bonafide (real): {real_count}")
    print(f"  Spoof (fake): {fake_count}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'dataset': 'ASVspoof2019_LA',
            'audio_dirs': {k: str(v) if v else None for k, v in audio_dirs.items()},
            'protocol_dir': str(protocol_dir) if protocol_dir else None,
            'total_files': len(all_metadata),
            'bonafide_count': real_count,
            'spoof_count': fake_count,
            'files': all_metadata,
        }, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print("\nTo use this dataset for feature extraction:")
    print("  Real audio: Look for files with 'bonafide' label in metadata")
    print("  Fake audio: Look for files with 'spoof' label in metadata")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and set up ASVspoof 2019 LA")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/asvspoof2019",
        help="Output directory"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download, just organize existing files"
    )

    args = parser.parse_args()

    setup_asvspoof_2019(
        output_dir=args.output_dir,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
