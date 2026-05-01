#!/usr/bin/env python3
"""
Download and sample SpoofCeleb dataset parts incrementally.

Strategy:
1. Download one tar.gz part at a time
2. Extract a representative sample (stratified by speaker)
3. Delete the archive to save space
4. Track metadata for reproducibility
"""

import gzip
import json
import random
import subprocess
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SPOOFCELEB_DIR = PROJECT_ROOT / "data" / "spoofceleb"
AUDIO_DIR = SPOOFCELEB_DIR / "audio"
METADATA_FILE = SPOOFCELEB_DIR / "sample_metadata.json"

# Sampling parameters
MAX_FILES_PER_ATTACK = 5000  # Max files to keep per attack type
MAX_FILES_PER_SPEAKER = 50   # Max files per speaker (for diversity)
RANDOM_SEED = 42

# Part suffixes (aa, ab, ac, ... az, ba, bb, ...)
PART_SUFFIXES = [
    'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj',
    'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at',
    'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba'
]


def download_part(suffix: str) -> Path:
    """Download a single part of the SpoofCeleb archive."""
    filename = f"spoofceleb.tar.gz{suffix}"
    local_path = SPOOFCELEB_DIR / filename

    if local_path.exists():
        print(f"  {filename} already exists")
        return local_path

    print(f"  Downloading {filename}...")
    cmd = [
        str(PROJECT_ROOT / "audio_deepfake" / "bin" / "huggingface-cli"),
        "download", "--repo-type", "dataset",
        "jungjee/spoofceleb", filename,
        "--local-dir", str(SPOOFCELEB_DIR)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Download failed: {result.stderr}")
        return None

    return local_path


def list_archive_contents(part_path: Path) -> list:
    """List contents of a gzipped tar part (may be incomplete for split archives)."""
    contents = []
    try:
        with gzip.open(part_path, 'rb') as gz:
            # Read in chunks and parse tar headers
            with tarfile.open(fileobj=gz, mode='r|') as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith('.flac'):
                        contents.append(member.name)
    except Exception as e:
        print(f"  Warning: {e}")
    return contents


def extract_sample_from_part(part_path: Path, metadata: dict) -> int:
    """Extract a stratified sample from one archive part."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    extracted_count = 0

    # Track files by attack type and speaker
    files_by_attack = defaultdict(lambda: defaultdict(list))

    try:
        with gzip.open(part_path, 'rb') as gz:
            with tarfile.open(fileobj=gz, mode='r|') as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith('.flac'):
                        continue

                    # Parse path: spoofceleb/flac/{split}/{attack}/{speaker}/{file}.flac
                    parts = member.name.split('/')
                    if len(parts) < 6:
                        continue

                    split = parts[2]  # train, dev, eval
                    attack = parts[3]  # a00 (real), a01-a23 (fake)
                    speaker = parts[4]  # id10xxx
                    filename = parts[5]

                    # Check if we should sample this file
                    attack_key = f"{split}_{attack}"
                    current_attack_count = sum(len(files) for files in files_by_attack[attack_key].values())

                    if current_attack_count >= MAX_FILES_PER_ATTACK:
                        continue

                    if len(files_by_attack[attack_key][speaker]) >= MAX_FILES_PER_SPEAKER:
                        continue

                    # Extract file
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        # Determine output path
                        label = "real" if attack == "a00" else "fake"
                        out_dir = AUDIO_DIR / label / attack / speaker
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / filename

                        with open(out_path, 'wb') as out_f:
                            out_f.write(f.read())

                        files_by_attack[attack_key][speaker].append(filename)
                        extracted_count += 1

                        if extracted_count % 1000 == 0:
                            print(f"    Extracted {extracted_count} files...")

                    except Exception:
                        pass  # Skip problematic files

    except EOFError:
        print("  Reached end of archive part (expected for split archives)")
    except Exception as e:
        print(f"  Error reading archive: {e}")

    # Update metadata
    for attack_key, speakers in files_by_attack.items():
        if attack_key not in metadata['attacks']:
            metadata['attacks'][attack_key] = {'speakers': {}, 'total_files': 0}

        for speaker, files in speakers.items():
            if speaker not in metadata['attacks'][attack_key]['speakers']:
                metadata['attacks'][attack_key]['speakers'][speaker] = []
            metadata['attacks'][attack_key]['speakers'][speaker].extend(files)
            metadata['attacks'][attack_key]['total_files'] += len(files)

    return extracted_count


def process_part(suffix: str, metadata: dict) -> bool:
    """Download, sample, and cleanup one archive part."""
    print(f"\nProcessing part: spoofceleb.tar.gz{suffix}")

    # Download
    part_path = download_part(suffix)
    if part_path is None or not part_path.exists():
        print("  Skipping - download failed")
        return False

    # Extract sample
    count = extract_sample_from_part(part_path, metadata)
    print(f"  Extracted {count} files from this part")

    # Delete archive to save space
    print("  Deleting archive to save space...")
    part_path.unlink()

    # Update metadata
    metadata['parts_processed'].append(suffix)
    metadata['last_updated'] = datetime.now().isoformat()

    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sample SpoofCeleb dataset')
    parser.add_argument('--parts', type=int, default=3,
                        help='Number of parts to process (default: 3)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting part index (default: 0 = aa)')
    args = parser.parse_args()

    SPOOFCELEB_DIR.mkdir(parents=True, exist_ok=True)

    # Load or initialize metadata
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        print(f"Resuming from previous run. Parts processed: {metadata['parts_processed']}")
    else:
        metadata = {
            'dataset': 'SpoofCeleb',
            'source': 'jungjee/spoofceleb',
            'sampling_params': {
                'max_files_per_attack': MAX_FILES_PER_ATTACK,
                'max_files_per_speaker': MAX_FILES_PER_SPEAKER,
                'random_seed': RANDOM_SEED
            },
            'parts_processed': [],
            'attacks': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    # Process requested parts
    start_idx = args.start
    end_idx = min(start_idx + args.parts, len(PART_SUFFIXES))

    for i in range(start_idx, end_idx):
        suffix = PART_SUFFIXES[i]
        if suffix in metadata['parts_processed']:
            print(f"\nSkipping part {suffix} (already processed)")
            continue

        process_part(suffix, metadata)

    # Print summary
    print("\n" + "="*50)
    print("Sampling Summary")
    print("="*50)

    total_real = 0
    total_fake = 0

    for attack_key, data in sorted(metadata['attacks'].items()):
        count = data['total_files']
        n_speakers = len(data['speakers'])
        if 'a00' in attack_key:
            total_real += count
            label = "real"
        else:
            total_fake += count
            label = "fake"
        print(f"  {attack_key}: {count} files from {n_speakers} speakers ({label})")

    print(f"\nTotal: {total_real} real, {total_fake} fake")
    print(f"Parts processed: {metadata['parts_processed']}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
