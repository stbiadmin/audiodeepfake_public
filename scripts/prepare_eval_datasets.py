#!/usr/bin/env python3
"""Prepare evaluation datasets by organizing files by label using symlinks.

Creates organized directory structure for extraction:
    data/eval_datasets/{dataset}/organized/
        real/  -> symlinks to bonafide/real files
        fake/  -> symlinks to spoof/fake files
"""

import csv
import os
import sys
from pathlib import Path


def prepare_asvspoof5(base_dir: Path):
    """Organize ASVspoof5 eval files by label."""
    tsv_path = base_dir / "ASVspoof5.eval.track_1.tsv"
    audio_dir = base_dir / "flac_E_eval"

    if not tsv_path.exists():
        print(f"ERROR: TSV not found: {tsv_path}")
        return False

    if not audio_dir.exists():
        print(f"ERROR: Audio dir not found: {audio_dir}")
        return False

    # Create organized directories
    real_dir = base_dir / "organized" / "real"
    fake_dir = base_dir / "organized" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Parse TSV and create symlinks
    real_count = 0
    fake_count = 0
    missing_count = 0
    total_lines = 0

    with open(tsv_path, 'r') as f:
        for line in f:
            total_lines += 1
            # TSV is actually space-separated
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            # Column 1 is utterance ID (filename without .flac)
            utterance_id = parts[1]
            # Column 8 is label (bonafide or spoof)
            label = parts[8]

            src_file = audio_dir / f"{utterance_id}.flac"
            if not src_file.exists():
                missing_count += 1
                continue

            if label == "bonafide":
                dst = real_dir / f"{utterance_id}.flac"
                real_count += 1
            else:  # spoof
                dst = fake_dir / f"{utterance_id}.flac"
                fake_count += 1

            if not dst.exists():
                os.symlink(src_file.resolve(), dst)

            # Progress indicator
            if (real_count + fake_count) % 50000 == 0:
                print(f"  Progress: {real_count + fake_count} files processed...")

    print(f"ASVspoof5: {real_count} real, {fake_count} fake ({missing_count} missing files)")
    return True


def prepare_in_the_wild(base_dir: Path):
    """Organize In-the-Wild files by label."""
    release_dir = base_dir / "release_in_the_wild"
    meta_path = release_dir / "meta.csv"

    if not meta_path.exists():
        print(f"ERROR: meta.csv not found: {meta_path}")
        return False

    if not release_dir.exists():
        print(f"ERROR: release_in_the_wild dir not found: {release_dir}")
        return False

    # Create organized directories
    real_dir = base_dir / "organized" / "real"
    fake_dir = base_dir / "organized" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_count = 0
    fake_count = 0
    missing_count = 0

    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['file']
            label = row['label']

            src_file = release_dir / filename
            if not src_file.exists():
                missing_count += 1
                continue

            if label == "bona-fide":
                dst = real_dir / filename
                real_count += 1
            else:  # spoof
                dst = fake_dir / filename
                fake_count += 1

            if not dst.exists():
                os.symlink(src_file.resolve(), dst)

            # Progress indicator
            if (real_count + fake_count) % 10000 == 0:
                print(f"  Progress: {real_count + fake_count} files processed...")

    print(f"In-the-Wild: {real_count} real, {fake_count} fake ({missing_count} missing files)")
    return True


def prepare_sonics(base_dir: Path):
    """Organize SONICS - all files are fake AI-generated music."""
    # Handle nested structure (fake_songs/fake_songs/)
    fake_songs_dir = base_dir / "fake_songs" / "fake_songs"
    if not fake_songs_dir.exists():
        fake_songs_dir = base_dir / "fake_songs"

    if not fake_songs_dir.exists():
        print(f"ERROR: fake_songs not found in {base_dir}")
        return False

    # Check if there are actually mp3 files
    mp3_files = list(fake_songs_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"ERROR: No MP3 files found in {fake_songs_dir}")
        return False

    # Create organized directory (only fake for SONICS)
    fake_dir = base_dir / "organized" / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)

    fake_count = 0
    for src_file in mp3_files:
        dst = fake_dir / src_file.name
        if not dst.exists():
            os.symlink(src_file.resolve(), dst)
        fake_count += 1

        # Progress indicator
        if fake_count % 10000 == 0:
            print(f"  Progress: {fake_count} files processed...")

    print(f"SONICS: 0 real, {fake_count} fake (AI-generated music only)")
    return True


def main():
    eval_datasets_dir = Path("data/eval_datasets")

    print("Preparing evaluation datasets...\n")

    # ASVspoof5
    asvspoof5_dir = eval_datasets_dir / "asvspoof5"
    if asvspoof5_dir.exists():
        print("Processing ASVspoof5...")
        prepare_asvspoof5(asvspoof5_dir)

    # In-the-Wild
    in_the_wild_dir = eval_datasets_dir / "in_the_wild"
    if in_the_wild_dir.exists():
        print("\nProcessing In-the-Wild...")
        prepare_in_the_wild(in_the_wild_dir)

    # SONICS
    sonics_dir = eval_datasets_dir / "sonics"
    if sonics_dir.exists():
        print("\nProcessing SONICS...")
        prepare_sonics(sonics_dir)

    print("\nDone! Organized directories created with symlinks.")
    print("\nNext step: Run feature extraction on each organized directory.")


if __name__ == "__main__":
    main()
