#!/usr/bin/env python3
"""Download MLAAD (fake) and M-AILABS (real) English datasets.

MLAAD: Multilingual Audio Anti-spoofing Dataset - synthetic speech from TTS models
M-AILABS: Real speech dataset used as source for MLAAD synthesis

This script downloads English-only subsets to use as training data.

Usage:
    # Download both MLAAD English (fake) and M-AILABS English (real)
    python scripts/download_mlaad.py

    # Download only MLAAD English
    python scripts/download_mlaad.py --mlaad-only

    # Download only M-AILABS English
    python scripts/download_mlaad.py --mailabs-only

    # Check status
    python scripts/download_mlaad.py --status

References:
    - MLAAD: https://huggingface.co/datasets/mueller91/MLAAD
    - M-AILABS: https://github.com/i-celeste-aurora/m-ailabs-dataset
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Get path to venv's tools
VENV_DIR = Path(__file__).parent.parent / 'audio_deepfake' / 'bin'
HF_CLI = str(VENV_DIR / 'huggingface-cli')

# M-AILABS download URLs (English only)
MAILABS_URLS = {
    'en_US': 'https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz',  # 102h, 7.5GB
    'en_UK': 'https://ics.tau-ceti.space/data/Training/stt_tts/en_UK.tgz',  # 45h, 3.5GB
}


def download_mlaad_english(output_dir: Path, verbose: bool = True) -> bool:
    """Download English-only subset of MLAAD using huggingface-cli.

    Args:
        output_dir: Directory to download into
        verbose: Print progress

    Returns:
        True if successful
    """
    output_dir = Path(output_dir)
    mlaad_dir = output_dir / 'MLAAD'
    mlaad_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading MLAAD English subset...")
        print(f"Output directory: {mlaad_dir}")
        print()

    # Use hf CLI with --include to filter for English only
    # MLAAD structure: fake/{language}/... where English is 'en'
    cmd = [
        HF_CLI, 'download',
        'mueller91/MLAAD',
        '--repo-type', 'dataset',
        '--local-dir', str(mlaad_dir),
        '--include', 'fake/en/*',  # English only
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")
        print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"ERROR: MLAAD download failed")
        return False

    if verbose:
        print(f"\n✓ MLAAD English download complete: {mlaad_dir}")

    return True


def download_mailabs_english(
    output_dir: Path,
    variants: list = ['en_US'],
    verbose: bool = True,
) -> bool:
    """Download M-AILABS English dataset(s).

    Args:
        output_dir: Directory to download into
        variants: Which English variants to download ('en_US', 'en_UK', or both)
        verbose: Print progress

    Returns:
        True if successful
    """
    output_dir = Path(output_dir)
    mailabs_dir = output_dir / 'M-AILABS'
    mailabs_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading M-AILABS English dataset(s): {variants}")
        print(f"Output directory: {mailabs_dir}")
        print()

    success = True
    for variant in variants:
        if variant not in MAILABS_URLS:
            print(f"WARNING: Unknown variant {variant}, skipping")
            continue

        url = MAILABS_URLS[variant]
        tgz_file = mailabs_dir / f'{variant}.tgz'
        extract_dir = mailabs_dir / variant

        # Skip if already extracted
        if extract_dir.exists() and any(extract_dir.rglob('*.wav')):
            if verbose:
                print(f"  {variant} already downloaded and extracted, skipping")
            continue

        # Download with wget (supports resume with -c)
        if verbose:
            print(f"  Downloading {variant} ({url})...")

        # Check if wget is available, fall back to curl
        wget_check = subprocess.run(['which', 'wget'], capture_output=True)
        if wget_check.returncode == 0:
            cmd = ['wget', '-c', '-O', str(tgz_file), url]
        else:
            cmd = ['curl', '-C', '-', '-L', '-o', str(tgz_file), url]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: Failed to download {variant}")
            success = False
            continue

        # Extract
        if verbose:
            print(f"  Extracting {variant}...")

        extract_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ['tar', '-xzf', str(tgz_file), '-C', str(extract_dir), '--strip-components=1'],
        )

        if result.returncode != 0:
            print(f"ERROR: Failed to extract {variant}")
            success = False
            continue

        # Optionally remove tgz to save space
        if verbose:
            print(f"  Cleaning up {tgz_file}...")
        tgz_file.unlink()

        if verbose:
            print(f"  ✓ {variant} complete")

    return success


def check_status(output_dir: Path):
    """Check download status."""
    output_dir = Path(output_dir)

    print(f"\nDataset Download Status")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")

    # Check MLAAD
    mlaad_dir = output_dir / 'MLAAD' / 'fake' / 'en'
    if mlaad_dir.exists():
        wav_count = len(list(mlaad_dir.rglob('*.wav')))
        models = [d.name for d in mlaad_dir.iterdir() if d.is_dir()]
        print(f"\nMLAAD English (fake):")
        print(f"  Directory: {mlaad_dir}")
        print(f"  WAV files: {wav_count}")
        print(f"  TTS models: {len(models)}")
        if models:
            print(f"  Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    else:
        print(f"\nMLAAD English: Not downloaded")

    # Check M-AILABS
    mailabs_dir = output_dir / 'M-AILABS'
    for variant in ['en_US', 'en_UK']:
        variant_dir = mailabs_dir / variant
        if variant_dir.exists():
            wav_count = len(list(variant_dir.rglob('*.wav')))
            print(f"\nM-AILABS {variant} (real):")
            print(f"  Directory: {variant_dir}")
            print(f"  WAV files: {wav_count}")
        else:
            print(f"\nM-AILABS {variant}: Not downloaded")


def main():
    parser = argparse.ArgumentParser(
        description='Download MLAAD and M-AILABS English datasets'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/experiments/mlaad_english'),
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--mlaad-only',
        action='store_true',
        help='Download only MLAAD (fake samples)'
    )
    parser.add_argument(
        '--mailabs-only',
        action='store_true',
        help='Download only M-AILABS (real samples)'
    )
    parser.add_argument(
        '--mailabs-variants',
        nargs='+',
        default=['en_US'],
        choices=['en_US', 'en_UK'],
        help='M-AILABS variants to download (default: en_US only)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check download status and exit'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output'
    )
    args = parser.parse_args()

    if args.status:
        check_status(args.output_dir)
        return

    verbose = not args.quiet
    success = True

    # Download MLAAD English (fake)
    if not args.mailabs_only:
        if verbose:
            print("="*50)
            print("Step 1: MLAAD English (fake samples)")
            print("="*50)
        if not download_mlaad_english(args.output_dir, verbose=verbose):
            success = False

    # Download M-AILABS English (real)
    if not args.mlaad_only:
        if verbose:
            print("\n" + "="*50)
            print("Step 2: M-AILABS English (real samples)")
            print("="*50)
        if not download_mailabs_english(
            args.output_dir,
            variants=args.mailabs_variants,
            verbose=verbose,
        ):
            success = False

    # Final status
    if verbose:
        print("\n" + "="*50)
        print("Download Summary")
        print("="*50)
        check_status(args.output_dir)

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
