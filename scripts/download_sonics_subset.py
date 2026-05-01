#!/usr/bin/env python3
"""Download SONICS 2k subset for music deepfake evaluation.

Downloads a stratified sample of 2,000 songs:
- 1,000 fake songs (proportional by algorithm: Suno v2/v3/v3.5, Udio-30s/120s)
- 1,000 real songs (from YouTube)

Usage:
    python scripts/download_sonics_subset.py
    python scripts/download_sonics_subset.py --fake-only  # Skip YouTube downloads
    python scripts/download_sonics_subset.py --max-real 100  # Limit real downloads
"""

import argparse
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import hf_hub_download


def download_fake_songs(sample_df: pd.DataFrame, output_dir: Path) -> int:
    """Download fake songs from SONICS zip files.

    Args:
        sample_df: DataFrame with fake song metadata
        output_dir: Directory to save songs

    Returns:
        Number of songs downloaded
    """
    fake_dir = output_dir / 'fake'
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Get list of files we need
    fake_files = set(sample_df[sample_df['target'] == 1]['filepath'].tolist())
    # Extract just the filename part
    fake_filenames = {Path(fp).name for fp in fake_files}

    print(f"\nNeed to download {len(fake_filenames)} fake songs")
    print("Downloading zip files from HuggingFace (this may take a while)...")

    downloaded = 0
    temp_dir = output_dir / 'temp_zips'
    temp_dir.mkdir(exist_ok=True)

    # Download and extract from each zip file
    for i in range(1, 11):
        zip_name = f"fake_songs/part_{i:02d}.zip"
        print(f"\nProcessing {zip_name}...")

        try:
            # Download zip file
            zip_path = hf_hub_download(
                repo_id='awsaf49/sonics',
                repo_type='dataset',
                filename=zip_name,
                local_dir=str(temp_dir),
            )

            # Extract only the files we need
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # List files in zip
                zip_contents = zf.namelist()

                # Find matching files
                for name in zip_contents:
                    base_name = Path(name).name
                    if base_name in fake_filenames:
                        # Extract this file
                        zf.extract(name, temp_dir)
                        # Move to output directory
                        src = temp_dir / name
                        dst = fake_dir / base_name
                        if src.exists():
                            src.rename(dst)
                            downloaded += 1
                            fake_filenames.discard(base_name)

                print(f"  Extracted {downloaded} files so far, {len(fake_filenames)} remaining")

            # Remove zip to save space
            os.remove(zip_path)

            # If we have all files, stop early
            if not fake_filenames:
                print("All fake files downloaded!")
                break

        except Exception as e:
            print(f"  Error processing {zip_name}: {e}")
            continue

    # Cleanup temp directory
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    return downloaded


def download_real_songs(
    sample_df: pd.DataFrame,
    output_dir: Path,
    max_songs: int = None,
    workers: int = 4,
) -> int:
    """Download real songs from YouTube.

    Args:
        sample_df: DataFrame with real song metadata
        output_dir: Directory to save songs
        max_songs: Maximum number of songs to download
        workers: Number of parallel downloads

    Returns:
        Number of songs downloaded
    """
    real_dir = output_dir / 'real'
    real_dir.mkdir(parents=True, exist_ok=True)

    # Get YouTube IDs
    real_songs = sample_df[sample_df['target'] == 0].copy()
    youtube_ids = real_songs['youtube_id'].dropna().tolist()

    if max_songs:
        youtube_ids = youtube_ids[:max_songs]

    print(f"\nDownloading {len(youtube_ids)} real songs from YouTube...")
    print("This requires yt-dlp: pip install yt-dlp")

    # Check if yt-dlp is available
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp not found. Installing...")
        os.system(f"{sys.executable} -m pip install yt-dlp")

    downloaded = 0
    failed = 0

    def download_one(yt_id: str) -> bool:
        """Download a single YouTube video as audio."""
        output_path = real_dir / f"{yt_id}.mp3"
        if output_path.exists():
            return True

        try:
            result = subprocess.run([
                'yt-dlp',
                '-x',  # Extract audio
                '--audio-format', 'mp3',
                '--audio-quality', '192K',
                '-o', str(real_dir / f"{yt_id}.%(ext)s"),
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                f"https://www.youtube.com/watch?v={yt_id}",
            ], capture_output=True, timeout=120)
            return result.returncode == 0
        except Exception:
            return False

    # Download with progress bar
    with tqdm(total=len(youtube_ids), desc="Downloading real songs") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_one, yt_id): yt_id for yt_id in youtube_ids}
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({'ok': downloaded, 'fail': failed})

    print(f"Downloaded: {downloaded}, Failed: {failed}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description='Download SONICS 2k subset for music deepfake evaluation'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/sonics'),
        help='Output directory'
    )
    parser.add_argument(
        '--sample-csv',
        type=Path,
        default=Path('data/sonics/test_sample_2k.csv'),
        help='Path to sample CSV file'
    )
    parser.add_argument(
        '--fake-only',
        action='store_true',
        help='Only download fake songs (skip YouTube)'
    )
    parser.add_argument(
        '--real-only',
        action='store_true',
        help='Only download real songs from YouTube'
    )
    parser.add_argument(
        '--max-real', '-n',
        type=int,
        default=None,
        help='Maximum number of real songs to download'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel YouTube downloads'
    )
    args = parser.parse_args()

    # Load sample CSV
    if not args.sample_csv.exists():
        print(f"Sample CSV not found: {args.sample_csv}")
        print("Run the sample creation script first.")
        return

    sample_df = pd.read_csv(args.sample_csv)
    print(f"Loaded {len(sample_df)} samples from {args.sample_csv}")
    print(f"  Fake: {(sample_df['target'] == 1).sum()}")
    print(f"  Real: {(sample_df['target'] == 0).sum()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download fake songs
    if not args.real_only:
        n_fake = download_fake_songs(sample_df, args.output_dir)
        print(f"\nFake songs downloaded: {n_fake}")

    # Download real songs
    if not args.fake_only:
        n_real = download_real_songs(
            sample_df,
            args.output_dir,
            max_songs=args.max_real,
            workers=args.workers,
        )
        print(f"\nReal songs downloaded: {n_real}")

    # Summary
    fake_dir = args.output_dir / 'fake'
    real_dir = args.output_dir / 'real'
    n_fake_files = len(list(fake_dir.glob('*.mp3'))) if fake_dir.exists() else 0
    n_real_files = len(list(real_dir.glob('*.mp3'))) if real_dir.exists() else 0

    print("\n=== Download Summary ===")
    print(f"Fake songs: {n_fake_files} files in {fake_dir}")
    print(f"Real songs: {n_real_files} files in {real_dir}")
    print(f"Total: {n_fake_files + n_real_files} files")


if __name__ == '__main__':
    main()
