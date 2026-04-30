#!/usr/bin/env python3
"""Download SONICS real songs with stratified sampling by year.

Ensures diversity by sampling proportionally across years, with fallback
to next candidate when a video is unavailable.

Usage:
    python scripts/download_sonics_real_stratified.py
    python scripts/download_sonics_real_stratified.py --target 500  # Smaller sample
"""

import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import random

import pandas as pd
from tqdm import tqdm


def check_video_available(yt_id: str, yt_dlp: str) -> bool:
    """Quick check if video is available without downloading."""
    try:
        result = subprocess.run(
            [yt_dlp, '--simulate', '--quiet', '--no-warnings',
             f'https://www.youtube.com/watch?v={yt_id}'],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except:
        return False


def download_video(yt_id: str, output_dir: Path, yt_dlp: str) -> bool:
    """Download a single video as MP3."""
    output_path = output_dir / f'{yt_id}.mp3'
    if output_path.exists():
        return True

    try:
        result = subprocess.run([
            yt_dlp,
            '-x', '--audio-format', 'mp3',
            '--audio-quality', '192K',
            '-o', str(output_dir / f'{yt_id}.%(ext)s'),
            '--no-playlist', '--quiet', '--no-warnings',
            f'https://www.youtube.com/watch?v={yt_id}'
        ], capture_output=True, timeout=120)
        return result.returncode == 0 and output_path.exists()
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download SONICS real songs with stratified sampling'
    )
    parser.add_argument('--target', '-n', type=int, default=1000,
                        help='Target number of songs to download')
    parser.add_argument('--output-dir', '-o', type=Path,
                        default=Path('data/sonics/real'),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yt_dlp = 'yt-dlp'

    # Load test set
    test_df = pd.read_csv('data/sonics/test.csv')
    real_songs = test_df[test_df['target'] == 0].copy()

    # Check existing downloads
    existing = set(f.stem for f in args.output_dir.glob('*.mp3'))
    print(f'Already downloaded: {len(existing)}')

    if len(existing) >= args.target:
        print(f'Already have {len(existing)} >= {args.target} target. Done!')
        return

    remaining = args.target - len(existing)
    print(f'Need to download: {remaining} more songs')

    # Group by year and shuffle within each year
    year_groups = defaultdict(list)
    for _, row in real_songs.iterrows():
        yt_id = row['youtube_id']
        if pd.notna(yt_id) and yt_id not in existing:
            year = row['year'] if pd.notna(row['year']) else 0
            year_groups[year].append(yt_id)

    # Shuffle within each year
    for year in year_groups:
        random.shuffle(year_groups[year])

    # Calculate proportional targets per year
    total_available = sum(len(ids) for ids in year_groups.values())
    year_targets = {}
    for year, ids in year_groups.items():
        proportion = len(ids) / total_available
        year_targets[year] = max(1, int(proportion * remaining))

    # Adjust to hit exact target
    total_target = sum(year_targets.values())
    if total_target < remaining:
        # Add to largest years
        for year in sorted(year_groups.keys(), key=lambda y: len(year_groups[y]), reverse=True):
            if total_target >= remaining:
                break
            year_targets[year] += 1
            total_target += 1

    print(f'\nStratified sampling across {len(year_groups)} years')
    print(f'Top years: {sorted(year_targets.items(), key=lambda x: -x[1])[:5]}')

    # Download with stratified sampling
    downloaded = len(existing)
    failed = 0
    year_downloaded = defaultdict(int)
    year_index = {year: 0 for year in year_groups}  # Track position in each year's list

    # Create a priority queue of years that still need downloads
    active_years = [y for y in year_groups if year_targets.get(y, 0) > 0]

    with tqdm(total=args.target, initial=downloaded, desc='Downloading') as pbar:
        while downloaded < args.target and active_years:
            # Round-robin through years for diversity
            for year in list(active_years):
                if downloaded >= args.target:
                    break

                # Check if this year still needs downloads and has candidates
                if year_downloaded[year] >= year_targets.get(year, 0):
                    active_years.remove(year)
                    continue

                ids = year_groups[year]
                idx = year_index[year]

                # Try candidates from this year
                while idx < len(ids) and year_downloaded[year] < year_targets[year]:
                    yt_id = ids[idx]
                    idx += 1
                    year_index[year] = idx

                    if download_video(yt_id, args.output_dir, yt_dlp):
                        downloaded += 1
                        year_downloaded[year] += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'ok': downloaded,
                            'fail': failed,
                            'year': int(year) if year else 'unk'
                        })
                        break  # Move to next year for diversity
                    else:
                        failed += 1

                # If year exhausted, remove from active
                if idx >= len(ids):
                    if year in active_years:
                        active_years.remove(year)
                        # Redistribute remaining quota to other years
                        shortfall = year_targets[year] - year_downloaded[year]
                        if shortfall > 0 and active_years:
                            for other_year in active_years:
                                year_targets[other_year] += shortfall // len(active_years)
                            print(f'\nYear {year} exhausted, redistributing {shortfall} to other years')

    print(f'\n=== Final Results ===')
    print(f'Downloaded: {downloaded}')
    print(f'Failed: {failed}')
    print(f'Success rate: {downloaded/(downloaded+failed)*100:.1f}%')

    # Save metadata for downloaded songs
    downloaded_ids = set(f.stem for f in args.output_dir.glob('*.mp3'))
    downloaded_df = real_songs[real_songs['youtube_id'].isin(downloaded_ids)]
    downloaded_df.to_csv(args.output_dir.parent / 'real_downloaded_metadata.csv', index=False)
    print(f'\nMetadata saved to {args.output_dir.parent / "real_downloaded_metadata.csv"}')

    # Show year distribution of downloads
    print(f'\nYear distribution of downloads:')
    print(downloaded_df['year'].value_counts().head(10))


if __name__ == '__main__':
    main()
