#!/usr/bin/env python3
"""
Extract audio from FakeAVCeleb videos for audio deepfake detection training.

Real audio: RealVideo-RealAudio
Fake audio: RealVideo-FakeAudio, FakeVideo-FakeAudio
"""

import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FAKEAVCELEB_DIR = PROJECT_ROOT / "data" / "fakeavceleb" / "FakeAVCeleb_v1.2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "fakeavceleb" / "audio"
FFMPEG = PROJECT_ROOT / "bin" / "ffmpeg"

# Categories to extract
REAL_DIRS = ["RealVideo-RealAudio"]
FAKE_DIRS = ["RealVideo-FakeAudio", "FakeVideo-FakeAudio"]


def extract_audio(video_path: Path, output_path: Path) -> bool:
    """Extract audio from video using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return True

    cmd = [
        str(FFMPEG),
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error extracting {video_path}: {e}")
        return False


def get_video_files(category_dirs: list, label: str) -> list:
    """Get all video files from specified category directories."""
    videos = []
    for cat_dir in category_dirs:
        cat_path = FAKEAVCELEB_DIR / cat_dir
        if cat_path.exists():
            for video in cat_path.rglob("*.mp4"):
                # Create output path preserving structure
                rel_path = video.relative_to(FAKEAVCELEB_DIR)
                output_path = OUTPUT_DIR / label / rel_path.with_suffix(".wav")
                videos.append((video, output_path))
    return videos


def main():
    print("FakeAVCeleb Audio Extraction")
    print("=" * 50)

    # Get all videos
    real_videos = get_video_files(REAL_DIRS, "real")
    fake_videos = get_video_files(FAKE_DIRS, "fake")

    print(f"Real videos to process: {len(real_videos)}")
    print(f"Fake videos to process: {len(fake_videos)}")

    all_videos = real_videos + fake_videos

    # Check how many already exist
    existing = sum(1 for _, out in all_videos if out.exists())
    print(f"Already extracted: {existing}")
    print(f"Remaining: {len(all_videos) - existing}")

    if existing == len(all_videos):
        print("All audio already extracted!")
        return

    # Extract with parallel processing
    num_workers = 4  # Conservative for M1

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(extract_audio, video, output): (video, output)
            for video, output in all_videos
            if not output.exists()
        }

        with tqdm(total=len(futures), desc="Extracting audio") as pbar:
            for future in as_completed(futures):
                video, output = futures[future]
                try:
                    if future.result():
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"\nError: {video}: {e}")
                    failed += 1
                pbar.update(1)

    print(f"\nExtraction complete!")
    print(f"Success: {success + existing}")
    print(f"Failed: {failed}")

    # Verify counts
    real_count = len(list((OUTPUT_DIR / "real").rglob("*.wav"))) if (OUTPUT_DIR / "real").exists() else 0
    fake_count = len(list((OUTPUT_DIR / "fake").rglob("*.wav"))) if (OUTPUT_DIR / "fake").exists() else 0
    print(f"\nFinal counts:")
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")


if __name__ == "__main__":
    main()
