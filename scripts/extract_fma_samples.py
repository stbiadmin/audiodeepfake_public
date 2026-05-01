#!/usr/bin/env python3
"""Extract sampled FMA tracks from the FMA medium zip file."""

import zipfile
from pathlib import Path

FMA_ZIP = Path("data/fma/fma_medium.zip")
SAMPLE_IDS_FILE = Path("data/fma/sampled_track_ids.txt")
OUTPUT_DIR = Path("data/fma/sampled_tracks")


def get_fma_path_from_id(track_id: int) -> str:
    """Convert track ID to FMA directory structure path.

    FMA uses 3-digit folder prefixes: track 123456 -> 123/123456.mp3
    """
    folder = str(track_id).zfill(6)[:3]
    return f"fma_medium/{folder}/{str(track_id).zfill(6)}.mp3"


def main():
    # Load sampled track IDs
    with open(SAMPLE_IDS_FILE) as f:
        track_ids = [int(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(track_ids)} track IDs to extract")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract tracks from zip
    extracted = 0
    missing = []

    with zipfile.ZipFile(FMA_ZIP, 'r') as zf:
        # Get list of files in zip
        zip_contents = set(zf.namelist())

        for track_id in track_ids:
            fma_path = get_fma_path_from_id(track_id)

            if fma_path in zip_contents:
                # Extract to output directory
                output_path = OUTPUT_DIR / f"{track_id}.mp3"

                with zf.open(fma_path) as src:
                    with open(output_path, 'wb') as dst:
                        dst.write(src.read())

                extracted += 1
                if extracted % 100 == 0:
                    print(f"Extracted {extracted}/{len(track_ids)} tracks...")
            else:
                missing.append(track_id)

    print("\n=== Extraction Complete ===")
    print(f"Extracted: {extracted} tracks")
    print(f"Missing: {len(missing)} tracks")

    if missing:
        print(f"\nFirst 10 missing track IDs: {missing[:10]}")
        # Save missing IDs for reference
        with open(OUTPUT_DIR.parent / "missing_track_ids.txt", 'w') as f:
            for tid in missing:
                f.write(f"{tid}\n")


if __name__ == "__main__":
    main()
