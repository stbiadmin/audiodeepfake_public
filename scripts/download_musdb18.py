"""Download MUSDB18 dataset for real music with stems.

MUSDB18 contains 150 full-length music tracks (~10 hours) with isolated stems:
- Mixture (full song)
- Drums
- Bass
- Vocals
- Other accompaniment

All audio: 44.1kHz stereo, encoded in STEMS format (.mp4)

IMPORTANT: MUSDB18 requires academic access request on Zenodo.
Visit https://zenodo.org/records/1117372 to request access.

Dataset: https://zenodo.org/records/1117372
Paper: https://sigsep.github.io/datasets/musdb.html
GitHub: https://github.com/sigsep/sigsep-mus-db

Alternative: Use the musdb Python package which can download automatically:
    pip install musdb
    import musdb
    mus = musdb.DB(download=True)
"""

from pathlib import Path

try:
    import musdb
    HAS_MUSDB = True
except ImportError:
    HAS_MUSDB = False


def setup_musdb18_via_package(
    output_dir: str = "data/musdb18",
):
    """Set up MUSDB18 using the musdb Python package.

    This method automatically handles downloading and setup.

    Args:
        output_dir: Directory to save the dataset
    """
    if not HAS_MUSDB:
        print("musdb package not installed. Install with: pip install musdb")
        print("Note: musdb requires ffmpeg for decoding.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up MUSDB18 via musdb package...")
    print("This will download the dataset if not present.")

    try:
        # Initialize with download=True to auto-download
        mus = musdb.DB(root=str(output_dir), download=True)

        print("\nMUSDB18 dataset:")
        print(f"  Location: {output_dir}")
        print(f"  Training tracks: {len(mus.load_mus_tracks(subsets='train'))}")
        print(f"  Test tracks: {len(mus.load_mus_tracks(subsets='test'))}")

        # List available stems
        print("  Available stems: mixture, drums, bass, vocals, other")

        return output_dir

    except Exception as e:
        print(f"Error setting up MUSDB18: {e}")
        print("\nManual setup instructions:")
        print("1. Visit https://zenodo.org/records/1117372")
        print("2. Request academic access")
        print("3. Download and extract to:", output_dir)
        return None


def extract_stems(
    musdb_dir: str = "data/musdb18",
    output_dir: str = "data/musdb18_stems",
    stems: list = None,
):
    """Extract specific stems from MUSDB18 to WAV files.

    Useful for creating separate datasets for:
    - music_instrumental (mixture minus vocals, or 'other' + 'drums' + 'bass')
    - music_with_vocals (full mixture)
    - isolated_vocals (just vocals stem)
    - isolated_instrument (individual stems)

    Args:
        musdb_dir: Path to MUSDB18 dataset
        output_dir: Directory to save extracted stems
        stems: List of stems to extract ['vocals', 'drums', 'bass', 'other', 'mixture']
               If None, extracts all stems
    """
    if not HAS_MUSDB:
        print("musdb package required for stem extraction. Install with: pip install musdb")
        return

    import soundfile as sf

    stems = stems or ['vocals', 'drums', 'bass', 'other', 'mixture']

    output_dir = Path(output_dir)

    # Create output directories for each stem
    for stem in stems:
        (output_dir / stem).mkdir(parents=True, exist_ok=True)

    # Also create instrumental (no vocals) directory
    (output_dir / 'instrumental').mkdir(parents=True, exist_ok=True)

    mus = musdb.DB(root=musdb_dir)

    print(f"Extracting stems from {len(mus)} tracks...")

    for i, track in enumerate(mus):
        print(f"  [{i+1}/{len(mus)}] {track.name}")

        # Extract each stem
        for stem in stems:
            if stem == 'mixture':
                audio = track.audio
            else:
                audio = track.targets[stem].audio if stem in track.targets else None

            if audio is not None:
                out_path = output_dir / stem / f"{track.name}.wav"
                sf.write(str(out_path), audio, track.rate)

        # Create instrumental (mixture - vocals)
        if 'mixture' in stems or 'vocals' in stems:
            instrumental = track.audio - track.targets['vocals'].audio
            out_path = output_dir / 'instrumental' / f"{track.name}.wav"
            sf.write(str(out_path), instrumental, track.rate)

    print(f"\nExtracted stems to {output_dir}")
    for stem in stems + ['instrumental']:
        stem_files = list((output_dir / stem).glob("*.wav"))
        print(f"  {stem}: {len(stem_files)} files")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download/setup MUSDB18 dataset")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/musdb18",
        help="Output directory for MUSDB18"
    )
    parser.add_argument(
        "--extract-stems",
        action="store_true",
        help="Extract stems to WAV files after download"
    )
    parser.add_argument(
        "--stems-dir",
        default="data/musdb18_stems",
        help="Output directory for extracted stems"
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=["vocals", "drums", "bass", "other", "mixture"],
        help="Which stems to extract"
    )

    args = parser.parse_args()

    # Setup MUSDB18
    result = setup_musdb18_via_package(output_dir=args.output_dir)

    # Extract stems if requested
    if result and args.extract_stems:
        extract_stems(
            musdb_dir=args.output_dir,
            output_dir=args.stems_dir,
            stems=args.stems,
        )


if __name__ == "__main__":
    main()
