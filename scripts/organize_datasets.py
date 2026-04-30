"""Organize downloaded datasets for the audio deepfake detection pipeline.

This script creates a unified directory structure:

data/
  experiments/
    single_voice/
      real/     <- LibriSpeech
      fake/     <- ASVspoof 2019 LA spoof
    music_instrumental/
      real/     <- MUSDB18 instrumental stems
      fake/     <- FakeMusicCaps (instrumental tracks)
    music_with_vocals/
      real/     <- MUSDB18 mixture (full songs)
      fake/     <- FakeMusicCaps (vocal tracks)
    isolated_vocals/
      real/     <- MUSDB18 vocal stems
      fake/     <- (extracted from fake music)
    deepspeak_v2/
      real/     <- DeepSpeak v2 authentic speaker audio
      fake/     <- DeepSpeak v2 voice-cloned audio (ElevenLabs, PlayHT, Speechify)

The pipeline's process_dataset() expects:
    process_dataset(real_dir, fake_dir, audio_type)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, List
import random


def create_symlinks_or_copy(
    src_files: List[Path],
    dest_dir: Path,
    use_symlinks: bool = True,
    max_files: Optional[int] = None,
):
    """Create symlinks or copy files to destination.

    Args:
        src_files: List of source file paths
        dest_dir: Destination directory
        use_symlinks: Use symlinks instead of copying (saves disk space)
        max_files: Maximum number of files to link/copy (None = all)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if max_files and len(src_files) > max_files:
        src_files = random.sample(src_files, max_files)

    for src in src_files:
        dest = dest_dir / src.name
        if dest.exists():
            continue

        if use_symlinks:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)

    return len(list(dest_dir.iterdir()))


def organize_single_voice(
    data_dir: Path,
    output_dir: Path,
    use_symlinks: bool = True,
):
    """Organize single voice datasets (LibriSpeech + ASVspoof)."""
    exp_dir = output_dir / "single_voice"
    real_dir = exp_dir / "real"
    fake_dir = exp_dir / "fake"

    print("\n=== Organizing single_voice ===")

    # Real: LibriSpeech
    librispeech_dir = data_dir / "librispeech" / "LibriSpeech"
    if librispeech_dir.exists():
        flac_files = list(librispeech_dir.rglob("*.flac"))
        if flac_files:
            count = create_symlinks_or_copy(flac_files, real_dir, use_symlinks)
            print(f"  Real (LibriSpeech): {count} files")
    else:
        print(f"  Real: LibriSpeech not found at {librispeech_dir}")

    # Fake: ASVspoof 2019 LA - spoof samples
    asvspoof_dir = data_dir / "asvspoof2019" / "LA"
    if asvspoof_dir.exists():
        # ASVspoof structure: LA/ASVspoof2019_LA_train/flac/
        # Spoof files typically have specific naming or are in protocol files
        flac_files = list(asvspoof_dir.rglob("*.flac"))

        # Filter for spoof samples using protocol if available
        protocol_file = asvspoof_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
        if protocol_file.exists():
            spoof_ids = set()
            with open(protocol_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[4] == 'spoof':
                        spoof_ids.add(parts[1])  # File ID

            spoof_files = [f for f in flac_files if f.stem in spoof_ids]
            if spoof_files:
                count = create_symlinks_or_copy(spoof_files, fake_dir, use_symlinks)
                print(f"  Fake (ASVspoof spoof): {count} files")
        else:
            # Without protocol, take all files (may need manual filtering)
            count = create_symlinks_or_copy(flac_files, fake_dir, use_symlinks)
            print(f"  Fake (ASVspoof all): {count} files (may need filtering)")
    else:
        print(f"  Fake: ASVspoof not found at {asvspoof_dir}")

    return exp_dir


def organize_music_instrumental(
    data_dir: Path,
    output_dir: Path,
    use_symlinks: bool = True,
):
    """Organize instrumental music datasets (MUSDB18 stems + FakeMusicCaps)."""
    exp_dir = output_dir / "music_instrumental"
    real_dir = exp_dir / "real"
    fake_dir = exp_dir / "fake"

    print("\n=== Organizing music_instrumental ===")

    # Real: MUSDB18 instrumental stems (accompaniment without vocals)
    musdb_stems = data_dir / "musdb18_stems" / "instrumental"
    if musdb_stems.exists():
        wav_files = list(musdb_stems.glob("*.wav"))
        if wav_files:
            count = create_symlinks_or_copy(wav_files, real_dir, use_symlinks)
            print(f"  Real (MUSDB18 instrumental): {count} files")
    else:
        # Try 'other' stem as fallback (accompaniment without vocals/drums/bass)
        musdb_other = data_dir / "musdb18_stems" / "other"
        if musdb_other.exists():
            wav_files = list(musdb_other.glob("*.wav"))
            count = create_symlinks_or_copy(wav_files, real_dir, use_symlinks)
            print(f"  Real (MUSDB18 other stem): {count} files")
        else:
            print(f"  Real: MUSDB18 stems not found")

    # Fake: FakeMusicCaps - filter for instrumental if possible
    fakemusiccaps_dir = data_dir / "fakemusiccaps"
    if fakemusiccaps_dir.exists():
        wav_files = list(fakemusiccaps_dir.rglob("*.wav"))
        # For now, use all FakeMusicCaps (they're all AI-generated)
        # Could filter by metadata if available
        count = create_symlinks_or_copy(wav_files, fake_dir, use_symlinks)
        print(f"  Fake (FakeMusicCaps): {count} files")
    else:
        print(f"  Fake: FakeMusicCaps not found at {fakemusiccaps_dir}")

    return exp_dir


def organize_music_with_vocals(
    data_dir: Path,
    output_dir: Path,
    use_symlinks: bool = True,
):
    """Organize music with vocals datasets."""
    exp_dir = output_dir / "music_with_vocals"
    real_dir = exp_dir / "real"
    fake_dir = exp_dir / "fake"

    print("\n=== Organizing music_with_vocals ===")

    # Real: MUSDB18 full mixtures
    musdb_mixture = data_dir / "musdb18_stems" / "mixture"
    if musdb_mixture.exists():
        wav_files = list(musdb_mixture.glob("*.wav"))
        if wav_files:
            count = create_symlinks_or_copy(wav_files, real_dir, use_symlinks)
            print(f"  Real (MUSDB18 mixture): {count} files")
    else:
        print(f"  Real: MUSDB18 mixtures not found")

    # Fake: FakeMusicCaps (same source, different audio type config)
    fakemusiccaps_dir = data_dir / "fakemusiccaps"
    if fakemusiccaps_dir.exists():
        wav_files = list(fakemusiccaps_dir.rglob("*.wav"))
        count = create_symlinks_or_copy(wav_files, fake_dir, use_symlinks)
        print(f"  Fake (FakeMusicCaps): {count} files")
    else:
        print(f"  Fake: FakeMusicCaps not found")

    return exp_dir


def organize_isolated_vocals(
    data_dir: Path,
    output_dir: Path,
    use_symlinks: bool = True,
):
    """Organize isolated vocals datasets."""
    exp_dir = output_dir / "isolated_vocals"
    real_dir = exp_dir / "real"
    fake_dir = exp_dir / "fake"

    print("\n=== Organizing isolated_vocals ===")

    # Real: MUSDB18 vocal stems
    musdb_vocals = data_dir / "musdb18_stems" / "vocals"
    if musdb_vocals.exists():
        wav_files = list(musdb_vocals.glob("*.wav"))
        if wav_files:
            count = create_symlinks_or_copy(wav_files, real_dir, use_symlinks)
            print(f"  Real (MUSDB18 vocals): {count} files")
    else:
        print(f"  Real: MUSDB18 vocal stems not found")

    # Fake: Would need to isolate vocals from AI-generated music
    # This requires running voice isolation on FakeMusicCaps
    print(f"  Fake: Requires voice isolation from AI music (run separately)")

    return exp_dir


def organize_deepspeak_v2(
    data_dir: Path,
    output_dir: Path,
    use_symlinks: bool = True,
    voice_cloned_only: bool = True,
    split: str = "train",
):
    """Organize DeepSpeak v2 dataset (talking head audio).

    DeepSpeak v2 contains video with audio. The download_deepspeak.py script
    extracts audio and organizes by split/real/fake with identity subdirectories.

    For fake samples, audio-config can be:
      - "real": original speaker audio (video is fake, audio is real)
      - "elevenlabs", "playht", "speechify": voice-cloned audio

    Args:
        data_dir: Root data directory
        output_dir: Output experiments directory
        use_symlinks: Use symlinks instead of copying
        voice_cloned_only: If True, only include voice-cloned fake samples
        split: Which split to organize ("train" or "test")
    """
    # Output directory includes split name to keep train/test separate
    exp_dir = output_dir / f"deepspeak_v2_{split}"
    real_dir = exp_dir / "real"
    fake_dir = exp_dir / "fake"

    print(f"\n=== Organizing deepspeak_v2 ({split}) ===")

    # Source data is organized as deepspeak_v2/split/real and deepspeak_v2/split/fake
    deepspeak_split_dir = data_dir / "deepspeak_v2" / split
    if not deepspeak_split_dir.exists():
        print(f"  DeepSpeak v2 {split} split not found at {deepspeak_split_dir}")
        print(f"  Run: python scripts/download_deepspeak.py --split {split}")
        return exp_dir

    # Check for metadata to filter voice-cloned samples
    metadata_path = deepspeak_split_dir / "metadata.json"
    voice_cloned_files = set()

    if metadata_path.exists() and voice_cloned_only:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract filenames of voice-cloned samples
        voice_clone_configs = {"elevenlabs", "playht", "speechify"}
        for entry in metadata.get('fake', []):
            audio_config = entry.get('audio_config', '')
            if audio_config in voice_clone_configs:
                file_path = Path(entry.get('file_path', ''))
                voice_cloned_files.add(file_path.name)

        print(f"  Found {len(voice_cloned_files)} voice-cloned samples in metadata")

    # Real: DeepSpeak authentic speaker audio
    deepspeak_real = deepspeak_split_dir / "real"
    if deepspeak_real.exists():
        # Collect all wav files from identity subdirectories
        wav_files = list(deepspeak_real.rglob("*.wav"))
        if wav_files:
            count = create_symlinks_or_copy(wav_files, real_dir, use_symlinks)
            print(f"  Real (DeepSpeak authentic): {count} files")
    else:
        print(f"  Real: DeepSpeak real audio not found")

    # Fake: DeepSpeak voice-cloned audio
    deepspeak_fake = deepspeak_split_dir / "fake"
    if deepspeak_fake.exists():
        wav_files = list(deepspeak_fake.rglob("*.wav"))

        # Filter for voice-cloned only if metadata available and flag set
        if voice_cloned_files and voice_cloned_only:
            wav_files = [f for f in wav_files if f.name in voice_cloned_files]
            print(f"  Filtering to {len(wav_files)} voice-cloned samples")

        if wav_files:
            count = create_symlinks_or_copy(wav_files, fake_dir, use_symlinks)
            print(f"  Fake (DeepSpeak voice-cloned): {count} files")
        else:
            print(f"  Fake: No voice-cloned samples found (may need --include-all-fake)")
    else:
        print(f"  Fake: DeepSpeak fake audio not found")

    return exp_dir


def check_dataset_status(data_dir: Path):
    """Check status of all downloaded datasets."""
    print("\n=== Dataset Status ===")

    datasets = {
        "LibriSpeech": data_dir / "librispeech" / "LibriSpeech",
        "ASVspoof 2019": data_dir / "asvspoof2019" / "LA",
        "MUSDB18": data_dir / "musdb18",
        "MUSDB18 Stems": data_dir / "musdb18_stems",
        "FakeMusicCaps": data_dir / "fakemusiccaps",
        "DeepSpeak v2": data_dir / "deepspeak_v2",
        "SONICS": data_dir / "sonics",
    }

    for name, path in datasets.items():
        if path.exists():
            if path.is_dir():
                # Count audio files
                audio_count = len(list(path.rglob("*.wav"))) + \
                             len(list(path.rglob("*.flac"))) + \
                             len(list(path.rglob("*.mp3")))
                print(f"  {name}: READY ({audio_count} audio files)")
            else:
                print(f"  {name}: READY (file exists)")
        else:
            # Check if download in progress
            zip_path = path.parent / f"{path.name}.zip"
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                print(f"  {name}: DOWNLOADING ({size_mb:.1f} MB)")
            else:
                print(f"  {name}: NOT FOUND")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize datasets for audio deepfake detection pipeline"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Root data directory containing downloaded datasets"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/experiments",
        help="Output directory for organized experiment data"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check dataset status, don't organize"
    )
    parser.add_argument(
        "--audio-types",
        nargs="+",
        default=["single_voice", "music_instrumental", "music_with_vocals", "isolated_vocals", "deepspeak_v2"],
        help="Which audio types to organize"
    )
    parser.add_argument(
        "--include-all-fake",
        action="store_true",
        help="For DeepSpeak v2: include all fake samples, not just voice-cloned"
    )
    parser.add_argument(
        "--deepspeak-split",
        default="train",
        choices=["train", "test"],
        help="For DeepSpeak v2: which split to organize (default: train)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    use_symlinks = not args.copy

    # Always check status
    check_dataset_status(data_dir)

    if args.status_only:
        return

    print(f"\nOrganizing datasets to {output_dir}")
    print(f"Using {'symlinks' if use_symlinks else 'file copies'}")

    # Organize each audio type
    if "single_voice" in args.audio_types:
        organize_single_voice(data_dir, output_dir, use_symlinks)

    if "music_instrumental" in args.audio_types:
        organize_music_instrumental(data_dir, output_dir, use_symlinks)

    if "music_with_vocals" in args.audio_types:
        organize_music_with_vocals(data_dir, output_dir, use_symlinks)

    if "isolated_vocals" in args.audio_types:
        organize_isolated_vocals(data_dir, output_dir, use_symlinks)

    if "deepspeak_v2" in args.audio_types:
        organize_deepspeak_v2(
            data_dir, output_dir, use_symlinks,
            voice_cloned_only=not args.include_all_fake,
            split=args.deepspeak_split
        )

    print("\n=== Organization Complete ===")
    print(f"Experiment data ready at: {output_dir}")
    print("\nTo run extraction:")
    print(f"  python scripts/extract_features.py {output_dir}/single_voice/real -t single_voice -l 0")
    print(f"  python scripts/extract_features.py {output_dir}/single_voice/fake -t single_voice -l 1")


if __name__ == "__main__":
    main()
