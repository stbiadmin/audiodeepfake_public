#!/usr/bin/env python3
"""Generate embeddings for feature JSON files that lack them.

Reads a feature JSON file, identifies entries without embeddings,
extracts MS-CLAP embeddings from the raw audio files, and saves
the updated JSON with embeddings included.

Usage:
    # Process a single file
    python sota_benchmarking/generate_embeddings.py \
        --input data/features/msclap/combined/audeter_combined.json \
        --output data/features/msclap/combined/audeter_combined.json

    # Process mlaad real/fake separately
    python sota_benchmarking/generate_embeddings.py \
        --input data/features/msclap/raw/mlaad_english_real.json \
        --output data/features/msclap/combined/mlaad_english_real_with_emb.json

    # Dry run to see how many entries need embeddings
    python sota_benchmarking/generate_embeddings.py \
        --input data/features/msclap/combined/audeter_combined.json --dry-run
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MS-CLAP extraction settings (must match original pipeline)
MSCLAP_CONFIG = {
    "sample_rate": 48000,
    "segment_duration": 2.0,
    "segment_hop": 1.0,
    "embedding_dim": 1024,
}


class EmbeddingExtractor:
    """Extracts MS-CLAP embeddings from audio files."""

    def __init__(self):
        self.model = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return
        print("Loading MS-CLAP model...")
        from msclap import CLAP
        self.model = CLAP(version='2023', use_cuda=False)
        self._loaded = True
        print("Model loaded.")

    def extract(self, audio_path):
        """Extract embeddings from an audio file.

        Returns list of embedding vectors (each 1024-dim) or None on failure.
        """
        import librosa
        import soundfile as sf

        self.load_model()

        try:
            audio, sr = librosa.load(
                audio_path,
                sr=MSCLAP_CONFIG["sample_rate"],
                mono=True
            )

            segment_samples = int(MSCLAP_CONFIG["segment_duration"] * sr)
            hop_samples = int(MSCLAP_CONFIG["segment_hop"] * sr)

            segments = []
            start = 0
            while start + segment_samples <= len(audio):
                segments.append(audio[start:start + segment_samples])
                start += hop_samples

            if len(segments) < 2:
                return None

            embeddings = []
            for segment in segments:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, segment, sr)
                    temp_path = f.name
                try:
                    emb = self.model.get_audio_embeddings([temp_path])
                    embeddings.append(emb[0].tolist())
                finally:
                    os.unlink(temp_path)

            return embeddings

        except Exception as e:
            print(f"  Error extracting from {audio_path}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate MS-CLAP embeddings for feature JSON entries"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input feature JSON file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report counts, don't extract")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Save progress every N entries")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start processing from this index (for resuming)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Stop processing at this index (exclusive, for parallel shards)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        entries = json.load(f)

    total = len(entries)
    needs_embedding = sum(
        1 for e in entries
        if not e.get("embeddings") or len(e.get("embeddings", [])) == 0
    )
    has_embedding = total - needs_embedding

    print(f"Total entries: {total}")
    print(f"  With embeddings: {has_embedding}")
    print(f"  Need embeddings: {needs_embedding}")

    if args.dry_run:
        # Show file_path availability
        has_path = sum(1 for e in entries if e.get("file_path"))
        print(f"  Have file_path: {has_path}")
        missing_path = sum(
            1 for e in entries
            if (not e.get("embeddings") or len(e.get("embeddings", [])) == 0)
            and not e.get("file_path")
        )
        print(f"  Need embeddings but no file_path: {missing_path}")
        return

    if needs_embedding == 0:
        print("All entries already have embeddings. Nothing to do.")
        return

    extractor = EmbeddingExtractor()

    processed = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    end_idx = args.end_idx if args.end_idx is not None else total

    for i, entry in enumerate(entries):
        if i < args.start_idx:
            continue
        if i >= end_idx:
            break

        if entry.get("embeddings") and len(entry.get("embeddings", [])) > 0:
            continue

        file_path = entry.get("file_path", "")
        if not file_path:
            skipped += 1
            continue

        # Resolve relative paths against project root
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = PROJECT_ROOT / file_path

        if not full_path.exists():
            print(f"  [{i}/{total}] File not found: {full_path}")
            errors += 1
            continue

        embeddings = extractor.extract(str(full_path))
        if embeddings is None:
            errors += 1
            continue

        entry["embeddings"] = embeddings
        processed += 1

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (needs_embedding - processed) / rate if rate > 0 else 0

        if processed % 10 == 0:
            print(
                f"  [{i}/{total}] Processed: {processed}/{needs_embedding} "
                f"({rate:.1f}/s, ~{remaining/3600:.1f}h remaining) "
                f"Errors: {errors}"
            )

        # Periodic save
        if processed % args.batch_size == 0:
            print(f"  Saving checkpoint at {processed} processed...")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(entries, f)

    # Final save - if using sharded mode, save only the processed slice
    if args.end_idx is not None:
        shard_path = args.output.parent / f"{args.output.stem}_shard_{args.start_idx}_{end_idx}{args.output.suffix}"
        print(f"\nSaving shard to {shard_path}...")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shard_path, "w") as f:
            json.dump(entries[args.start_idx:end_idx], f)
    else:
        print(f"\nSaving final output to {args.output}...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(entries, f)

    elapsed = time.time() - start_time
    print(f"\nDone. Processed: {processed}, Errors: {errors}, Skipped: {skipped}")
    print(f"Time: {elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()
