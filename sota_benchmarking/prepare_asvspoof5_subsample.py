#!/usr/bin/env python3
"""Prepare a balanced 10K subsample of ASVspoof5 for SOTA benchmarking.

Samples 5K real entries (with embeddings) from real_fixed.json using reservoir
sampling, and 5K fake entries from fake.json. Saves to eval_data/asvspoof5/.

Real entries already have embeddings in real_fixed.json. Fake entries need
embedding extraction via generate_embeddings.py after this script runs.

Usage:
    python sota_benchmarking/prepare_asvspoof5_subsample.py
    # Then extract fake embeddings:
    python sota_benchmarking/generate_embeddings.py \
        --input sota_benchmarking/eval_data/asvspoof5/fake.json
"""

import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASVSPOOF5_DIR = PROJECT_ROOT / "data" / "eval_features" / "msclap" / "asvspoof5"
OUTPUT_DIR = Path(__file__).resolve().parent / "eval_data" / "asvspoof5"

SAMPLE_SIZE = 5000
SEED = 42


def reservoir_sample_with_embeddings(json_path, n, seed=42):
    """Stream a JSON array file and reservoir-sample n entries that have embeddings.

    Uses a character-level parser to avoid loading the entire file into memory.
    real_fixed.json is 22GB so we can't load it all at once.
    """
    rng = random.Random(seed)
    reservoir = []
    count = 0
    skipped = 0

    print(f"  Streaming {json_path.name} (reservoir sampling {n} entries)...")
    start = time.time()

    # Use json.JSONDecoder for incremental parsing
    decoder = json.JSONDecoder()
    buffer = ""
    depth = 0
    in_array = False

    with open(json_path, "r") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            buffer += chunk

            while buffer:
                buffer = buffer.lstrip()
                if not buffer:
                    break

                # Skip array brackets
                if buffer[0] == "[":
                    in_array = True
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                # Try to decode a JSON object
                try:
                    obj, end_idx = decoder.raw_decode(buffer)
                    buffer = buffer[end_idx:]
                except json.JSONDecodeError:
                    # Incomplete object, need more data
                    break

                # Check if entry has embeddings
                if not obj.get("embeddings") or len(obj["embeddings"]) == 0:
                    skipped += 1
                    continue

                # Verify audio file exists
                file_path = obj.get("file_path", "")
                if file_path:
                    full_path = Path(file_path)
                    if not full_path.is_absolute():
                        full_path = PROJECT_ROOT / file_path
                    if not full_path.exists():
                        skipped += 1
                        continue

                count += 1

                # Reservoir sampling
                if len(reservoir) < n:
                    reservoir.append(obj)
                else:
                    j = rng.randint(0, count - 1)
                    if j < n:
                        reservoir[j] = obj

                if count % 10000 == 0:
                    elapsed = time.time() - start
                    print(f"    Processed {count} valid entries ({skipped} skipped), "
                          f"{elapsed:.0f}s elapsed...")

    elapsed = time.time() - start
    print(f"    Done: {count} valid entries found, {skipped} skipped, "
          f"sampled {len(reservoir)}, {elapsed:.0f}s")
    return reservoir


def sample_fake_entries(json_path, n, seed=42):
    """Load fake.json and randomly sample n entries."""
    print(f"  Loading {json_path.name}...")
    with open(json_path, "r") as f:
        entries = json.load(f)
    print(f"    Loaded {len(entries)} entries")

    # Filter to entries with audio files on disk
    valid = []
    for e in entries:
        file_path = e.get("file_path", "")
        if file_path:
            full_path = Path(file_path)
            if not full_path.is_absolute():
                full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                valid.append(e)

    print(f"    {len(valid)} entries have audio files on disk")

    rng = random.Random(seed)
    sampled = rng.sample(valid, min(n, len(valid)))
    print(f"    Sampled {len(sampled)} entries")
    return sampled


def main():
    print("Preparing ASVspoof5 subsample for SOTA benchmarking\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_fixed = ASVSPOOF5_DIR / "real_fixed.json"
    fake_json = ASVSPOOF5_DIR / "fake.json"

    if not real_fixed.exists():
        print(f"ERROR: {real_fixed} not found")
        sys.exit(1)
    if not fake_json.exists():
        print(f"ERROR: {fake_json} not found")
        sys.exit(1)

    # Sample real entries (with embeddings) via reservoir sampling
    print("Sampling real entries (with embeddings):")
    real_sample = reservoir_sample_with_embeddings(real_fixed, SAMPLE_SIZE, SEED)

    # Save real
    real_out = OUTPUT_DIR / "real.json"
    print(f"  Saving {len(real_sample)} real entries to {real_out}...")
    with open(real_out, "w") as f:
        json.dump(real_sample, f)
    print(f"    Saved ({real_out.stat().st_size / 1024 / 1024:.1f} MB)")

    # Free memory
    del real_sample

    # Sample fake entries (no embeddings yet)
    print("\nSampling fake entries (will need embedding extraction):")
    fake_sample = sample_fake_entries(fake_json, SAMPLE_SIZE, SEED)

    # Save fake
    fake_out = OUTPUT_DIR / "fake.json"
    print(f"  Saving {len(fake_sample)} fake entries to {fake_out}...")
    with open(fake_out, "w") as f:
        json.dump(fake_sample, f)
    print(f"    Saved ({fake_out.stat().st_size / 1024 / 1024:.1f} MB)")

    # Summary
    print("\n--- Summary ---")
    print(f"Real: {SAMPLE_SIZE} entries WITH embeddings -> {real_out}")
    print(f"Fake: {SAMPLE_SIZE} entries WITHOUT embeddings -> {fake_out}")
    print("\nNext step: extract embeddings for fake entries:")
    print("  python sota_benchmarking/generate_embeddings.py \\")
    print(f"      --input {fake_out}")


if __name__ == "__main__":
    main()
