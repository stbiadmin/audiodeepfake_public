"""Combine feature files for all audio types.

Creates combined JSON files for classification training:
- Music types: original_real + augmented_real + fake
- Other types: real + fake

For both LAION-CLAP and MS-CLAP embeddings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# Base paths
FEATURES_DIR = Path("data/features")

# Audio types with their augmentation status
AUDIO_TYPES = {
    "single_voice": False,           # No augmentation
    "music_instrumental": True,      # Has augmentation
    "music_with_vocals": True,       # Has augmentation
    "deepspeak_v2_train": False,     # No augmentation
}

EMBEDDING_MODELS = ["laion_clap", "msclap"]


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load JSON file and return list of samples."""
    with open(path) as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: Path) -> None:
    """Save data to JSON file (compact format for efficiency)."""
    with open(path, 'w') as f:
        json.dump(data, f)


def combine_features(audio_type: str, embedding_model: str, has_augmentation: bool) -> Dict[str, Any]:
    """Combine real (+ augmented if applicable) and fake samples."""
    base_dir = FEATURES_DIR / embedding_model
    combined_dir = base_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Load original real
    original_real = load_json(base_dir / f"{audio_type}_real.json")
    for sample in original_real:
        sample["augmented"] = False

    # Load augmented real if applicable
    augmented_real = []
    if has_augmentation:
        augmented_path = base_dir / f"{audio_type}_real_augmented.json"
        if augmented_path.exists():
            augmented_real = load_json(augmented_path)
            for sample in augmented_real:
                sample["augmented"] = True

    # Load fake
    fake = load_json(base_dir / f"{audio_type}_fake.json")
    for sample in fake:
        sample["augmented"] = False

    # Combine all samples
    combined = original_real + augmented_real + fake

    # Save combined file
    output_path = combined_dir / f"{audio_type}_combined.json"
    save_json(combined, output_path)

    return {
        "audio_type": audio_type,
        "embedding_model": embedding_model,
        "original_real": len(original_real),
        "augmented_real": len(augmented_real),
        "total_real": len(original_real) + len(augmented_real),
        "fake": len(fake),
        "total": len(combined),
        "output_path": str(output_path)
    }


def main():
    print("=" * 60)
    print("Combining Feature Files for All Audio Types")
    print("=" * 60)

    results = []

    for embedding_model in EMBEDDING_MODELS:
        print(f"\n{embedding_model.upper()}")
        print("-" * 40)

        for audio_type, has_augmentation in AUDIO_TYPES.items():
            result = combine_features(audio_type, embedding_model, has_augmentation)
            results.append(result)

            print(f"\n{audio_type}:")
            print(f"  Original real: {result['original_real']}")
            print(f"  Augmented real: {result['augmented_real']}")
            print(f"  Total real: {result['total_real']}")
            print(f"  Fake: {result['fake']}")
            print(f"  TOTAL: {result['total']}")
            print(f"  Output: {result['output_path']}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for r in results:
        real_fake_ratio = r['total_real'] / r['fake']
        print(f"{r['embedding_model']}/{r['audio_type']}: "
              f"{r['total_real']} real + {r['fake']} fake = {r['total']} total "
              f"(ratio: {real_fake_ratio:.2f})")


if __name__ == "__main__":
    main()
