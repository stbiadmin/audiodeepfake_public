#!/usr/bin/env python3
"""Generate sample training data for external reviewers.

Creates stratified samples from the training feature data, preserving
class distributions and dataset ratios. Extracts embeddings on-the-fly
for samples that are missing them.

Usage:
    # Generate 5% sample (default)
    python scripts/generate_sample_training_data.py

    # Generate 10% sample
    python scripts/generate_sample_training_data.py --percent 10

    # Generate samples for specific embedding type only
    python scripts/generate_sample_training_data.py --embedding msclap
"""

import argparse
import json
import random
import shutil
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_training_data"

# Embedding model configurations - MUST match original extraction settings
EMBEDDING_CONFIGS = {
    "msclap": {
        "source_dir": PROJECT_ROOT / "data" / "features" / "msclap",
        "model_id": "msclap",
        "embedding_dim": 1024,
        "sample_rate": 48000,
        "segment_duration": 2.0,
        "segment_hop": 1.0,
        "description": "MS-CLAP (Microsoft CLAP, larger_clap_music_and_speech) - Primary model",
        "datasets": [
            ("single_voice_real.json", "single_voice_fake.json", "single_voice"),
            ("deepspeak_v2_train_real.json", "deepspeak_v2_train_fake.json", "deepspeak_v2"),
            ("mlaad_english_real.json", "mlaad_english_fake.json", "mlaad"),
            ("fakeavceleb_real.json", "fakeavceleb_fake.json", "fakeavceleb"),
            ("music_instrumental_real.json", "music_instrumental_fake.json", "music_instrumental"),
            ("music_with_vocals_real.json", "music_with_vocals_fake.json", "music_with_vocals"),
        ],
    },
    "msclap_optc": {
        "source_dir": PROJECT_ROOT / "data" / "features" / "msclap_optc",
        "model_id": "msclap",  # Same model, different segmentation
        "embedding_dim": 1024,
        "sample_rate": 48000,
        # Option C: Shorter segments for short-duration files (e.g., DeepSpeak)
        # Allows processing files as short as 2.25s instead of 4s
        "segment_duration": 1.5,
        "segment_hop": 0.75,
        "description": "MS-CLAP with Option C segmentation (1.5s segments, 0.75s hop) for short audio",
        "datasets": [
            ("single_voice_real.json", "single_voice_fake.json", "single_voice"),
            ("deepspeak_real.json", "deepspeak_fake.json", "deepspeak"),
            ("audeter_real.json", "audeter_fake.json", "audeter"),
        ],
    },
    "wavlm": {
        "source_dir": PROJECT_ROOT / "data" / "features" / "wavlm" / "raw",
        "model_id": "wavlm",
        "embedding_dim": 1024,
        "sample_rate": 16000,
        "segment_duration": 2.0,
        "segment_hop": 1.0,
        "description": "WavLM-Large - Speech-focused self-supervised model",
        "datasets": [
            ("single_voice_real.json", "single_voice_fake.json", "single_voice"),
            ("deepspeak_v2_train_real.json", "deepspeak_v2_train_fake.json", "deepspeak_v2"),
            ("mlaad_english_real.json", "mlaad_english_fake.json", "mlaad"),
        ],
    },
    "laion_clap": {
        "source_dir": PROJECT_ROOT / "data" / "features" / "laion_clap",
        "model_id": "laion_clap",
        "embedding_dim": 512,
        "sample_rate": 48000,
        "segment_duration": 2.0,
        "segment_hop": 1.0,
        "description": "LAION-CLAP (laion/clap-htsat-unfused) - General audio embeddings",
        "datasets": [
            ("single_voice_real.json", "single_voice_fake.json", "single_voice"),
            ("deepspeak_v2_train_real.json", "deepspeak_v2_train_fake.json", "deepspeak_v2"),
            ("music_instrumental_real.json", "music_instrumental_fake.json", "music_instrumental"),
            ("music_with_vocals_real.json", "music_with_vocals_fake.json", "music_with_vocals"),
        ],
    },
}


class EmbeddingExtractor:
    """Extracts embeddings using the appropriate model."""

    def __init__(self, model_id: str, config: Dict):
        self.model_id = model_id
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self):
        """Lazy load the embedding model."""
        if self._loaded:
            return

        print(f"    Loading {self.model_id} model...")

        if self.model_id == "msclap":
            from msclap import CLAP
            self.model = CLAP(version='2023', use_cuda=False)

        elif self.model_id == "laion_clap":
            import laion_clap
            self.model = laion_clap.CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()

        elif self.model_id == "wavlm":
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
            self.model.eval()

        self._loaded = True

    def extract(self, audio_path: str) -> Optional[Tuple[List[List[float]], List[float]]]:
        """Extract embeddings and similarities from audio file.

        Returns:
            Tuple of (embeddings, similarities) or None if extraction fails
        """
        import os
        import tempfile

        import librosa
        import soundfile as sf
        import torch

        self.load_model()

        try:
            # Load audio at correct sample rate
            audio, sr = librosa.load(
                audio_path,
                sr=self.config["sample_rate"],
                mono=True
            )

            # Segment audio
            segment_samples = int(self.config["segment_duration"] * sr)
            hop_samples = int(self.config["segment_hop"] * sr)

            segments = []
            start = 0
            while start + segment_samples <= len(audio):
                segment = audio[start:start + segment_samples]
                segments.append(segment)
                start += hop_samples

            if len(segments) < 2:
                return None

            # Extract embeddings based on model type
            embeddings = []

            if self.model_id == "msclap":
                # MS-CLAP requires file paths, use temp files
                for segment in segments:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        sf.write(f.name, segment, sr)
                        temp_path = f.name
                    try:
                        emb = self.model.get_audio_embeddings([temp_path])
                        embeddings.append(emb[0].tolist())
                    finally:
                        os.unlink(temp_path)

            elif self.model_id == "laion_clap":
                # LAION-CLAP accepts audio arrays directly
                for segment in segments:
                    with torch.no_grad():
                        emb = self.model.get_audio_embedding_from_data(
                            x=[segment], use_tensor=False
                        )
                    embeddings.append(emb[0].tolist())

            elif self.model_id == "wavlm":
                for segment in segments:
                    inputs = self.processor(
                        segment, sampling_rate=sr, return_tensors="pt", padding=True
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        emb = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(emb[0].tolist())

            # Compute pairwise cosine similarities
            similarities = []
            for i, j in combinations(range(len(embeddings)), 2):
                emb_i = np.array(embeddings[i])
                emb_j = np.array(embeddings[j])
                sim = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))
                similarities.append(sim)

            return embeddings, similarities

        except Exception as e:
            print(f"      Error extracting from {audio_path}: {e}")
            return None


def load_features(filepath: Path) -> List[Dict]:
    """Load features from JSON file."""
    if not filepath.exists():
        return []
    with open(filepath) as f:
        return json.load(f)


def stratified_sample(data: List[Dict], percent: float, seed: int = 42) -> List[Dict]:
    """Sample data while preserving distribution."""
    if not data:
        return []

    random.seed(seed)
    n_samples = max(1, int(len(data) * percent / 100))
    n_samples = min(n_samples, len(data))

    return random.sample(data, n_samples)


def anonymize_paths(data: List[Dict]) -> List[Dict]:
    """Anonymize file paths for privacy."""
    anonymized = []
    for i, item in enumerate(data):
        item_copy = item.copy()
        if "file_path" in item_copy:
            original_path = Path(item_copy["file_path"])
            item_copy["file_path"] = f"sample_{i:06d}{original_path.suffix}"
        anonymized.append(item_copy)
    return anonymized


def has_valid_embeddings(sample: Dict, expected_dim: int) -> bool:
    """Check if sample has valid embeddings."""
    if "embeddings" not in sample or not sample["embeddings"]:
        return False
    if not isinstance(sample["embeddings"], list):
        return False
    if len(sample["embeddings"]) == 0:
        return False
    # Check dimension of first embedding
    if len(sample["embeddings"][0]) != expected_dim:
        return False
    return True


def generate_samples(
    embedding_type: str,
    percent: float,
    output_dir: Path,
    seed: int = 42,
) -> Dict:
    """Generate samples for a specific embedding type, extracting embeddings as needed."""

    config = EMBEDDING_CONFIGS.get(embedding_type)
    if not config:
        print(f"Unknown embedding type: {embedding_type}")
        return {}

    source_dir = config["source_dir"]
    embed_output_dir = output_dir / embedding_type
    embed_output_dir.mkdir(parents=True, exist_ok=True)

    # Create extractor (lazy loaded)
    extractor = EmbeddingExtractor(config["model_id"], config)

    stats = {
        "embedding_type": embedding_type,
        "description": config["description"],
        "embedding_dim": config["embedding_dim"],
        "segment_duration": config["segment_duration"],
        "segment_hop": config["segment_hop"],
        "sample_percent": percent,
        "seed": seed,
        "datasets": {},
    }

    for real_file, fake_file, dataset_name in config["datasets"]:
        real_path = source_dir / real_file
        fake_path = source_dir / fake_file

        real_data = load_features(real_path)
        fake_data = load_features(fake_path)

        if not real_data and not fake_data:
            print(f"  Skipping {dataset_name}: no data found")
            continue

        # Sample each class (keep original paths for now)
        real_sampled = stratified_sample(real_data, percent, seed)
        fake_sampled = stratified_sample(fake_data, percent, seed)

        # Process samples - extract embeddings if missing, write progressively
        print(f"  {dataset_name}:")

        dataset_dir = embed_output_dir / dataset_name

        real_count = process_samples(
            real_sampled, extractor, config["embedding_dim"], "real",
            dataset_dir / "real.json"
        )
        fake_count = process_samples(
            fake_sampled, extractor, config["embedding_dim"], "fake",
            dataset_dir / "fake.json"
        )

        # Record stats
        stats["datasets"][dataset_name] = {
            "original_real": len(real_data),
            "original_fake": len(fake_data),
            "sampled_real": real_count,
            "sampled_fake": fake_count,
        }

        print(f"    Final: {real_count} real, {fake_count} fake")

    # Save stats
    with open(embed_output_dir / "sample_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def process_samples(
    samples: List[Dict],
    extractor: EmbeddingExtractor,
    expected_dim: int,
    label: str,
    output_path: Path,
) -> int:
    """Process samples, extracting embeddings where missing. Writes progressively.

    Returns:
        Number of samples written
    """
    if not samples:
        return 0

    # Count how many need extraction
    need_extraction = sum(
        1 for s in samples if not has_valid_embeddings(s, expected_dim)
    )

    print(f"    {label}: {len(samples)} sampled, {need_extraction} need embedding extraction")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    extracted_count = 0
    failed_count = 0
    last_write = 0

    for sample in tqdm(samples, desc=f"    {label}", leave=False):
        if has_valid_embeddings(sample, expected_dim):
            processed.append(sample)
        else:
            # Need to extract embeddings
            audio_path = sample.get("file_path", "")
            full_path = PROJECT_ROOT / audio_path

            if not full_path.exists():
                full_path = Path(audio_path)
                if not full_path.exists():
                    failed_count += 1
                    continue

            result = extractor.extract(str(full_path))
            if result:
                embeddings, similarities = result
                sample_copy = sample.copy()
                sample_copy["embeddings"] = embeddings
                sample_copy["similarities"] = similarities
                processed.append(sample_copy)
                extracted_count += 1
            else:
                failed_count += 1

        # Progressive write every 200 samples
        if len(processed) - last_write >= 200:
            anonymized = anonymize_paths(processed)
            with open(output_path, 'w') as f:
                json.dump(anonymized, f, indent=2)
            last_write = len(processed)

    # Final write
    if processed:
        anonymized = anonymize_paths(processed)
        with open(output_path, 'w') as f:
            json.dump(anonymized, f, indent=2)

    print(f"    {label}: wrote {len(processed)}, extracted {extracted_count}, failed {failed_count}")
    return len(processed)


def generate_readme(output_dir: Path, all_stats: Dict, percent: float):
    """Generate README.md with schema documentation."""

    readme_content = f"""# Sample Training Data for Audio Deepfake Detection

This directory contains a **{percent}% stratified sample** of the training data used
in the audio deepfake detection research. The samples preserve the original class
distributions (real vs fake) for each dataset.

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure

```
sample_training_data/
├── README.md                    # This file
├── msclap/                      # MS-CLAP embeddings (primary)
│   ├── single_voice/            # ASVspoof 2019 speech data
│   │   ├── real.json
│   │   └── fake.json
│   ├── deepspeak_v2/            # DeepSpeak v2 voice cloning
│   ├── mlaad/                   # MLAAD (84 TTS models)
│   ├── fakeavceleb/             # FakeAVCeleb SV2TTS
│   ├── music_instrumental/      # MUSDB18 + FakeMusicCaps
│   ├── music_with_vocals/       # Music with vocals
│   └── sample_stats.json        # Sampling statistics
├── wavlm/                       # WavLM embeddings
├── laion_clap/                  # LAION-CLAP embeddings
└── msclap_optc/                 # MS-CLAP with Option C segmentation
```

## Data Schema

Each JSON file contains a list of sample records. Each record has the following structure:

```json
{{
  "file_path": "sample_000001.wav",
  "audio_type": "deepspeak_v2_train",
  "label": 0,
  "duration": 4.62,
  "segment_count": 3,
  "similarity_count": 3,
  "features": {{
    "mean": 0.899,
    "std": 0.021,
    "variance": 0.00045,
    "min": 0.874,
    "max": 0.926,
    "peak_to_peak": 0.052,
    "skewness": 0.087,
    "kurtosis": -1.5,
    "bimodality_coefficient": null,
    "q5": 0.876,
    "q10": 0.879,
    "q25": 0.886,
    "q50": 0.898,
    "q75": 0.912,
    "q90": 0.920,
    "q95": 0.923,
    "iqr": 0.026,
    "tail_weight_ratio": 0.0,
    "trimmed_mean": 0.899,
    "entropy": 1.099,
    "gini_coefficient": 0.013,
    "coefficient_of_variation": 0.024,
    "variance_mean_ratio": 0.0005,
    "kurtosis_variance_ratio": -3334.5,
    "skewness_kurtosis_ratio": -0.058,
    "iqr_range_ratio": 0.5,
    "median_mean_diff": -0.001,
    "n_samples": 3.0
  }},
  "metadata": {{
    "embedding_model": "msclap"
  }},
  "embeddings": [
    [0.123, 0.456, ...],  // embedding vector for segment 1
    [0.234, 0.567, ...],  // embedding vector for segment 2
    [0.345, 0.678, ...]   // embedding vector for segment 3
  ],
  "similarities": [0.92, 0.88, 0.91]  // pairwise cosine similarities
}}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | string | Anonymized filename (original paths removed for privacy) |
| `audio_type` | string | Dataset/category identifier |
| `label` | int | Class label: 0 = real (authentic), 1 = fake (AI-generated) |
| `duration` | float | Audio duration in seconds |
| `segment_count` | int | Number of segments extracted from the audio |
| `similarity_count` | int | Number of pairwise similarity values |
| `features` | object | 30 statistical features computed from similarity distribution |
| `metadata` | object | Additional metadata (embedding model, etc.) |
| `embeddings` | array | List of embedding vectors (one per segment) |
| `similarities` | array | Pairwise cosine similarities between all segment embeddings |

### Statistical Features

The `features` object contains 30 statistical measures computed from the pairwise
cosine similarity distribution:

**Basic Statistics:**
- `mean`, `std`, `variance`: Central tendency and spread
- `min`, `max`, `peak_to_peak`: Range statistics

**Distribution Shape:**
- `skewness`: Asymmetry of the distribution
- `kurtosis`: Tailedness of the distribution
- `bimodality_coefficient`: Measure of bimodality (null if insufficient data)

**Percentiles:**
- `q5`, `q10`, `q25`, `q50`, `q75`, `q90`, `q95`: Distribution quantiles
- `iqr`: Interquartile range (q75 - q25)

**Information Theory:**
- `entropy`: Shannon entropy of binned distribution
- `gini_coefficient`: Inequality measure
- `coefficient_of_variation`: Normalized standard deviation

**Derived Ratios:**
- `variance_mean_ratio`, `kurtosis_variance_ratio`, `skewness_kurtosis_ratio`
- `iqr_range_ratio`, `median_mean_diff`

**Other:**
- `tail_weight_ratio`: Ratio of extreme values
- `trimmed_mean`: Mean after removing outliers
- `n_samples`: Number of similarity values used

## Loading the Data

### Python

```python
import json
import numpy as np

# Load a single dataset
with open("msclap/single_voice/real.json") as f:
    real_samples = json.load(f)

# Access features for classification
for sample in real_samples:
    features = sample["features"]
    label = sample["label"]
    embeddings = np.array(sample["embeddings"])  # Shape: (n_segments, embedding_dim)
    similarities = sample["similarities"]

    # Use features for training
    feature_vector = [
        features["mean"],
        features["std"],
        features["entropy"],
        # ... etc
    ]
```

### Pandas

```python
import pandas as pd
import json

with open("msclap/single_voice/real.json") as f:
    data = json.load(f)

# Flatten features into DataFrame
records = []
for item in data:
    record = item["features"].copy()
    record["label"] = item["label"]
    record["duration"] = item["duration"]
    record["n_embeddings"] = len(item["embeddings"])
    records.append(record)

df = pd.DataFrame(records)
print(df.describe())
```

## Embedding Types

| Type | Model | Dimension | Segmentation | Description |
|------|-------|-----------|--------------|-------------|
| `msclap` | MS-CLAP | 1024 | 2.0s / 1.0s hop | Primary model used in paper |
| `msclap_optc` | MS-CLAP | 1024 | 1.5s / 0.75s hop | Option C: shorter segments for short audio files |
| `wavlm` | WavLM-Large | 1024 | 2.0s / 1.0s hop | Speech-focused self-supervised model |
| `laion_clap` | LAION-CLAP | 512 | 2.0s / 1.0s hop | General audio embeddings |

### Option C Segmentation

The `msclap_optc` configuration uses shorter segment windows (1.5s instead of 2.0s)
and smaller hop size (0.75s instead of 1.0s). This "Option C" setting was developed
to handle short-duration audio files (e.g., DeepSpeak v2 fake samples which average
3.78 seconds). With standard 2.0s segments and minimum 3 segments requirement, files
shorter than 4 seconds would be excluded. Option C reduces the minimum duration to
2.25 seconds, allowing 89% of DeepSpeak fake samples to be processed vs only 42%
with standard settings.

## Dataset Sources

| Dataset | Domain | Real Source | Fake Source |
|---------|--------|-------------|-------------|
| `single_voice` | Speech | LibriSpeech audiobooks | ASVspoof 2019 LA (TTS/VC systems) |
| `deepspeak_v2` | Speech | DeepSpeak video audio | ElevenLabs, PlayHT, Speechify voice cloning |
| `mlaad` | Speech | M-AILABS audiobooks | 84 different TTS models |
| `fakeavceleb` | Speech | VoxCeleb celebrity clips | SV2TTS voice cloning |
| `audeter` | Speech | In-the-wild YouTube audio | Modern TTS systems (2024-2025) |
| `music_instrumental` | Music | MUSDB18 instrumental stems | FakeMusicCaps (5 AI music models) |
| `music_with_vocals` | Music | MUSDB18 full mix | FakeMusicCaps (5 AI music models) |

## Sample Statistics

"""

    # Add statistics tables for each embedding type
    for embed_type, stats in all_stats.items():
        readme_content += f"\n### {embed_type}\n\n"
        readme_content += f"**{stats.get('description', '')}**\n\n"
        readme_content += f"- Embedding dimension: {stats.get('embedding_dim', 'N/A')}\n"
        readme_content += f"- Segment duration: {stats.get('segment_duration', 'N/A')}s\n"
        readme_content += f"- Segment hop: {stats.get('segment_hop', 'N/A')}s\n\n"

        readme_content += "| Dataset | Original Real | Original Fake | Sampled Real | Sampled Fake |\n"
        readme_content += "|---------|---------------|---------------|--------------|---------------|\n"

        for dataset_name, ds_stats in stats.get("datasets", {}).items():
            readme_content += (
                f"| {dataset_name} | "
                f"{ds_stats['original_real']:,} | "
                f"{ds_stats['original_fake']:,} | "
                f"{ds_stats['sampled_real']:,} | "
                f"{ds_stats['sampled_fake']:,} |\n"
            )

    readme_content += """
## Citation

If you use this data in your research, please cite:

```bibtex
@article{audiodf2026,
  title={Audio Deepfake Detection via Temporal Coherence Analysis of CLAP Embeddings},
  author={},
  year={2026}
}
```

## License

This sample data is provided for research purposes only. The original datasets
have their own licenses which should be respected.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample training data for external reviewers"
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=5,
        help="Percentage of data to sample (default: 5)"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        choices=list(EMBEDDING_CONFIGS.keys()),
        help="Generate samples for specific embedding type only"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )

    args = parser.parse_args()

    print(f"Generating {args.percent}% sample of training data")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print()

    # Clean output directory
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which embedding types to process
    if args.embedding:
        embed_types = [args.embedding]
    else:
        embed_types = list(EMBEDDING_CONFIGS.keys())

    all_stats = {}

    for embed_type in embed_types:
        print(f"\n{'='*70}")
        print(f"Processing {embed_type}")
        print(f"{'='*70}")
        stats = generate_samples(
            embed_type,
            args.percent,
            args.output_dir,
            seed=args.seed,
        )
        if stats:
            all_stats[embed_type] = stats

    # Generate README
    print("\nGenerating README.md...")
    generate_readme(args.output_dir, all_stats, args.percent)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_samples = 0
    for embed_type, stats in all_stats.items():
        embed_total = sum(
            ds["sampled_real"] + ds["sampled_fake"]
            for ds in stats.get("datasets", {}).values()
        )
        total_samples += embed_total
        print(f"{embed_type}: {embed_total:,} samples")

    print(f"\nTotal: {total_samples:,} samples")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
