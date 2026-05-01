#!/usr/bin/env python3
"""Download evaluation datasets for generalization testing.

Downloads external datasets that were NOT used during training to test
model generalization and robustness in the wild.

Datasets:
    - Deepfake-Eval-2024: In-the-wild deepfakes from social media
    - ASVspoof 5: 2024 challenge with latest TTS generators
    - SONICS: AI-generated music (Suno, Udio)
    - MLAAD: Multilingual audio anti-spoofing
    - In-the-Wild: Celebrity/politician deepfakes

Usage:
    python scripts/download_eval_datasets.py --dataset deepfake_eval_2024
    python scripts/download_eval_datasets.py --all
    python scripts/download_eval_datasets.py --list
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Dataset configurations
EVAL_DATASETS = {
    'deepfake_eval_2024': {
        'name': 'Deepfake-Eval-2024',
        'source': 'huggingface',
        'repo': 'nuriachandra/Deepfake-Eval-2024',
        'subset': 'audio',
        'license': 'CC BY-SA 4.0',
        'size_gb': 15,
        'description': '56.5h in-the-wild audio from social media, 52 languages',
        'priority': 'HIGH',
    },
    'asvspoof5': {
        'name': 'ASVspoof 5',
        'source': 'huggingface',
        'repo': 'jungjee/asvspoof5',
        'license': 'Research only',
        'size_gb': 20,
        'description': '2024 challenge, ~2000 speakers, 32 attack algorithms',
        'priority': 'HIGH',
    },
    'sonics': {
        'name': 'SONICS',
        'source': 'huggingface',
        'repo': 'awsaf49/sonics',
        'subset': 'test',
        'license': 'Research',
        'size_gb': 40,
        'description': '97k songs (4,751h), Suno/Udio AI music',
        'priority': 'HIGH',
    },
    'mlaad': {
        'name': 'MLAAD v8',
        'source': 'huggingface',
        'repo': 'mueller91/MLAAD',
        'license': 'CC-BY-NC 4.0',
        'size_gb': 25,
        'description': '570h synthetic speech, 40 languages, 119 TTS models',
        'priority': 'MEDIUM',
    },
    'in_the_wild': {
        'name': 'In-the-Wild',
        'source': 'website',
        'url': 'https://deepfake-total.com/in_the_wild',
        'license': 'Research',
        'size_gb': 5,
        'description': '38h real/spoofed celebrity audio',
        'priority': 'MEDIUM',
        'notes': 'Requires manual download from website',
    },
}


def list_datasets():
    """Print available datasets and their details."""
    print("\n" + "="*70)
    print("Available Evaluation Datasets")
    print("="*70)

    total_size = 0
    for key, info in EVAL_DATASETS.items():
        print(f"\n{info['name']} ({key})")
        print(f"  Priority: {info['priority']}")
        print(f"  Size: ~{info['size_gb']}GB")
        print(f"  Source: {info['source']}")
        print(f"  License: {info['license']}")
        print(f"  Description: {info['description']}")
        if 'notes' in info:
            print(f"  Notes: {info['notes']}")
        total_size += info['size_gb']

    print(f"\n{'='*70}")
    print(f"Total size (all datasets): ~{total_size}GB")
    print("="*70)


def download_huggingface_dataset(
    repo: str,
    output_dir: Path,
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> bool:
    """Download dataset from HuggingFace.

    Args:
        repo: HuggingFace repo ID (e.g., 'nuriachandra/Deepfake-Eval-2024')
        output_dir: Directory to save dataset
        subset: Optional subset/split to download
        max_samples: Optional limit on number of samples

    Returns:
        True if successful, False otherwise
    """
    try:
        from datasets import load_dataset

        print(f"Downloading from HuggingFace: {repo}")
        print(f"Output directory: {output_dir}")

        # Load dataset
        if subset:
            print(f"Loading subset: {subset}")
            ds = load_dataset(repo, subset, trust_remote_code=True)
        else:
            ds = load_dataset(repo, trust_remote_code=True)

        # Save to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_dir))

        # Save metadata
        metadata = {
            'repo': repo,
            'subset': subset,
            'num_samples': len(ds) if hasattr(ds, '__len__') else 'unknown',
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"Error downloading {repo}: {e}")
        return False


def download_deepfake_eval_2024(output_dir: Path) -> bool:
    """Download Deepfake-Eval-2024 dataset."""
    print("\n" + "="*60)
    print("Downloading Deepfake-Eval-2024")
    print("="*60)

    try:
        from datasets import load_dataset

        # This dataset has audio split
        print("Loading audio subset from HuggingFace...")
        ds = load_dataset(
            'nuriachandra/Deepfake-Eval-2024',
            trust_remote_code=True,
        )

        # Create output structure
        audio_dir = output_dir / 'deepfake_eval_2024'
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        ds.save_to_disk(str(audio_dir / 'data'))

        # Save metadata
        metadata = {
            'name': 'Deepfake-Eval-2024',
            'source': 'huggingface',
            'repo': 'nuriachandra/Deepfake-Eval-2024',
            'license': 'CC BY-SA 4.0',
        }
        with open(audio_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to: {audio_dir}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024")
        print("2. Accept the license agreement")
        print("3. Download the audio subset")
        print(f"4. Extract to: {output_dir / 'deepfake_eval_2024'}")
        return False


def download_asvspoof5(output_dir: Path) -> bool:
    """Download ASVspoof 5 dataset from HuggingFace."""
    print("\n" + "="*60)
    print("Downloading ASVspoof 5")
    print("="*60)

    try:
        from datasets import load_dataset

        asvspoof_dir = output_dir / 'asvspoof5'
        asvspoof_dir.mkdir(parents=True, exist_ok=True)

        print("Loading ASVspoof5 from HuggingFace (jungjee/asvspoof5)...")
        ds = load_dataset(
            'jungjee/asvspoof5',
            trust_remote_code=True,
        )

        # Save dataset
        ds.save_to_disk(str(asvspoof_dir / 'data'))

        # Save metadata
        metadata = {
            'name': 'ASVspoof 5',
            'source': 'huggingface',
            'repo': 'jungjee/asvspoof5',
            'license': 'Research only',
        }
        with open(asvspoof_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to: {asvspoof_dir}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/datasets/jungjee/asvspoof5")
        print("2. Accept license agreement if required")
        print(f"3. Download and extract to: {output_dir / 'asvspoof5'}")
        return False


def download_sonics(output_dir: Path, max_samples: Optional[int] = None) -> bool:
    """Download SONICS dataset using snapshot_download."""
    print("\n" + "="*60)
    print("Downloading SONICS")
    print("="*60)

    try:
        from huggingface_hub import snapshot_download

        sonics_dir = output_dir / 'sonics'
        sonics_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading SONICS from HuggingFace using snapshot_download...")
        print("Note: This is a large dataset (~40GB).")

        # Use snapshot_download for more reliable download
        snapshot_download(
            repo_id="awsaf49/sonics",
            repo_type="dataset",
            local_dir=str(sonics_dir),
        )

        # Save metadata
        metadata = {
            'name': 'SONICS',
            'source': 'huggingface',
            'repo': 'awsaf49/sonics',
            'download_method': 'snapshot_download',
        }
        with open(sonics_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to: {sonics_dir}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        print("\nManual download instructions:")
        print("1. pip install huggingface_hub")
        print("2. Run:")
        print("   from huggingface_hub import snapshot_download")
        print(f"   snapshot_download(repo_id='awsaf49/sonics', repo_type='dataset', local_dir='{sonics_dir}')")
        return False


def download_mlaad(output_dir: Path) -> bool:
    """Download MLAAD dataset."""
    print("\n" + "="*60)
    print("Downloading MLAAD")
    print("="*60)

    try:
        from datasets import load_dataset

        mlaad_dir = output_dir / 'mlaad'
        mlaad_dir.mkdir(parents=True, exist_ok=True)

        print("Loading MLAAD from HuggingFace...")
        print("Note: You may need to accept the license agreement first.")

        ds = load_dataset(
            'mueller91/MLAAD',
            trust_remote_code=True,
        )

        # Save
        ds.save_to_disk(str(mlaad_dir / 'data'))

        # Save metadata
        metadata = {
            'name': 'MLAAD v8',
            'source': 'huggingface',
            'repo': 'mueller91/MLAAD',
            'license': 'CC-BY-NC 4.0',
        }
        with open(mlaad_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to: {mlaad_dir}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/datasets/mueller91/MLAAD")
        print("2. Accept the CC-BY-NC 4.0 license")
        print("3. Download the dataset")
        print(f"4. Extract to: {output_dir / 'mlaad'}")
        return False


def download_in_the_wild(output_dir: Path) -> bool:
    """Download In-the-Wild dataset."""
    print("\n" + "="*60)
    print("In-the-Wild Dataset")
    print("="*60)

    itw_dir = output_dir / 'in_the_wild'
    itw_dir.mkdir(parents=True, exist_ok=True)

    print("\nIn-the-Wild requires manual download:")
    print("1. Visit: https://deepfake-total.com/in_the_wild")
    print("2. Download the audio files")
    print(f"3. Extract to: {itw_dir}")
    print("\nExpected structure:")
    print(f"  {itw_dir}/")
    print("    politicians/")
    print("      real/")
    print("      fake/")

    # Create placeholder
    with open(itw_dir / 'README.txt', 'w') as f:
        f.write("In-the-Wild Dataset\n")
        f.write("===================\n\n")
        f.write("Please download manually from:\n")
        f.write("https://deepfake-total.com/in_the_wild\n")

    return False


def download_dataset(
    dataset_key: str,
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> bool:
    """Download a single dataset.

    Args:
        dataset_key: Dataset identifier
        output_dir: Base output directory
        max_samples: Optional limit on samples (for testing)

    Returns:
        True if successful
    """
    if dataset_key not in EVAL_DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(EVAL_DATASETS.keys())}")
        return False

    info = EVAL_DATASETS[dataset_key]
    print(f"\nDownloading: {info['name']}")
    print(f"Size: ~{info['size_gb']}GB")
    print(f"License: {info['license']}")

    if dataset_key == 'deepfake_eval_2024':
        return download_deepfake_eval_2024(output_dir)
    elif dataset_key == 'asvspoof5':
        return download_asvspoof5(output_dir)
    elif dataset_key == 'sonics':
        return download_sonics(output_dir, max_samples)
    elif dataset_key == 'mlaad':
        return download_mlaad(output_dir)
    elif dataset_key == 'in_the_wild':
        return download_in_the_wild(output_dir)
    else:
        print(f"Download not implemented for: {dataset_key}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download evaluation datasets for generalization testing'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=list(EVAL_DATASETS.keys()),
        help='Dataset to download'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets'
    )
    parser.add_argument(
        '--tier1',
        action='store_true',
        help='Download Tier 1 datasets only (recommended ~45GB)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data/eval_datasets'),
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit samples (for testing)'
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset and not args.all and not args.tier1:
        parser.print_help()
        print("\nUse --list to see available datasets")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Determine which datasets to download
    if args.all:
        datasets = list(EVAL_DATASETS.keys())
    elif args.tier1:
        datasets = ['deepfake_eval_2024', 'asvspoof5', 'sonics']
    else:
        datasets = [args.dataset]

    # Download
    results = {}
    for ds in datasets:
        success = download_dataset(ds, args.output_dir, args.max_samples)
        results[ds] = success

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for ds, success in results.items():
        status = "SUCCESS" if success else "MANUAL DOWNLOAD REQUIRED"
        print(f"  {ds}: {status}")


if __name__ == '__main__':
    main()
