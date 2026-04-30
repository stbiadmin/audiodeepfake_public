#!/usr/bin/env python3
"""Experiment 1A: Direct classification with pre-trained AASIST models.

Evaluates AASIST, AASIST-L, and AASIST3 on ITW and ASVspoof5 eval datasets.
No training - purely inference with published pre-trained weights.

Models:
    AASIST/AASIST-L: clovaai/aasist (GitHub), trained on ASVspoof 2019 LA
    AASIST3: MTUCI/AASIST3 (HuggingFace), trained on ASVspoof 2024 + MLAAD

Requirements:
    pip install einops transformers

Setup:
    git clone https://github.com/clovaai/aasist.git vendor/aasist

Usage:
    python sota_benchmarking/run_aasist_baseline.py
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DATA_DIR = PROJECT_ROOT / 'sota_benchmarking' / 'eval_data'
OUTPUT_DIR = PROJECT_ROOT / 'sota_benchmarking' / 'results' / 'aasist_baseline'

# AASIST expects 16kHz, 64600 samples (~4.04 seconds)
TARGET_SR = 16000
TARGET_LENGTH = 64600

EVAL_DATASETS = ['in_the_wild', 'asvspoof5']


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = float('nan')
    return float(eer)


def load_eval_data(dataset_name):
    """Load eval dataset file paths and labels from JSON files."""
    ds_dir = EVAL_DATA_DIR / dataset_name
    real_path = ds_dir / 'real.json'
    fake_path = ds_dir / 'fake.json'

    entries = []
    for label, path in [(0, real_path), (1, fake_path)]:
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for entry in data:
            fp = entry.get('file_path', '')
            if fp:
                # Resolve relative paths from project root
                full_path = PROJECT_ROOT / fp if not Path(fp).is_absolute() else Path(fp)
                entries.append({'file_path': str(full_path), 'label': label})

    return entries


def load_audio(file_path, target_sr=TARGET_SR, target_length=TARGET_LENGTH):
    """Load audio, resample to 16kHz, tile-pad/truncate to target_length.

    Uses tile-padding (repeating audio) to match AASIST's training convention
    from data_utils.pad(), rather than zero-padding.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        return None

    # Pad or truncate (matching AASIST's data_utils.pad method)
    if len(audio) < target_length:
        num_repeats = int(target_length / len(audio)) + 1
        audio = np.tile(audio, num_repeats)[:target_length]
    else:
        audio = audio[:target_length]

    return torch.FloatTensor(audio)


# AASIST Model Loaders
def load_aasist_model(variant='AASIST'):
    """Load AASIST or AASIST-L from vendor/aasist/.

    Args:
        variant: 'AASIST' or 'AASIST-L'

    Returns:
        model, config_name
    """
    aasist_dir = PROJECT_ROOT / 'vendor' / 'aasist'
    if not aasist_dir.exists():
        raise FileNotFoundError(
            f"AASIST repo not found at {aasist_dir}. "
            f"Run: git clone https://github.com/clovaai/aasist.git {aasist_dir}"
        )

    # Import AASIST model directly via importlib to avoid conflict with project's models/ package
    import importlib.util
    spec = importlib.util.spec_from_file_location("AASIST", str(aasist_dir / "models" / "AASIST.py"))
    aasist_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aasist_module)
    AASISTModel = aasist_module.Model

    # Load config
    if variant == 'AASIST-L':
        config_path = aasist_dir / 'config' / 'AASIST-L.conf'
        weight_path = aasist_dir / 'models' / 'weights' / 'AASIST-L.pth'
    else:
        config_path = aasist_dir / 'config' / 'AASIST.conf'
        weight_path = aasist_dir / 'models' / 'weights' / 'AASIST.pth'

    if not weight_path.exists():
        raise FileNotFoundError(f"Weights not found: {weight_path}")

    # Parse config
    import json as _json
    with open(config_path, 'r') as f:
        config = _json.load(f)

    model_config = config['model_config']
    model = AASISTModel(model_config)

    # Load weights
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_aasist3_model():
    """Load AASIST3 from HuggingFace (MTUCI/AASIST3)."""
    try:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    model_name = "MTUCI/AASIST3"
    print(f"  Loading {model_name} from HuggingFace...")

    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()

    return model


# Inference
def run_aasist_inference(model, entries, batch_size=32, device='cpu'):
    """Run AASIST/AASIST-L inference on eval entries.

    Returns:
        scores: np.array of spoof probabilities
        labels: np.array of true labels
        valid_mask: indices of successfully processed entries
    """
    model = model.to(device)
    scores = []
    labels = []

    for i, entry in enumerate(entries):
        if i % 500 == 0:
            print(f"    Processing {i}/{len(entries)}...")

        audio = load_audio(entry['file_path'])
        if audio is None:
            continue

        with torch.no_grad():
            audio = audio.unsqueeze(0).to(device)  # (1, T)
            result = model(audio)
            # AASIST returns (last_hidden, output) where output is (batch, 2)
            # AASIST labels: 0=spoof, 1=bonafide (from data_utils.py)
            output = result[1] if isinstance(result, tuple) else result
            probs = torch.softmax(output, dim=1)
            spoof_prob = probs[0, 0].item()  # index 0 = spoof class

        scores.append(spoof_prob)
        labels.append(entry['label'])

    return np.array(scores), np.array(labels)


def run_aasist3_inference(model, entries, device='cpu'):
    """Run AASIST3 inference on eval entries."""
    model = model.to(device)
    scores = []
    labels = []

    for i, entry in enumerate(entries):
        if i % 500 == 0:
            print(f"    Processing {i}/{len(entries)}...")

        audio = load_audio(entry['file_path'])
        if audio is None:
            continue

        with torch.no_grad():
            audio = audio.unsqueeze(0).to(device)
            output = model(audio)
            logits = output.logits if hasattr(output, 'logits') else output
            probs = torch.softmax(logits, dim=1)
            # AASIST3 label mapping may differ - check model config
            # Typically: 0=bonafide, 1=spoof
            spoof_prob = probs[0, 1].item()

        scores.append(spoof_prob)
        labels.append(entry['label'])

    return np.array(scores), np.array(labels)


def evaluate_scores(y_true, y_scores, threshold=0.5):
    """Compute all metrics from true labels and spoof probability scores."""
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0,
        'eer': compute_eer(y_true, y_scores),
    }
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AASIST baseline evaluation")
    parser.add_argument("--models", nargs='+',
                        default=['AASIST', 'AASIST-L', 'AASIST3'],
                        choices=['AASIST', 'AASIST-L', 'AASIST3'],
                        help="Which models to evaluate")
    parser.add_argument("--datasets", nargs='+', default=EVAL_DATASETS,
                        help="Which eval datasets to use")
    parser.add_argument("--device", default='cpu',
                        help="Device for inference (cpu, mps, cuda)")
    args = parser.parse_args()

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'

    print("=" * 70)
    print("Experiment 1A: AASIST Baseline Direct Classification")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name in args.models:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        # Load model
        try:
            if model_name == 'AASIST3':
                model = load_aasist3_model()
            else:
                model = load_aasist_model(variant=model_name)
            print(f"  Model loaded successfully.")
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}): {e}")
            continue

        # Evaluate on each dataset
        for dataset_name in args.datasets:
            print(f"\n  Dataset: {dataset_name}")
            entries = load_eval_data(dataset_name)
            if not entries:
                print(f"    No data found, skipping.")
                continue

            n_real = sum(1 for e in entries if e['label'] == 0)
            n_fake = sum(1 for e in entries if e['label'] == 1)
            print(f"    Samples: {len(entries)} (real={n_real}, fake={n_fake})")

            # Run inference
            if model_name == 'AASIST3':
                scores, labels = run_aasist3_inference(model, entries, device=device)
            else:
                scores, labels = run_aasist_inference(model, entries, device=device)

            print(f"    Successfully processed: {len(scores)}/{len(entries)}")

            if len(scores) == 0:
                continue

            # Compute metrics
            metrics = evaluate_scores(labels, scores)
            print(f"    F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  "
                  f"EER={metrics['eer']:.3f}  Acc={metrics['accuracy']:.3f}")

            all_results.append({
                'dataset': dataset_name,
                'model': model_name,
                'n_samples': len(scores),
                **metrics,
            })

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("SUMMARY: AASIST Baseline Results")
        print(f"{'=' * 70}")

        print(f"\n{'Model':<15} {'Dataset':<15} {'N':>6} {'F1':>7} {'AUC':>7} {'EER':>7}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['model']:<15} {r['dataset']:<15} {r['n_samples']:>6} "
                  f"{r['f1']:>7.3f} {r['auc']:>7.3f} {r['eer']:>7.3f}")

        df_results = pd.DataFrame(all_results)
        csv_path = OUTPUT_DIR / 'aasist_baseline_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo results generated. Check model setup instructions above.")


if __name__ == '__main__':
    main()
