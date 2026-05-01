#!/usr/bin/env python3
"""Experiment 2A: Direct classification with pre-trained SSL-based detectors.

Evaluates off-the-shelf wav2vec2/SSL anti-spoofing models on eval datasets.
No training - purely inference with published pre-trained weights.

Models:
    wav2vec2-XLSR: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (HF)
    SSL_Anti-spoofing: TakHemlata/SSL_Anti-spoofing (GitHub)

Usage:
    python sota_benchmarking/run_ssl_baselines.py
    python sota_benchmarking/run_ssl_baselines.py --models wav2vec2-XLSR
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DATA_DIR = PROJECT_ROOT / 'sota_benchmarking' / 'eval_data'
OUTPUT_DIR = PROJECT_ROOT / 'sota_benchmarking' / 'results' / 'ssl_baselines'

TARGET_SR = 16000
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
    """Load eval dataset file paths and labels."""
    ds_dir = EVAL_DATA_DIR / dataset_name
    entries = []
    for label, fname in [(0, 'real.json'), (1, 'fake.json')]:
        path = ds_dir / fname
        if not path.exists():
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        for entry in data:
            fp = entry.get('file_path', '')
            if fp:
                full_path = PROJECT_ROOT / fp if not Path(fp).is_absolute() else Path(fp)
                entries.append({'file_path': str(full_path), 'label': label})
    return entries


def load_audio_np(file_path, target_sr=TARGET_SR):
    """Load audio as numpy array at target sample rate."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception:
        return None


def evaluate_scores(y_true, y_scores, threshold=0.5):
    """Compute metrics from labels and spoof probability scores."""
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


# wav2vec2-XLSR (HuggingFace pipeline)
class Wav2Vec2XLSRDetector:
    """Gustking/wav2vec2-large-xlsr-deepfake-audio-classification via HF."""

    MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

    def __init__(self, device='cpu'):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        print(f"  Loading {self.MODEL_ID}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.device = device
        self.model.to(device)

        # Check label mapping
        self.label2id = self.model.config.label2id
        print(f"    Label mapping: {self.label2id}")

    def predict(self, audio_np, sr=TARGET_SR):
        """Return spoof probability for audio."""
        inputs = self.feature_extractor(
            audio_np, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Map label to spoof probability
        # Common mappings: 'fake'/'spoof' = 1, 'real'/'bonafide' = 0
        if 'fake' in self.label2id:
            spoof_idx = self.label2id['fake']
        elif 'spoof' in self.label2id:
            spoof_idx = self.label2id['spoof']
        elif '1' in self.label2id:
            spoof_idx = self.label2id['1']
        else:
            # Default: assume index 1 is spoof
            spoof_idx = 1

        return probs[0, spoof_idx].item()


# SSL Anti-Spoofing (TakHemlata)
class SSLAntiSpoofDetector:
    """TakHemlata/SSL_Anti-spoofing: wav2vec 2.0 + classifier.

    Requires cloning repo and downloading weights:
        git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git vendor/ssl_antispoofing
        Download weights to vendor/ssl_antispoofing/Best_LA_model_for_DF.pth
    """

    def __init__(self, device='cpu'):
        vendor_dir = PROJECT_ROOT / 'vendor' / 'ssl_antispoofing'
        if not vendor_dir.exists():
            raise FileNotFoundError(
                f"SSL_Anti-spoofing repo not found at {vendor_dir}. "
                f"Run: git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git {vendor_dir}"
            )

        sys.path.insert(0, str(vendor_dir))

        # Import model architecture
        from model import Model as SSLModel

        self.device = device

        # Load model
        weight_path = vendor_dir / 'Best_LA_model_for_DF.pth'
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weight_path}. "
                f"Download from the SSL_Anti-spoofing repo's Google Drive link."
            )

        self.model = SSLModel(device=device)
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        print("  Loaded SSL_Anti-spoofing model.")

    def predict(self, audio_np, sr=TARGET_SR):
        """Return spoof probability for audio."""
        audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(audio_tensor)
            probs = torch.softmax(output, dim=1)
            # SSL_Anti-spoofing uses ASVspoof convention: spoof=0, bonafide=1
            return probs[0, 0].item()  # index 0 = spoof class


# Main
AVAILABLE_MODELS = {
    'wav2vec2-XLSR': Wav2Vec2XLSRDetector,
    'SSL_Anti-spoofing': SSLAntiSpoofDetector,
}


def run_model_eval(detector, entries, model_name):
    """Run inference for all entries, return scores and labels."""
    scores = []
    labels = []
    errors = 0

    for i, entry in enumerate(entries):
        if i % 500 == 0:
            print(f"    Processing {i}/{len(entries)}...")

        audio = load_audio_np(entry['file_path'])
        if audio is None:
            errors += 1
            continue

        try:
            score = detector.predict(audio)
            scores.append(score)
            labels.append(entry['label'])
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error on {entry['file_path']}: {e}")

    if errors > 0:
        print(f"    {errors} files failed to process.")

    return np.array(scores), np.array(labels)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SSL baseline evaluation")
    parser.add_argument("--models", nargs='+',
                        default=list(AVAILABLE_MODELS.keys()),
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Which models to evaluate")
    parser.add_argument("--datasets", nargs='+', default=EVAL_DATASETS)
    parser.add_argument("--device", default='cpu')
    args = parser.parse_args()

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    print("=" * 70)
    print("Experiment 2A: SSL-Based Baseline Direct Classification")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name in args.models:
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        # Load model
        try:
            detector_cls = AVAILABLE_MODELS[model_name]
            detector = detector_cls(device=device)
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}): {e}")
            continue

        # Evaluate
        for dataset_name in args.datasets:
            print(f"\n  Dataset: {dataset_name}")
            entries = load_eval_data(dataset_name)
            if not entries:
                print("    No data found, skipping.")
                continue

            n_real = sum(1 for e in entries if e['label'] == 0)
            n_fake = sum(1 for e in entries if e['label'] == 1)
            print(f"    Samples: {len(entries)} (real={n_real}, fake={n_fake})")

            scores, labels = run_model_eval(detector, entries, model_name)
            print(f"    Successfully processed: {len(scores)}/{len(entries)}")

            if len(scores) == 0:
                continue

            metrics = evaluate_scores(labels, scores)
            print(f"    F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}  "
                  f"EER={metrics['eer']:.3f}  Acc={metrics['accuracy']:.3f}")

            all_results.append({
                'dataset': dataset_name,
                'model': model_name,
                'n_samples': len(scores),
                **metrics,
            })

        # Cleanup
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("SUMMARY: SSL Baseline Results")
        print(f"{'=' * 70}")

        print(f"\n{'Model':<25} {'Dataset':<15} {'N':>6} {'F1':>7} {'AUC':>7} {'EER':>7}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['model']:<25} {r['dataset']:<15} {r['n_samples']:>6} "
                  f"{r['f1']:>7.3f} {r['auc']:>7.3f} {r['eer']:>7.3f}")

        df_results = pd.DataFrame(all_results)
        csv_path = OUTPUT_DIR / 'ssl_baseline_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo results generated. Check model setup instructions above.")


if __name__ == '__main__':
    main()
