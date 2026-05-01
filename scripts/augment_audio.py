"""Audio augmentation script to expand real sample count.

Creates augmented versions of real audio files using conservative parameters
based on best practices from audio ML research:
- Pitch shifting: ±1, ±2 semitones (most common range for speech/music)
- Time stretching: 0.95x, 1.05x (mild tempo changes)
- Noise injection: 20-30 dB SNR (speech robustness)
- Codec compression: MP3 128kbps (real-world conditions)
- Reverb: Room simulation (domain variability)
- NO combined transformations (to avoid unrealistic artifacts)

References:
- audiomentations library: ±4 semitones max, 0.8-1.25x time stretch
- Speech recognition research: ±2 semitones is most common
- Best practice: 5-10x augmentation ratio, not 30x+
- "A study on data augmentation in voice anti-spoofing" (2022): noise/codec/reverb

Usage:
    python scripts/augment_audio.py data/experiments/music_instrumental/real \
        --output-dir data/experiments/music_instrumental/real_augmented

    # For speech with all augmentation types:
    python scripts/augment_audio.py data/experiments/single_voice/real \
        --output-dir data/experiments/single_voice/real_augmented \
        --mode speech
"""

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm

# Conservative augmentation parameters (based on research best practices)
PITCH_SHIFTS = [-2, -1, 1, 2]  # semitones (±2 is standard for speech)
TIME_STRETCHES = [0.95, 1.05]  # speed multipliers (mild changes only)
# NO combined transformations - research shows this can create artifacts

# Speech-specific augmentations (v2.2) for domain generalization
# Based on: "A study on data augmentation in voice anti-spoofing" (2022)
SPEECH_AUGMENTATIONS = [
    ('noise_20', 'noise', 20, None),    # 20 dB SNR - moderate noise
    ('noise_30', 'noise', 30, None),    # 30 dB SNR - light noise
    ('mp3_128', 'codec', 'mp3', 128),   # MP3 128kbps compression
    ('reverb_small', 'reverb', 0.3, None),  # Small room reverb
]


def add_noise(y, snr_db):
    """Add Gaussian noise at specified SNR level.

    Args:
        y: Audio signal
        snr_db: Signal-to-noise ratio in decibels

    Returns:
        Noisy audio signal
    """
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return y + noise


def apply_codec(y, sr, codec='mp3', bitrate=128):
    """Apply codec compression using ffmpeg.

    Args:
        y: Audio signal
        sr: Sample rate
        codec: Codec type ('mp3', 'ogg', 'aac')
        bitrate: Bitrate in kbps

    Returns:
        Compressed/decompressed audio signal
    """
    # Check for ffmpeg in local bin directory first
    ffmpeg_path = 'ffmpeg'
    local_ffmpeg = Path(__file__).parent.parent / 'bin' / 'ffmpeg'
    if local_ffmpeg.exists():
        ffmpeg_path = str(local_ffmpeg)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
        sf.write(tmp_in.name, y, sr)
        tmp_in_path = tmp_in.name

    ext = {'mp3': '.mp3', 'ogg': '.ogg', 'aac': '.m4a'}[codec]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_final:
        tmp_final_path = tmp_final.name

    try:
        # Encode to lossy format
        subprocess.run([
            ffmpeg_path, '-y', '-i', tmp_in_path,
            '-b:a', f'{bitrate}k', tmp_out_path
        ], capture_output=True, check=True)

        # Decode back to wav
        subprocess.run([
            ffmpeg_path, '-y', '-i', tmp_out_path, tmp_final_path
        ], capture_output=True, check=True)

        # Load compressed audio
        y_compressed, _ = librosa.load(tmp_final_path, sr=sr, mono=True)
        return y_compressed

    finally:
        # Cleanup temp files
        for f in [tmp_in_path, tmp_out_path, tmp_final_path]:
            try:
                os.unlink(f)
            except:
                pass


def apply_reverb(y, sr, reverb_amount=0.3):
    """Apply simple room reverb using exponential decay impulse response.

    Args:
        y: Audio signal
        sr: Sample rate
        reverb_amount: Reverb intensity (0-1)

    Returns:
        Reverbed audio signal
    """
    # Create simple exponential decay impulse response
    reverb_time = 0.3  # 300ms reverb tail
    ir_length = int(sr * reverb_time)
    ir = np.exp(-np.linspace(0, 5, ir_length))
    ir = ir / np.sum(ir)  # Normalize

    # Apply convolution reverb
    y_reverb = fftconvolve(y, ir, mode='full')[:len(y)]

    # Mix dry and wet signals
    y_mixed = (1 - reverb_amount) * y + reverb_amount * y_reverb

    # Normalize to prevent clipping
    max_val = np.max(np.abs(y_mixed))
    if max_val > 1.0:
        y_mixed = y_mixed / max_val

    return y_mixed


def apply_augmentation(y, sr, aug_type, param1, param2=None):
    """Apply augmentation based on type and parameters.

    Supported augmentation types:
    - pitch: pitch shift by param1 semitones
    - time: time stretch by param1 rate
    - noise: add noise at param1 dB SNR
    - codec: compress with param1 codec at param2 kbps
    - reverb: add reverb with param1 intensity
    """
    if aug_type == 'pitch':
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=param1)
    elif aug_type == 'time':
        return librosa.effects.time_stretch(y, rate=param1)
    elif aug_type == 'noise':
        return add_noise(y, snr_db=param1)
    elif aug_type == 'codec':
        return apply_codec(y, sr, codec=param1, bitrate=param2)
    elif aug_type == 'reverb':
        return apply_reverb(y, sr, reverb_amount=param1)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def augment_file(input_path, output_dir, augmentations):
    """Augment a single file with specified transformations."""
    results = []

    try:
        # Load audio at native sample rate
        y, sr = librosa.load(input_path, sr=None, mono=False)

        # Handle stereo by converting to mono
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        stem = Path(input_path).stem

        for aug_name, aug_type, param1, param2 in augmentations:
            try:
                y_aug = apply_augmentation(y, sr, aug_type, param1, param2)
                output_path = output_dir / f"{stem}_{aug_name}.wav"
                sf.write(output_path, y_aug, sr)
                results.append((str(output_path), True, None))
            except Exception as e:
                results.append((f"{stem}_{aug_name}", False, str(e)))

    except Exception as e:
        results.append((str(input_path), False, str(e)))

    return results


def generate_augmentation_plan(n_original, mode='music'):
    """Generate augmentation list based on mode.

    Args:
        n_original: Number of original files
        mode: 'music' for pitch/time only, 'speech' for all augmentations

    Returns list of tuples: (name, aug_type, param1, param2)

    Music mode (conservative):
    - 4 pitch shifts (±1, ±2 semitones)
    - 2 time stretches (0.95x, 1.05x)
    - Total: 6 augmentations per file (~7x expansion)

    Speech mode (for generalization):
    - 4 pitch shifts + 2 time stretches
    - 2 noise levels (20, 30 dB SNR)
    - 1 codec compression (MP3 128kbps)
    - 1 reverb effect
    - Total: 10 augmentations per file (~11x expansion)
    """
    augmentations = []

    # Single pitch shifts only
    for ps in PITCH_SHIFTS:
        name = f"ps{ps:+d}"
        augmentations.append((name, 'pitch', ps, None))

    # Single time stretches only
    for ts in TIME_STRETCHES:
        name = f"ts{int(ts*100)}"
        augmentations.append((name, 'time', ts, None))

    # Add speech-specific augmentations for domain generalization
    if mode == 'speech':
        augmentations.extend(SPEECH_AUGMENTATIONS)

    n_aug_per_file = len(augmentations)
    expected_total = n_original + n_original * n_aug_per_file

    print(f"  Mode: {mode}")
    print(f"  Original files: {n_original}")
    print(f"  Augmentations per file: {n_aug_per_file}")
    print(f"  Expected total (original + augmented): {expected_total}")
    print(f"  Expansion ratio: {expected_total / n_original:.1f}x")

    return augmentations


def main():
    parser = argparse.ArgumentParser(description='Augment audio files with conservative parameters')
    parser.add_argument('input_dir', type=str, help='Directory containing original audio files')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for augmented files')
    parser.add_argument('--mode', '-m', type=str, choices=['music', 'speech'], default='music',
                        help='Augmentation mode: music (pitch/time only) or speech (all augs)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--extensions', '-e', nargs='+', default=['.wav', '.flac', '.mp3'],
                        help='Audio file extensions to process')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files (follow symlinks)
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))

    # Resolve symlinks to actual files
    audio_files = [f.resolve() if f.is_symlink() else f for f in audio_files]
    audio_files = sorted(set(audio_files))

    n_original = len(audio_files)
    print(f"\nFound {n_original} original audio files")

    if n_original == 0:
        print("No audio files found!")
        return

    # Generate augmentation plan based on mode
    augmentations = generate_augmentation_plan(n_original, mode=args.mode)

    if len(augmentations) == 0:
        print("No augmentations configured!")
        return

    print(f"\nAugmentation types ({len(augmentations)}):")
    for name, aug_type, p1, p2 in augmentations[:10]:
        print(f"  - {name}")
    if len(augmentations) > 10:
        print(f"  ... and {len(augmentations) - 10} more")

    # Process files with ThreadPoolExecutor (avoids pickling issues)
    print(f"\nProcessing {n_original} files with {len(augmentations)} augmentations each...")

    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(augment_file, str(f), output_dir, augmentations): str(f)
            for f in audio_files
        }

        with tqdm(total=len(futures), desc="Augmenting") as pbar:
            for future in as_completed(futures):
                results = future.result()
                for path, success, error in results:
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        if error:
                            tqdm.write(f"Error: {path}: {error}")
                pbar.update(1)

    print("\n=== Augmentation Complete ===")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files (original + augmented): {n_original + success_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
