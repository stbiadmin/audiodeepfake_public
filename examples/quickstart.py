"""Minimal quick-start for the audio deepfake detector.

Runs detection on the demo clips shipped under data/sound-samples/. Loads the
default speech model, prints the predicted label and confidence for each clip.
"""

from pathlib import Path

from inference import AudioDeepfakeDetector


def main() -> None:
    samples_dir = Path(__file__).resolve().parent.parent / "data" / "sound-samples"
    audio_paths = sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".flac"})

    if not audio_paths:
        raise SystemExit(f"No audio files found under {samples_dir}")

    detector = AudioDeepfakeDetector()
    detector.warmup()

    for path in audio_paths:
        result = detector.detect(str(path))
        print(f"{path.name:40s}  {result['label']:5s}  conf={result['confidence']:.3f}  model={result['model']}")


if __name__ == "__main__":
    main()
