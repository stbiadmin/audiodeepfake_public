"""Smoke tests covering the import surface and the feature-extraction pipeline.

These tests do not require any downloaded datasets. They exercise the segment
-> similarity -> feature path on the demo clips shipped in
data/sound-samples/.
"""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_core():
    from core.feature_extractor import DistributionFeatureExtractor  # noqa: F401
    from core.segmenter import AudioSegmenter  # noqa: F401
    from core.similarity_computer import SimilarityComputer  # noqa: F401


def test_import_inference():
    from inference import detect, AudioDeepfakeDetector  # noqa: F401


def test_all_features_count():
    from training.data_loader import ALL_FEATURES

    assert len(ALL_FEATURES) == 29, (
        f"Expected 29 statistical features (per the paper); found {len(ALL_FEATURES)}"
    )


def test_feature_extraction_on_random_similarities():
    from core.feature_extractor import DistributionFeatureExtractor

    rng = np.random.default_rng(0)
    similarities = rng.uniform(-1, 1, size=200)
    extractor = DistributionFeatureExtractor()
    features = extractor.extract_all_features(similarities)

    assert isinstance(features, dict)
    assert "entropy" in features
    assert "mean" in features
    assert np.isfinite(features["mean"])


def test_demo_audio_present():
    samples_dir = REPO_ROOT / "data" / "sound-samples"
    audio = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
    assert len(audio) > 0, f"No demo audio under {samples_dir}"


@pytest.mark.slow
def test_end_to_end_detect_on_sample():
    from inference import detect

    samples_dir = REPO_ROOT / "data" / "sound-samples"
    target = next(samples_dir.glob("*.wav"), None)
    if target is None:
        pytest.skip("No .wav demo audio available")

    result = detect(str(target))
    assert result["label"] in {"real", "fake"}
    assert 0.0 <= result["confidence"] <= 1.0
