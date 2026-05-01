# Audio Deepfake Detection Using Temporal Coherence Analysis

This repository contains the code, pretrained classifiers, and reproduction
scripts accompanying the paper
*"Audio Deepfake Detection Using Temporal Coherence Analysis"*
(Norman & Barrington, UC Berkeley, 2026).

The method segments an audio recording into overlapping windows, computes
MS-CLAP embeddings per segment, derives statistical features from the
distribution of pairwise cosine similarities between segments, and trains
gradient-boosted decision trees on those features. A weighted ensemble of five
domain-specific experts is used for speech detection; an unsupervised
percentile classifier is used for cross-domain music detection.

## Headline results

| Setting | Eval set | F1 | AUC | Notes |
|---|---|---|---|---|
| 5-expert speech ensemble | In-the-Wild (n = 12,394) | 0.675 | 0.708 | Weights: see below |
| Single best expert (audeter) | ASVspoof5 (n = 10,000) | 0.776 | 0.869 | |
| Domain-adaptive music classifier | SONICS / FMA cross-domain | 0.938 | 0.976 | No target labels |
| Embedding comparison (best feature, q90) | ASVspoof5 modern TTS | - | 0.949 (MS-CLAP) | vs. 0.633 (WavLM) |

## Installation

```bash
git clone https://github.com/stbiadmin/audiodeepfake_public.git
cd audiodeepfake_public

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For machines without a discrete GPU (laptops, CI), install the CPU-only
torch wheel instead to skip the ~700 MB CUDA download:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Tested with Python 3.10 on macOS (Apple Silicon, MPS) and x86_64 Linux
(CPU and CUDA). No GPU is required for inference; speech ensemble inference
runs at roughly one file per second on CPU and is dominated by the MS-CLAP
embedding step.

To clone the AASIST baseline used in the paper's comparison:

```bash
./setup_baselines.sh
```

## Quick start

The repository ships with a small set of demo audio clips under
`data/sound-samples/` so the pipeline can be exercised without downloading
any datasets.

```python
from inference import detect

result = detect("data/sound-samples/test_clap_short.wav")
print(result)
# {'label': 'real', 'confidence': 0.81, 'model': 'speech'}
```

For multiple files or low-latency inference, instantiate the detector
directly and call `warmup()` once:

```python
from inference import AudioDeepfakeDetector

detector = AudioDeepfakeDetector()
detector.warmup()

for path in ["a.wav", "b.mp3", "c.flac"]:
    print(path, detector.detect(path))
```

## Data acquisition

Datasets are not redistributed in this repository. Each dataset has its own
license; refer to `data/README.md` for download URLs, citations, expected
directory layout, and which datasets require a research-use agreement.
Convenience download scripts are provided in `scripts/download_*.py` for
those datasets that allow direct download.

## Reproducing paper results

The full reproduction recipe (per table and figure) is in
`REPRODUCIBILITY.md`. As a starting point:

```bash
# 5-expert speech ensemble on In-the-Wild
python scripts/evaluate_ensembles.py \
    --eval-dataset in_the_wild \
    --models ds_msclap sv_msclap sv_ds_msclap mlaad_msclap audeter_msclap

# WavLM vs. MS-CLAP supporting comparison (paper Table 3)
python sota_benchmarking/run_ssl_temporal_eval.py

# TreeSHAP feature attribution
python scripts/run_treeshap_analysis.py
```

## Repository layout

```
audiodeepfake_public/
├── config/                Audio-type definitions and base configuration
├── core/                  Segmentation, similarity, statistical features
├── training/              XGBoost trainer, RFE, domain adaptation (CORAL/MMD)
├── inference/             Detector, embedding engine, classifiers
├── evaluation/            Metrics (F1, AUC, EER), ablations, generalization
├── models/                Ensemble code and trained classifier checkpoints
│   └── trained/           5 paper experts + music adaptive (~2 MB)
├── utilities/             CLAP / MS-CLAP / WavLM / wav2vec2 / AASIST wrappers
├── scripts/               CLI entry points (extract, evaluate, ablate, download)
├── sota_benchmarking/     AASIST + SSL baselines + paper MoE cross-eval
├── data/
│   ├── README.md          Dataset acquisition instructions
│   └── sound-samples/     Small demo audio for the quick-start
└── setup_baselines.sh     Fetch external baselines (AASIST) into vendor/
```

## Method summary

1. **Segmentation.** Audio is split into 2-second windows with 50% overlap.
   Files producing fewer than three segments are skipped.
2. **Embedding.** Each segment is embedded with MS-CLAP (1024-dim).
3. **Similarity.** Pairwise cosine similarities are computed across segments.
4. **Statistical features.** 29 descriptors of the similarity distribution
   are extracted: basic statistics, distribution shape, percentiles, tail
   measures, information-theoretic quantities, derived ratios, and normality
   tests. The full list is in `core/feature_extractor.py` and
   `training/data_loader.py`.
5. **Classification.** XGBoost with 5-fold cross-validation, early stopping,
   and recursive feature elimination to a per-expert subset of 8 features.
6. **Speech ensemble.** Weighted average of five experts trained on different
   speech corpora:

   | Expert | Training corpus | Weight |
   |---|---|---|
   | `ds_msclap` | DeepSpeak v2 | 0.30 |
   | `audeter_msclap` | AUDETER (modern TTS) | 0.30 |
   | `sv_ds_msclap` | ASVspoof + DeepSpeak | 0.20 |
   | `sv_msclap` | ASVspoof 2019 LA | 0.10 |
   | `mlaad_msclap` | MLAAD (84 TTS systems) | 0.10 |

   Decision threshold: τ = 0.30 (optimised on a held-out portion of training
   data; see `training/trainer.py`).
7. **Music classifier.** A domain-adaptive percentile rule that compares each
   batch's mean similarity against the 50th percentile within the batch,
   without requiring target-domain labels.

## Pretrained classifiers

`models/trained/` contains the checkpoints used to produce the paper's
numbers:

| File | Domain | Training data |
|---|---|---|
| `ds_msclap_model.pkl` | Speech | DeepSpeak v2 |
| `sv_msclap_model.pkl` | Speech | ASVspoof 2019 LA |
| `sv_ds_msclap_model.pkl` | Speech | ASVspoof + DeepSpeak |
| `mlaad_msclap_model.pkl` | Speech | MLAAD |
| `audeter_msclap_model.pkl` | Speech | AUDETER |
| `mi_adaptive_model.pkl` | Music | Adaptive percentile (MUSDB18 + FakeMusicCaps) |
| `mi_msclap_model.pkl` | Music (instrumental) | MUSDB18 + FakeMusicCaps |
| `mv_msclap_model.pkl` | Music (with vocals) | MUSDB18 + FakeMusicCaps |
| `singlevoice_msclap_consistent_model.pkl` | Speech | LibriSpeech + ASVspoof |

Each `.pkl` contains a fitted XGBoost classifier together with its scaler and
selected-feature list.

## Citation

```bibtex
@misc{norman2026audiodeepfake,
  title  = {Audio Deepfake Detection Using Temporal Coherence Analysis},
  author = {Norman, Justin and Barrington, Sarah},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/stbiadmin/audiodeepfake_public}
}
```

## License

Released under the MIT License (see `LICENSE`).

This repository depends on third-party models and datasets with their own
licenses. See `THIRD_PARTY_NOTICES.md` for the full list.

## Contact

Questions, bug reports, and reproduction issues are welcome via the GitHub
Issues tracker. For correspondence, contact Justin Norman
(`justin.norman@berkeley.edu`) or Sarah Barrington
(`sbarrington@berkeley.edu`).
