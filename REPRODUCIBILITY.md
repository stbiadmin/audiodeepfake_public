# Reproducibility Guide

Step-by-step instructions for reproducing each numerical claim in the paper.
All commands assume the working directory is the repository root and that
the virtual environment described in `README.md` is active.

## Prerequisites

1. Install runtime dependencies (`pip install -r requirements.txt`).
2. Acquire the relevant datasets following `data/README.md` and place them
   under `data/`.
3. Pre-extract feature distributions for the datasets you intend to evaluate
   (this takes most of the wall-clock time; everything downstream is fast).

```bash
# Example: extract features for the In-the-Wild benchmark
python scripts/extract_features.py \
    data/in_the_wild \
    --type single_voice \
    --embedding msclap \
    --output-dir data/eval_features/msclap/in_the_wild
```

Per-dataset extraction commands are listed below.

## Hardware and runtime

The experiments in the paper were run on an Apple M1 Pro (CPU and MPS) and an
x86_64 Linux machine with one NVIDIA GPU. None of the published numbers
require a GPU; the dominant cost is the MS-CLAP embedding step
(approximately one file per second on CPU). Recipes here assume that
features have already been extracted and cached.

## Per-table reproduction

### Table 3 - Embedding comparison (MS-CLAP vs. WavLM)

```bash
python sota_benchmarking/run_ssl_temporal_eval.py
```

Expected: best-feature AUC of 0.949 for MS-CLAP and 0.633 for WavLM on the
ASVspoof5 modern-TTS subset (q90 percentile feature).

### Table 4 - Top features per audio type

```bash
python evaluation/feature_importance.py \
    --features-glob 'data/features/msclap/*' \
    --output results/table4_top_features.csv
```

### Table 5 - Feature-label inversion

```bash
python scripts/scientific_analysis.py --analysis inversion \
    --train data/features/msclap/single_voice_train.json \
    --eval data/eval_features/msclap/in_the_wild/features_combined.json \
    --output results/table5_inversion.csv
```

Expected: 21 of 29 features reverse their discriminative direction between
training and the in-the-wild evaluation set.

### Table 6 - 5-expert speech ensemble on In-the-Wild

```bash
python scripts/evaluate_ensembles.py \
    --eval-dataset in_the_wild \
    --models ds_msclap sv_msclap sv_ds_msclap mlaad_msclap audeter_msclap \
    --weights 0.30 0.10 0.20 0.10 0.30 \
    --threshold 0.30
```

Expected: ensemble F1 = 0.675, AUC = 0.708.

### Table 7 - Music detection (in-distribution and cross-domain)

```bash
# In-distribution (FakeMusicCaps held-out splits)
python scripts/evaluate_ensembles.py \
    --eval-dataset music_instrumental \
    --models mi_msclap

# Cross-domain (SONICS / FMA) using the adaptive percentile classifier
python scripts/evaluate_ensembles.py \
    --eval-dataset sonics_fma \
    --models mi_adaptive
```

Expected: in-distribution F1 = 0.995; cross-domain F1 = 0.938.

### Table 8 - State-of-the-art baselines

```bash
# AASIST and AASIST-L
./setup_baselines.sh
python sota_benchmarking/run_aasist_baseline.py \
    --eval in_the_wild asvspoof5

# wav2vec2-XLSR fine-tuned baseline
python sota_benchmarking/run_ssl_baselines.py \
    --eval in_the_wild asvspoof5
```

### Table 9 - Ensemble size ablation

```bash
python scripts/ablation_ensemble_features.py
```

### Table 10 - Threshold calibration ablation

Inside `scripts/evaluate_ensembles.py`, vary `--threshold` over
{0.20, 0.30, 0.40, 0.50}. Optimal value is 0.30 (F1 = 0.675).

### Table 11 - Domain adaptation ablation

```bash
python scripts/run_generalization_eval.py \
    --eval in_the_wild \
    --adaptations none coral target_normalization threshold_calibration
```

Expected: CORAL alignment and target normalization both degrade F1 relative
to no adaptation; threshold calibration provides a small positive gain.

### Table 12 - Feature-count ablation

```bash
python scripts/ablation_feature_count.py
```

### TreeSHAP feature attribution

```bash
python scripts/run_treeshap_analysis.py \
    --max-samples 1000 \
    --output-dir results/shap_analysis
```

Produces summary plots and per-expert feature-importance bar charts.

## Notes on stochastic variation

XGBoost training uses fixed random seeds where applicable, but exact
reproduction across hardware and library versions can vary at the third
decimal. Use the pinned versions in `requirements.txt` for the closest
match to the paper's numbers.
