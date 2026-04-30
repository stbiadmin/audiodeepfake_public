#!/bin/bash
# MLAAD English feature extraction script

PYTHON="python"
SCRIPT="scripts/extract_features.py"
LOG_DIR="logs/mlaad_extraction"
OUT_DIR="data/features/wavlm/raw"

mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "=== Starting MLAAD WavLM Extractions ==="
echo "Started at: $(date)"

# 1. MLAAD real (M-AILABS en_US) - 46,294 files
echo ""
echo "[1/2] mlaad_english/real (46,294 files)..."
$PYTHON $SCRIPT data/experiments/mlaad_english/real \
    -t mlaad_english -l 0 --embedding-model wavlm \
    -o "$OUT_DIR/mlaad_english_real.json" -p "**/*.wav" \
    2>&1 | tee "$LOG_DIR/mlaad_real.log"

# 2. MLAAD fake - 12,689 files
echo ""
echo "[2/2] mlaad_english/fake (12,689 files)..."
$PYTHON $SCRIPT data/experiments/mlaad_english/fake \
    -t mlaad_english -l 1 --embedding-model wavlm \
    -o "$OUT_DIR/mlaad_english_fake.json" -p "**/*.wav" \
    2>&1 | tee "$LOG_DIR/mlaad_fake.log"

echo ""
echo "=== MLAAD extractions complete ==="
echo "Finished at: $(date)"
