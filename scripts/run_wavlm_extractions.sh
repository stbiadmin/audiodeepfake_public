#!/bin/bash
# Sequential WavLM feature extraction script

PYTHON="python"
SCRIPT="scripts/extract_features.py"
LOG_DIR="logs/wavlm_extraction"
OUT_DIR="data/features/wavlm/raw"

mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "=== Starting Sequential WavLM Extractions ==="
echo "Started at: $(date)"

# 1. single_voice real
echo ""
echo "[1/4] single_voice/real (2,703 files)..."
$PYTHON $SCRIPT data/experiments/single_voice/real \
    -t single_voice -l 0 --embedding-model wavlm \
    -o "$OUT_DIR/single_voice_real.json" -p "*.flac" \
    2>&1 | tee "$LOG_DIR/single_voice_real.log"

# 2. single_voice fake  
echo ""
echo "[2/4] single_voice/fake (22,800 files)..."
$PYTHON $SCRIPT data/experiments/single_voice/fake \
    -t single_voice -l 1 --embedding-model wavlm \
    -o "$OUT_DIR/single_voice_fake.json" -p "**/*.flac" \
    2>&1 | tee "$LOG_DIR/single_voice_fake.log"

# 3. deepspeak real
echo ""
echo "[3/4] deepspeak/real (7,515 files)..."
$PYTHON $SCRIPT data/experiments/deepspeak_v2_train/real \
    -t deepspeak_v2_train -l 0 --embedding-model wavlm \
    -o "$OUT_DIR/deepspeak_v2_train_real.json" -p "**/*.wav" \
    2>&1 | tee "$LOG_DIR/deepspeak_real.log"

# 4. deepspeak fake
echo ""
echo "[4/4] deepspeak/fake (1,984 files)..."
$PYTHON $SCRIPT data/experiments/deepspeak_v2_train/fake \
    -t deepspeak_v2_train -l 1 --embedding-model wavlm \
    -o "$OUT_DIR/deepspeak_v2_train_fake.json" -p "**/*.wav" \
    2>&1 | tee "$LOG_DIR/deepspeak_fake.log"

echo ""
echo "=== All extractions complete ==="
echo "Finished at: $(date)"
