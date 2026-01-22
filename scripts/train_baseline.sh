#!/bin/bash
# ============================================================
# Train baseline model
# ============================================================

set -e

# Default values
BACKBONE="${BACKBONE:-mobilenet_v3_small}"
DATA_DIR="${DATA_DIR:-./data/raw/plantvillage}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

echo "🥔 Training Baseline Model"
echo "=========================="
echo "Backbone: $BACKBONE"
echo "Data: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo ""

uv run python src/train.py \
    --model baseline \
    --backbone "$BACKBONE" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --output_dir "$OUTPUT_DIR" \
    --use_andean_aug \
    "$@"
