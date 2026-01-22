#!/bin/bash
# ============================================================
# Train MDFAN model (Multi-source Domain Adaptation)
# ============================================================

set -e

# Default values
BACKBONE="${BACKBONE:-resnet50}"
SOURCE_DIRS="${SOURCE_DIRS:-./data/raw/plantvillage ./data/raw/local_commercial}"
TARGET_DIR="${TARGET_DIR:-./data/raw/andean_field}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
LAMBDA_MMD="${LAMBDA_MMD:-1.0}"
LAMBDA_ADV="${LAMBDA_ADV:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

echo "🥔 Training MDFAN Model"
echo "========================"
echo "Backbone: $BACKBONE"
echo "Sources: $SOURCE_DIRS"
echo "Target: $TARGET_DIR"
echo "λ_mmd: $LAMBDA_MMD | λ_adv: $LAMBDA_ADV"
echo ""

uv run python src/train.py \
    --model mdfan \
    --backbone "$BACKBONE" \
    --source_dirs $SOURCE_DIRS \
    --target_dir "$TARGET_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --lambda_mmd "$LAMBDA_MMD" \
    --lambda_adv "$LAMBDA_ADV" \
    --output_dir "$OUTPUT_DIR" \
    --use_andean_aug \
    "$@"
