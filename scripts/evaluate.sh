#!/bin/bash
# ============================================================
# Evaluate model with OOD detection
# ============================================================

set -e

CHECKPOINT="${1:?Usage: ./scripts/evaluate.sh <checkpoint> [test_dir] [ood_dir]}"
TEST_DIR="${2:-./data/raw/andean_field/test}"
OOD_DIR="${3:-./data/raw/andean_field/ood_classes}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval}"

echo "🥔 Evaluating Model"
echo "==================="
echo "Checkpoint: $CHECKPOINT"
echo "Test dir: $TEST_DIR"
echo "OOD dir: $OOD_DIR"
echo ""

uv run python src/eval.py \
    --checkpoint "$CHECKPOINT" \
    --test_dir "$TEST_DIR" \
    --ood_dir "$OOD_DIR" \
    --ood_method entropy \
    --output_dir "$OUTPUT_DIR" \
    --visualize \
    --save_predictions
