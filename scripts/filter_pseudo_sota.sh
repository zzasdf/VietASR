#!/bin/bash
# ============================================================================
# Multi-Stage Pseudo-label Filtering Pipeline (SOTA 2024)
# ============================================================================
# 
# This pipeline filters pseudo-labels through 5 stages:
#   Stage 0: Text Normalization (Numbers -> Text, Tone, Punctuation)
#   Stage 1-2: Heuristics & Repetition (CPU only)
#   Stage 3: Language Model PPL Scoring (CPU only, uses kenlm)
#   Stage 4: Cross-Model CER Check (GPU required)
#   Stage 5: Ranking & Top K% Selection (CPU only)
#
# IMPORTANT: This script only ADDS new files, never deletes original data.
#
# Usage:
#   bash scripts/filter_pseudo_sota.sh [DATASET_NAME] [INPUT_FILE] [OUTPUT_DIR] [GPU_ID]
#
# Example:
#   bash scripts/filter_pseudo_sota.sh call_center /data/transcripts.txt /data/filtered 0
#
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================
# Arguments
DATASET_NAME="${1:-call_center}"
INPUT_FILE="${2:-/data/raw/pseudo/call_center/transcripts.txt}"
OUTPUT_DIR="${3:-/data/raw/pseudo/call_center/filtered}"
GPU_ID="${4:-0}"

# Paths (derived)
LM_PATH="/vietasr/data4/lm/lm_4gram.arpa"

echo "=============================================="
echo "Multi-Stage Pipeline: ${DATASET_NAME}"
echo "=============================================="
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_DIR}"

# Thresholds
MIN_DURATION=1.0  # OPTIMIZED: Was 2.0
MAX_DURATION=25.0
MIN_WORDS=3
MAX_WORDS=50
MIN_VIET_RATIO=0.7  # OPTIMIZED: Was 0.9 (Too strict for normalized text)
PPL_HIGH_PERCENTILE=90
PPL_LOW_PERCENTILE=5
TARGET_RETENTION=0.2  # Keep top 20%

# ============================================================================
# Setup
# ============================================================================
echo ""
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Stage 0: Text Normalization (Numbers -> Text, Tone, Punctuation)
# ============================================================================
echo "=============================================="
echo "Stage 0: Text Normalization"
echo "=============================================="

STAGE0_OUTPUT="${OUTPUT_DIR}/transcripts_norm.txt"

if [ -f "${STAGE0_OUTPUT}" ]; then
    echo "[SKIP] Stage 0 output already exists: ${STAGE0_OUTPUT}"
    echo "       Delete it to re-run this stage."
else
    # Sử dụng script chuẩn hóa có sẵn
    python /vietasr/scripts/data/normalize_transcripts.py \
        "${INPUT_FILE}" \
        "${STAGE0_OUTPUT}"
fi

STAGE0_COUNT=$(wc -l < "${STAGE0_OUTPUT}")
echo "Stage 0 output: ${STAGE0_COUNT} samples"

# Update input for next stages
INPUT_FOR_NEXT_STAGE="${STAGE0_OUTPUT}"

# ============================================================================
# Stage 1-2: Heuristics & Repetition Filter
# ============================================================================
echo "=============================================="
echo "Stage 1-2: Heuristics & Repetition Filter"
echo "=============================================="

STAGE12_OUTPUT="${OUTPUT_DIR}/stage1_2.txt"
STAGE12_STATS="${OUTPUT_DIR}/stage1_2_stats.json"

if [ -f "${STAGE12_OUTPUT}" ]; then
    echo "[SKIP] Stage 1-2 output already exists: ${STAGE12_OUTPUT}"
    echo "       Delete it to re-run this stage."
else
    python /vietasr/scripts/filter_stage1_2_heuristics.py \
        --input "${INPUT_FOR_NEXT_STAGE}" \
        --output "${STAGE12_OUTPUT}" \
        --stats-output "${STAGE12_STATS}" \
        --min-duration ${MIN_DURATION} \
        --max-duration ${MAX_DURATION} \
        --min-words ${MIN_WORDS} \
        --max-words ${MAX_WORDS} \
        --min-viet-ratio ${MIN_VIET_RATIO}
fi

STAGE12_COUNT=$(wc -l < "${STAGE12_OUTPUT}")
echo ""
echo "Stage 1-2 output: ${STAGE12_COUNT} samples"
echo ""

# ============================================================================
# Stage 3: Language Model PPL Scoring
# ============================================================================
echo "=============================================="
echo "Stage 3: Language Model PPL Scoring"
echo "=============================================="

STAGE3_OUTPUT="${OUTPUT_DIR}/stage3.txt"

if [ ! -f "${LM_PATH}" ]; then
    echo "[WARNING] LM file not found: ${LM_PATH}"
    echo "          Skipping Stage 3, copying Stage 1-2 output directly."
    cp "${STAGE12_OUTPUT}" "${STAGE3_OUTPUT}"
else
    if [ -f "${STAGE3_OUTPUT}" ]; then
        echo "[SKIP] Stage 3 output already exists: ${STAGE3_OUTPUT}"
    else
        python /vietasr/scripts/filter_stage3_lm_score.py \
            --input "${STAGE12_OUTPUT}" \
            --lm-path "${LM_PATH}" \
            --output "${STAGE3_OUTPUT}" \
            --ppl-high-percentile ${PPL_HIGH_PERCENTILE} \
            --ppl-low-percentile ${PPL_LOW_PERCENTILE}
    fi
fi

STAGE3_COUNT=$(wc -l < "${STAGE3_OUTPUT}")
echo ""
echo "Stage 3 output: ${STAGE3_COUNT} samples"
echo ""

# ============================================================================
# Stage 4: Cross-Model CER Check (Optional/Manual)
# ============================================================================
# Note: Stage 4 requires GPU decoding which is best run separately or uncommented if GPU available
# See README_filter_pseudo.md for instructions

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "PIPELINE SUMMARY (${DATASET_NAME})"
echo "=============================================="

ORIGINAL_COUNT=$(wc -l < "${INPUT_FILE}")
if [ "${ORIGINAL_COUNT}" -gt 0 ]; then
    RETENTION=$(awk "BEGIN {printf \"%.1f\", ${STAGE3_COUNT} * 100 / ${ORIGINAL_COUNT}}")
else
    RETENTION=0
fi

echo "Original samples:     ${ORIGINAL_COUNT}"
echo "After Stage 0 (Norm): ${STAGE0_COUNT}"
echo "After Stage 1-2:      ${STAGE12_COUNT}"
echo "After Stage 3:        ${STAGE3_COUNT} (${RETENTION}%)"
echo ""
echo "Output files:"
echo "  - ${STAGE0_OUTPUT}"
echo "  - ${STAGE12_OUTPUT}"
echo "  - ${STAGE12_STATS}"
echo "  - ${STAGE3_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. Review stats: cat ${STAGE12_STATS}"
echo "  2. Run Stage 4 (Cross-Model CER) for deeper filtering (requires GPU decode)"
echo "  3. Use ${STAGE3_OUTPUT} for training if Stage 4 is not needed"
