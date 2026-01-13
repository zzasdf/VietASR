#!/bin/bash
# ============================================================================
# Dual-Model Pseudo-label Filtering Pipeline
# ============================================================================
# 
# Based on Inter-ASR Agreement (IDIAP INTERSPEECH 2025)
# Filters pseudo-labels using BOTH Whisper and Zipformer
#
# Usage:
#   bash scripts/filter_pseudo_dual_model.sh <GPU_ID> [OPTIONS]
#
# Example:
#   bash scripts/filter_pseudo_dual_model.sh 0
#   bash scripts/filter_pseudo_dual_model.sh 2 --skip-whisper
#
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================
GPU_ID="${1:-0}"
SKIP_WHISPER="${2:-}"

# Paths (adjust as needed)
PSEUDO_DIR="/data/raw/pseudo/call_center"
WER_FILE="${PSEUDO_DIR}/wer.txt"
TRANSCRIPTS_FILE="${PSEUDO_DIR}/transcripts.txt"
WAVS_DIR="${PSEUDO_DIR}/wavs"
OUTPUT_DIR="${PSEUDO_DIR}/filtered"
MANIFEST_DIR="${PSEUDO_DIR}/manifests"

# Model configuration
EXP_DIR="/vietasr/data4/exp"
EPOCH=30
AVG=15
BPE_MODEL="/vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"

# Thresholds
WHISPER_WER_THRESHOLD=0.3
ZIPFORMER_WER_THRESHOLD=0.3
MIN_DURATION=1.5
MAX_DURATION=25.0

# ============================================================================
# Setup
# ============================================================================
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "=============================================="
echo "Dual-Model Pseudo-label Filtering Pipeline"
echo "=============================================="
echo "GPU: ${GPU_ID}"
echo "Pseudo dir: ${PSEUDO_DIR}"
echo "Whisper WER threshold: ${WHISPER_WER_THRESHOLD}"
echo "Zipformer WER threshold: ${ZIPFORMER_WER_THRESHOLD}"
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MANIFEST_DIR}"

# ============================================================================
# Step 1: Pre-filter by Whisper WER
# ============================================================================
CANDIDATES_FILE="${OUTPUT_DIR}/candidates_whisper.txt"

if [ "${SKIP_WHISPER}" != "--skip-whisper" ]; then
    echo "=============================================="
    echo "Step 1: Pre-filtering by Whisper WER"
    echo "=============================================="
    
    if [ ! -f "${WER_FILE}" ]; then
        echo "[ERROR] WER file not found: ${WER_FILE}"
        echo "Falling back to transcripts.txt without Whisper filtering"
        cp "${TRANSCRIPTS_FILE}" "${CANDIDATES_FILE}"
    else
        python /vietasr/scripts/filter_by_whisper_wer.py \
            --wer-file "${WER_FILE}" \
            --output "${CANDIDATES_FILE}" \
            --max-wer ${WHISPER_WER_THRESHOLD} \
            --min-duration ${MIN_DURATION} \
            --max-duration ${MAX_DURATION} \
            --inspect-num 10
    fi
else
    echo "[SKIP] Using transcripts.txt directly"
    cp "${TRANSCRIPTS_FILE}" "${CANDIDATES_FILE}"
fi

CANDIDATES_COUNT=$(wc -l < "${CANDIDATES_FILE}")
echo ""
echo "Candidates after Whisper filter: ${CANDIDATES_COUNT}"
echo ""

# ============================================================================
# Step 2: Create Lhotse manifests for candidates
# ============================================================================
echo "=============================================="
echo "Step 2: Creating Lhotse manifests"
echo "=============================================="

python /vietasr/scripts/prepare_custom_data.py \
    --transcript "${CANDIDATES_FILE}" \
    --wav-dir "${WAVS_DIR}" \
    --output-dir "${MANIFEST_DIR}" \
    --dataset-name candidates

CUTS_FILE="${MANIFEST_DIR}/candidates_cuts.jsonl.gz"
if [ ! -f "${CUTS_FILE}" ]; then
    echo "[ERROR] Failed to create cuts file"
    exit 1
fi

echo "Created: ${CUTS_FILE}"
echo ""

# ============================================================================
# Step 3: Decode with Zipformer
# ============================================================================
echo "=============================================="
echo "Step 3: Decoding with Zipformer (GPU: ${GPU_ID})"
echo "=============================================="

cd /vietasr/ASR/zipformer

python decode.py \
    --epoch ${EPOCH} \
    --avg ${AVG} \
    --exp-dir ${EXP_DIR} \
    --manifest-dir ${MANIFEST_DIR} \
    --cuts-name candidates \
    --bpe-model ${BPE_MODEL} \
    --on-the-fly-feats True \
    --decoding-method greedy_search \
    --max-duration 300 \
    --use-averaged-model True

cd /vietasr

# Find the recogs file
RECOGS_FILE=$(find ${EXP_DIR}/greedy_search -name "recogs-candidates-*.txt" -type f | sort -r | head -1)

if [ -z "${RECOGS_FILE}" ]; then
    echo "[ERROR] Recogs file not found"
    exit 1
fi

echo "Found recogs: ${RECOGS_FILE}"
echo ""

# ============================================================================
# Step 4: Filter by Zipformer WER
# ============================================================================
echo "=============================================="
echo "Step 4: Filtering by Zipformer WER"
echo "=============================================="

FINAL_MANIFEST="${OUTPUT_DIR}/final_filtered_cuts.jsonl.gz"

python /vietasr/scripts/filter_bad_labels.py \
    --manifest-in "${CUTS_FILE}" \
    --decode-results "${RECOGS_FILE}" \
    --manifest-out "${FINAL_MANIFEST}" \
    --wer-threshold ${ZIPFORMER_WER_THRESHOLD} \
    --inspect-num 20

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="

if [ -f "${FINAL_MANIFEST}" ]; then
    FINAL_COUNT=$(zcat "${FINAL_MANIFEST}" | wc -l)
    ORIGINAL_COUNT=$(wc -l < "${TRANSCRIPTS_FILE}")
    RETENTION=$(echo "scale=1; ${FINAL_COUNT} * 100 / ${ORIGINAL_COUNT}" | bc)
    
    echo "Original samples: ${ORIGINAL_COUNT}"
    echo "After Whisper:    ${CANDIDATES_COUNT}"
    echo "Final output:     ${FINAL_COUNT} (${RETENTION}%)"
    echo ""
    echo "Output file: ${FINAL_MANIFEST}"
else
    echo "[ERROR] Final manifest not created"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Review filtered samples: zcat ${FINAL_MANIFEST} | head -10"
echo "  2. Use for training: --manifest-dir ${OUTPUT_DIR}"
