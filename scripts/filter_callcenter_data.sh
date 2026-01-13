#!/bin/bash
# Script: filter_callcenter_data.sh
# Pipeline lọc dữ liệu cho Call Center ASR
# 
# Based on:
# - "Efficient Data Selection for Domain Adaptation of ASR" (IDIAP, INTERSPEECH 2025)
# - "Improved Noisy Student Training for ASR" (Google, INTERSPEECH 2020)
#
# Usage: bash scripts/filter_callcenter_data.sh

set -e

# ============================
# Configuration
# ============================
SCRIPTS_DIR="/vietasr/scripts"
FBANK_DIR="/vietasr/fbank"
DATA_RAW_DIR="/data/raw"
DATA_FILTERED_DIR="/data/filtered"

# Human-labeled datasets
HUMAN_DATASETS=("tongdai" "tongdai_22112024" "tongdai_092022")

# Pseudo-labeled dataset
PSEUDO_DATASET="tongdai_112025"
PSEUDO_DIR="${DATA_RAW_DIR}/pesudo/${PSEUDO_DATASET}"

# Thresholds
WER_THRESHOLD_HUMAN=0.5      # 50% WER cho human-labeled
WER_THRESHOLD_PSEUDO=0.3     # 30% WER cho pseudo-labeled
TARGET_RETENTION_PSEUDO=0.15 # Giữ 15% pseudo-labeled tốt nhất

# ============================
# Phase 1: Human-labeled Data
# ============================
echo "=============================================="
echo "PHASE 1: Filtering Human-labeled Data"
echo "=============================================="
echo "Strategy: Remove samples with WER > ${WER_THRESHOLD_HUMAN} (50%)"
echo ""

for dataset in "${HUMAN_DATASETS[@]}"; do
    echo "----------------------------------------"
    echo "Processing: ${dataset}"
    
    manifest_in="${FBANK_DIR}/${dataset}/vietASR_cuts_${dataset}.jsonl.gz"
    manifest_out="${FBANK_DIR}/${dataset}/vietASR_cuts_${dataset}_filtered.jsonl.gz"
    
    # Tìm file decode results
    recogs_pattern="${FBANK_DIR}/../data4/exp/*/recogs-${dataset}*.txt"
    recogs_file=$(find ${FBANK_DIR}/../data4/exp -name "recogs-${dataset}*.txt" 2>/dev/null | head -1)
    
    if [ ! -f "${manifest_in}" ]; then
        echo "  [SKIP] Manifest not found: ${manifest_in}"
        continue
    fi
    
    if [ -z "${recogs_file}" ]; then
        echo "  [WARNING] Decode results not found for ${dataset}"
        echo "  [ACTION] Run: python ASR/zipformer/decode.py --test-set ${dataset}"
        continue
    fi
    
    echo "  Input:  ${manifest_in}"
    echo "  Decode: ${recogs_file}"
    
    # Count before
    count_before=$(zcat "${manifest_in}" | wc -l)
    echo "  Samples before: ${count_before}"
    
    # Run filter
    python3 "${SCRIPTS_DIR}/filter_bad_labels.py" \
        --manifest-in "${manifest_in}" \
        --decode-results "${recogs_file}" \
        --manifest-out "${manifest_out}" \
        --wer-threshold ${WER_THRESHOLD_HUMAN} \
        --inspect-num 10
    
    # Count after
    if [ -f "${manifest_out}" ]; then
        count_after=$(zcat "${manifest_out}" | wc -l)
        removed=$((count_before - count_after))
        percent=$(echo "scale=1; ${removed} * 100 / ${count_before}" | bc)
        echo "  Samples after: ${count_after} (removed: ${removed} = ${percent}%)"
    fi
    echo ""
done

# ============================
# Phase 2: Pseudo-labeled Data
# ============================
echo ""
echo "=============================================="
echo "PHASE 2: Filtering Pseudo-labeled Data"
echo "=============================================="
echo "Strategy: Multi-criteria filtering, keep top ${TARGET_RETENTION_PSEUDO}%"
echo "Dataset: ${PSEUDO_DATASET}"
echo ""

pseudo_transcripts="${PSEUDO_DIR}/transcripts.txt"
pseudo_audio_dir="${PSEUDO_DIR}/audio"

if [ ! -f "${pseudo_transcripts}" ]; then
    echo "[ERROR] Transcripts not found: ${pseudo_transcripts}"
    exit 1
fi

mkdir -p "${DATA_FILTERED_DIR}/pesudo/${PSEUDO_DATASET}"
filtered_output="${DATA_FILTERED_DIR}/pesudo/${PSEUDO_DATASET}/transcripts_filtered.txt"

# Option 1: Basic filtering (without WER comparison)
echo "Running multi-criteria filtering..."
python3 "${SCRIPTS_DIR}/filter_pseudo_strict.py" \
    --transcripts-in "${pseudo_transcripts}" \
    --transcripts-out "${filtered_output}" \
    --audio-dir "${pseudo_audio_dir}" \
    --min-duration 2.0 \
    --max-duration 25.0 \
    --min-words 3 \
    --max-words 50 \
    --min-viet-ratio 0.9 \
    --target-retention ${TARGET_RETENTION_PSEUDO} \
    --deduplicate \
    --inspect-num 10

# Count result
if [ -f "${filtered_output}" ]; then
    count_before=$(wc -l < "${pseudo_transcripts}")
    count_after=$(wc -l < "${filtered_output}")
    echo ""
    echo "Pseudo-label filtering result:"
    echo "  Before: ${count_before}"
    echo "  After:  ${count_after} (${TARGET_RETENTION_PSEUDO}%)"
fi

# ============================
# Summary
# ============================
echo ""
echo "=============================================="
echo "FILTERING COMPLETE"
echo "=============================================="
echo ""
echo "Human-labeled filtered outputs:"
for dataset in "${HUMAN_DATASETS[@]}"; do
    manifest_out="${FBANK_DIR}/${dataset}/vietASR_cuts_${dataset}_filtered.jsonl.gz"
    if [ -f "${manifest_out}" ]; then
        count=$(zcat "${manifest_out}" | wc -l)
        echo "  ${dataset}: ${count} samples"
    fi
done
echo ""
echo "Pseudo-labeled filtered output:"
echo "  ${filtered_output}"
echo ""
echo "Next steps:"
echo "1. Review filtered outputs"
echo "2. Combine filtered datasets for training"
echo "3. Train model with clean data"
