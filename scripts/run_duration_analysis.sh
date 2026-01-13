#!/bin/bash
# Script chạy phân tích theo duration trên tất cả test sets
# Usage: bash scripts/run_duration_analysis.sh

set -e

# ============================================
# CẤU HÌNH - THAY ĐỔI THEO NHU CẦU
# ============================================

# GPU sử dụng (chọn 1 GPU để đo công bằng)
export CUDA_VISIBLE_DEVICES=2

# Model checkpoints
EXP_DIR="/vietasr/data4/exp_finetune_v3"
EPOCH=19
AVG=17
BPE_MODEL="/vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"

# Inference settings (giữ cố định để so sánh công bằng)
BEAM_SIZE=10
METHOD="modified_beam_search"
NUM_WARMUP=10

# Output directory
OUTPUT_DIR="/vietasr/results/duration_analysis_$(date +%Y%m%d_%H%M%S)"

# Test directories
TEST_DIRS=(
    "/data/raw/test/tongdai_clean"
    "/data/raw/test/call_center_private"
    "/data/raw/test/tongdai_other_reviewed"
    "/data/raw/test/regions.dong_nam_bo"
    "/data/raw/test/regions.bac_trung_bo"
)

# ============================================
# MAIN
# ============================================

echo "=================================================="
echo "DURATION ANALYSIS BENCHMARK"
echo "=================================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Model: ${EXP_DIR}"
echo "Epoch: ${EPOCH}, Avg: ${AVG}"
echo "Method: ${METHOD}, Beam: ${BEAM_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "=================================================="

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run analysis on each test set
for TEST_DIR in "${TEST_DIRS[@]}"; do
    if [ -d "${TEST_DIR}" ]; then
        TEST_NAME=$(basename ${TEST_DIR})
        echo ""
        echo "=================================================="
        echo "Processing: ${TEST_NAME}"
        echo "=================================================="
        
        python /vietasr/scripts/analyze_by_duration.py \
            --test-dir ${TEST_DIR} \
            --exp-dir ${EXP_DIR} \
            --bpe-model ${BPE_MODEL} \
            --epoch ${EPOCH} \
            --avg ${AVG} \
            --device cuda \
            --beam-size ${BEAM_SIZE} \
            --method ${METHOD} \
            --output-dir ${OUTPUT_DIR} \
            --num-warmup ${NUM_WARMUP}
        
        echo "Done: ${TEST_NAME}"
    else
        echo "WARNING: Directory not found: ${TEST_DIR}"
    fi
done

# Generate combined summary
echo ""
echo "=================================================="
echo "GENERATING COMBINED SUMMARY"
echo "=================================================="

python3 << 'EOF'
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/vietasr/results/duration_analysis")

# Find all summary files
summaries = list(output_dir.glob("*_summary.json"))

if not summaries:
    print("No summary files found")
    sys.exit(0)

# Print combined table
print("\n" + "=" * 120)
print("COMBINED DURATION ANALYSIS RESULTS")
print("=" * 120)

# Header
bins = ["0-4s", "4-8s", "8-12s", "12-16s", "16-20s", "20s+"]
header = f"{'Test Set':<25}"
for b in bins:
    header += f" | {b:>12}"
header += " | {'Overall':>12}"
print(header)
print("-" * 120)

for summary_file in sorted(summaries):
    with open(summary_file) as f:
        data = json.load(f)
    
    test_name = data["test_name"]
    row_wer = f"{test_name:<25}"
    row_rtf = f"{'  RTF':<25}"
    row_cnt = f"{'  Count':<25}"
    
    total_words = 0
    total_errors = 0
    total_duration = 0
    total_latency = 0
    
    for b in bins:
        if b in data["bins"]:
            stats = data["bins"][b]
            row_wer += f" | {stats['wer']:>10.2f}%"
            row_rtf += f" | {stats['rtf']:>12.4f}"
            row_cnt += f" | {stats['count']:>12}"
            # For overall calculation - need to estimate words from WER and errors
            total_duration += stats["total_duration"]
            total_latency += stats["total_latency"]
        else:
            row_wer += f" | {'-':>12}"
            row_rtf += f" | {'-':>12}"
            row_cnt += f" | {0:>12}"
    
    # Overall
    overall_rtf = total_latency / total_duration if total_duration > 0 else 0
    row_wer += f" | {'see json':>12}"
    row_rtf += f" | {overall_rtf:>12.4f}"
    
    print(row_wer)
    print(row_rtf)
    print(row_cnt)
    print()

print("=" * 120)
print(f"Results saved to: {output_dir}")
EOF

echo ""
echo "=================================================="
echo "ANALYSIS COMPLETE!"
echo "Results: ${OUTPUT_DIR}"
echo "=================================================="
