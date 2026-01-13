#!/bin/bash
# Script so sánh 2 models với cùng điều kiện
#
# QUAN TRỌNG: Để so sánh công bằng, cần:
# 1. Cùng GPU
# 2. Cùng beam-size  
# 3. Cùng method (modified_beam_search)
# 4. Cùng preprocessing (kaldifeat Fbank)
# 5. Warmup trước khi đo
#
# Usage: bash scripts/compare_two_models.sh

set -e

# ============================================
# CẤU HÌNH 2 MODELS
# ============================================

export CUDA_VISIBLE_DEVICES=2

# Model 1: Your current model
MODEL1_NAME="zipformer_finetune_v3"
MODEL1_EXP_DIR="/vietasr/data4/exp_finetune_v3"
MODEL1_EPOCH=19
MODEL1_AVG=17

# Model 2: Team's API model (chạy từ checkpoint)
# TODO: Điền thông tin model của team
MODEL2_NAME="team_api_model"
MODEL2_EXP_DIR="/path/to/team/exp"  # <-- THAY ĐỔI
MODEL2_EPOCH=30                      # <-- THAY ĐỔI
MODEL2_AVG=5                         # <-- THAY ĐỔI

# Common settings (PHẢI GIỐNG NHAU)
BPE_MODEL="/vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"
BEAM_SIZE=10
METHOD="modified_beam_search"
NUM_WARMUP=10

# Test set to compare
TEST_DIR="/data/raw/test/tongdai_clean"  # Bắt đầu với 1 test set

# Output
OUTPUT_DIR="/vietasr/results/model_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

# ============================================
# RUN COMPARISON
# ============================================

echo "=================================================="
echo "MODEL COMPARISON - SAME CONDITIONS"
echo "=================================================="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Beam Size: ${BEAM_SIZE}"
echo "Method: ${METHOD}"
echo "Test: ${TEST_DIR}"
echo "=================================================="

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run Model 1
echo ""
echo ">>> Running Model 1: ${MODEL1_NAME}"
python /vietasr/scripts/analyze_by_duration.py \
    --test-dir ${TEST_DIR} \
    --exp-dir ${MODEL1_EXP_DIR} \
    --bpe-model ${BPE_MODEL} \
    --epoch ${MODEL1_EPOCH} \
    --avg ${MODEL1_AVG} \
    --device cuda \
    --beam-size ${BEAM_SIZE} \
    --method ${METHOD} \
    --output-dir ${OUTPUT_DIR}/${MODEL1_NAME} \
    --num-warmup ${NUM_WARMUP}

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Run Model 2 (uncomment when ready)
# echo ""
# echo ">>> Running Model 2: ${MODEL2_NAME}"
# python /vietasr/scripts/analyze_by_duration.py \
#     --test-dir ${TEST_DIR} \
#     --exp-dir ${MODEL2_EXP_DIR} \
#     --bpe-model ${BPE_MODEL} \
#     --epoch ${MODEL2_EPOCH} \
#     --avg ${MODEL2_AVG} \
#     --device cuda \
#     --beam-size ${BEAM_SIZE} \
#     --method ${METHOD} \
#     --output-dir ${OUTPUT_DIR}/${MODEL2_NAME} \
#     --num-warmup ${NUM_WARMUP}

# ============================================
# GENERATE COMPARISON REPORT
# ============================================

echo ""
echo "=================================================="
echo "GENERATING COMPARISON REPORT"
echo "=================================================="

python3 << EOF
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")

models = ["${MODEL1_NAME}", "${MODEL2_NAME}"]
bins = ["0-4s", "4-8s", "8-12s", "12-16s", "16-20s", "20s+"]

print("\n" + "=" * 100)
print("COMPARISON: WER by Duration Bin")
print("=" * 100)

header = f"{'Duration':<12}"
for m in models:
    header += f" | {m[:20]:>20}"
print(header)
print("-" * 100)

for b in bins:
    row = f"{b:<12}"
    for m in models:
        summary_file = output_dir / m / f"*_summary.json"
        files = list((output_dir / m).glob("*_summary.json"))
        if files:
            with open(files[0]) as f:
                data = json.load(f)
            if b in data.get("bins", {}):
                wer = data["bins"][b]["wer"]
                rtf = data["bins"][b]["rtf"]
                row += f" | WER:{wer:>5.2f}% RTF:{rtf:.4f}"
            else:
                row += f" | {'-':>20}"
        else:
            row += f" | {'N/A':>20}"
    print(row)

print("=" * 100)
EOF

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================================="
