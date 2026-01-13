#!/bin/bash
# Decode with Word-level LM Rescoring
# Properly uses 17GB KenLM for fair comparison with API
#
# Usage: bash scripts/decode/decode_lm_rescore.sh [epoch] [avg] [gpu]
# Example: bash scripts/decode/decode_lm_rescore.sh 19 17 0

set -e  # Exit on error

#==============================================================================
# CONFIGURABLE PARAMETERS
#==============================================================================

EPOCH=${1:-19}
AVG=${2:-17}
GPU=${3:-0}

# Test set - CHANGE THIS
TEST_SETS="tongdai_clean"
MANIFEST_DIR="fbank/tongdai_clean"

# Model paths
EXP_DIR="data4/exp_finetune_v3"
BPE_MODEL="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"

# 17GB Word-level KenLM
WORD_LM="/data/raw/dolphin/lm_kenlm/4_gram_word_all.binary"
WORD_LM_SCALE=0.3

# Decoding params
BEAM_SIZE=10
MAX_DURATION=1000

# Model params
USE_AVERAGED_MODEL=1
FINAL_DOWNSAMPLE=1
USE_LAYER_NORM=0
NORM=1

#==============================================================================

echo "=============================================="
echo "Decode with Word-level LM Rescoring"
echo "=============================================="
echo "Epoch: $EPOCH, Avg: $AVG, GPU: $GPU"
echo "Test sets: $TEST_SETS"
echo "Manifest: $MANIFEST_DIR"
echo "Word LM: $WORD_LM"
echo "LM Scale: $WORD_LM_SCALE"
echo ""
echo "NOTE: LM will be loaded first (~22 min), then decode"
echo ""

python scripts/decode/decode_with_lm_rescore.py \
    --epoch $EPOCH \
    --avg $AVG \
    --gpu $GPU \
    --test-sets "$TEST_SETS" \
    --manifest-dir "$MANIFEST_DIR" \
    --exp-dir "$EXP_DIR" \
    --bpe-model "$BPE_MODEL" \
    --word-lm-path "$WORD_LM" \
    --word-lm-scale $WORD_LM_SCALE \
    --beam-size $BEAM_SIZE \
    --max-duration $MAX_DURATION \
    --use-averaged-model $USE_AVERAGED_MODEL \
    --final-downsample $FINAL_DOWNSAMPLE \
    --use-layer-norm $USE_LAYER_NORM \
    --norm $NORM

echo ""
echo "✅ Done!"
