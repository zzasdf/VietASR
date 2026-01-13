#!/bin/bash
# Batch decode with word-level KenLM
# All parameters can be adjusted in this file
#
# Usage: bash scripts/decode/decode_lm.sh [epoch] [avg] [gpu]
# Example: bash scripts/decode/decode_lm.sh 19 17 0

#==============================================================================
# CONFIGURABLE PARAMETERS - Adjust these as needed
#==============================================================================

# Model checkpoint to use
EPOCH=${1:-19}
AVG=${2:-17}
GPU=${3:-0}

# Test set configuration
TEST_SETS="tongdai_other_reviewed"
MANIFEST_DIR="fbank/tongdai_hard"
CUTS_NAME=""

# Model paths
EXP_DIR="data4/exp_finetune_v3"
BPE_MODEL="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"

# Language model configuration (17GB word-level KenLM)
WORD_LM="/data/raw/dolphin/lm_kenlm/4_gram_word_all.binary"
WORD_LM_SCALE=0.3

# Decoding parameters
DECODING_METHOD="fast_beam_search_nbest"
BEAM_SIZE=10
MAX_DURATION=1000

# Model architecture parameters (match training)
USE_AVERAGED_MODEL=1
FINAL_DOWNSAMPLE=1
USE_LAYER_NORM=0

# Text normalization (0=basic filler removal, 1=extended tech terms)
NORM=1


echo "=============================================="
echo "Batch Decode with Word-level KenLM"
echo "=============================================="
echo "Epoch: $EPOCH, Avg: $AVG, GPU: $GPU"
echo "Test sets: $TEST_SETS"
echo "Manifest dir: $MANIFEST_DIR"
echo "Exp dir: $EXP_DIR"
echo "Word LM: $WORD_LM (scale=$WORD_LM_SCALE)"
echo "Decoding: $DECODING_METHOD, beam=$BEAM_SIZE"
echo "Norm: $NORM"
echo ""

# Check if running in Docker or Host
if command -v docker &> /dev/null; then
    echo "🐳 Running via Docker exec..."
    CONTAINER_ID="02ff3f6f9ff9"
    CMD_PREFIX="docker exec -w /vietasr $CONTAINER_ID"
else
    echo "⚠️ Docker command not found. Assuming we are already inside the container."
    CMD_PREFIX=""
fi

$CMD_PREFIX python scripts/decode/batch_decode_lm.py \
    --epoch $EPOCH \
    --avg $AVG \
    --gpu $GPU \
    --test-sets "$TEST_SETS" \
    --manifest-dir "$MANIFEST_DIR" \
    --exp-dir "$EXP_DIR" \
    --bpe-model "$BPE_MODEL" \
    --word-lm-path "$WORD_LM" \
    --word-lm-scale $WORD_LM_SCALE \
    --decoding-method "$DECODING_METHOD" \
    --beam-size $BEAM_SIZE \
    --max-duration $MAX_DURATION \
    --use-averaged-model $USE_AVERAGED_MODEL \
    --final-downsample $FINAL_DOWNSAMPLE \
    --use-layer-norm $USE_LAYER_NORM \
    --norm $NORM

echo "Done!"
