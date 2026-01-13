#!/bin/bash
# Decode script with word-level KenLM N-best rescoring
# This uses the new --word-lm-path and --word-lm-scale options

export CUDA_VISIBLE_DEVICES=0

# ===== Configuration =====
EXP_DIR="data4/exp_finetune_v2"
EPOCH=7
AVG=5
BPE_MODEL="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"
MANIFEST_DIR="fbank/tongdai_clean"
CUTS_NAME="tongdai_clean"

# Word-level KenLM settings (in docker path)
WORD_LM_PATH="/data/raw/dolphin/lm_kenlm/4_gram_word_all.binary"
WORD_LM_SCALE=0.3  # Start with 0.3, tune in range 0.1-0.5

echo "=== Decoding with Word-level KenLM Rescoring ==="
echo "Model: $EXP_DIR epoch-${EPOCH} avg-${AVG}"
echo "Test set: $MANIFEST_DIR"
echo "Word LM: $WORD_LM_PATH (scale=$WORD_LM_SCALE)"
echo ""

python SSL/zipformer_fbank/decode.py \
    --epoch $EPOCH \
    --avg $AVG \
    --exp-dir "$EXP_DIR" \
    --max-duration 800 \
    --bpe-model "$BPE_MODEL" \
    --decoding-method modified_beam_search \
    --manifest-dir "$MANIFEST_DIR" \
    --use-averaged-model 1 \
    --final-downsample 1 \
    --use-layer-norm 0 \
    --beam-size 10 \
    --cuts-name "$CUTS_NAME" \
    --word-lm-path "$WORD_LM_PATH" \
    --word-lm-scale "$WORD_LM_SCALE"

echo ""
echo "=== Decode Complete ==="
echo "Results saved to: ${EXP_DIR}/modified_beam_search_use_avg/"
