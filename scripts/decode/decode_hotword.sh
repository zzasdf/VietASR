#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

# Paths
EXP_DIR="data4/exp/"
BPE_MODEL="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"
MANIFEST_DIR="data6/fbank"
CONTEXT_FILE="data4/hotword.txt"
CONTEXT_SCORE=3.0

# Create sample hotwords file if it doesn't exist
if [ ! -f "$CONTEXT_FILE" ]; then
    echo "Creating sample hotwords file at $CONTEXT_FILE..."
    cat > "$CONTEXT_FILE" << EOF
viettel
vingroup
hà nội
hồ chí minh
chuyển đổi số
EOF
    echo "Created $CONTEXT_FILE"
fi

# Normalize text before decoding
echo "Normalizing text in $MANIFEST_DIR..."
python normalize_cuts.py --data-dir "$MANIFEST_DIR" --cuts-name test

echo "=== Decoding with Hotword Support ==="
echo "Hotwords: $CONTEXT_FILE"
echo "Score: $CONTEXT_SCORE"

# Decode
python SSL/zipformer_fbank/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir "$EXP_DIR" \
    --max-duration 1000 \
    --bpe-model "$BPE_MODEL" \
    --decoding-method modified_beam_search \
    --manifest-dir "$MANIFEST_DIR" \
    --use-averaged-model 1 \
    --final-downsample 1 \
    --cuts-name test \
    --use-layer-norm 0 \
    --beam-size 10 \
    --context-file "$CONTEXT_FILE" \
    --context-score "$CONTEXT_SCORE"
