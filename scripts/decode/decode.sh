#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

# Normalize text before decoding
echo "Normalizing text in data_fbank/fbank..."
python normalize_cuts.py --data-dir fbank/tongdai_25112024 --cuts-name tongdai_25112024

# Decode
python SSL/zipformer_fbank/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir data4/exp \
    --max-duration 1000 \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --decoding-method modified_beam_search \
    --manifest-dir fbank/tongdai_25112024 \
    --use-averaged-model 1 \
    --final-downsample 1 \
    --cuts-name tongdai_25112024 \
    --use-layer-norm 0 \
    --beam-size 10 \
