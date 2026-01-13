#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

python SSL/zipformer_fbank/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir SSL/zipformer_fbank/exp_finetune \
    --max-duration 1000 \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --decoding-method  modified_beam_search \
    --manifest-dir fbank/tongdai_private
    --use-averaged-model 1 \
    --final-downsample 0 \
    --cuts-name call_center_private \
    --use-layer-norm 1 \
    --beam-size 10 \
    --norm 1
