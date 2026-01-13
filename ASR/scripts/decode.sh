#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

python ./zipformer/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir ../zipformer/exp_tongdai \
    --max-duration 1000 \
    --bpe-model ../viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --decoding-method modified_beam_search \
    --manifest-dir ../data2/fbank \
    --use-averaged-model 1 \
    --cuts-name test  \
    --beam-size 4

