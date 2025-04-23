#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

python ./zipformer/decode.py \
    --epoch $1 \
    --avg $2 \
    --use-averaged-model 1 \
    --exp-dir zipformer/asr-100h \
    --max-duration 300 \
    --bpe-model data/unigram_5000.model \
    --decoding-method greedy_search \
    --manifest-dir ../SSL/data/dataocean \
