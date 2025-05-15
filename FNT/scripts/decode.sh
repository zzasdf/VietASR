#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH

python ./zipformer/decode.py \
    --decoding-method greedy_search \
    --epoch 100 \
    --avg 10 \
    --use-averaged-model 1 \
    --exp-dir exp/fnt-100h \
    --bpe-model ../ASR/data/unigram_5000.model \
    --manifest-dir data/devtest \
    --max-duration 100 \
    --num-workers 0 \
    --device 7 \
    --max-sym-per-frame 2 \
    "${@}"  # pass remaining arguments
