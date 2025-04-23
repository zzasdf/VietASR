#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python ./zipformer_fbank/decode.py \
    --decoding-method greedy_search \
    --epoch 30 \
    --avg 5 \
    --use-averaged-model 1 \
    --exp-dir exp/finetune-iter1 \
    --bpe-model ../ASR/data/unigram_5000.model \
    --manifest-dir data/devtest \
    --max-duration 300 \
    --final-downsample 1 \
    --use-layer-norm 1 \
    --num-workers 0 \
    --device 7 \
    "${@}"  # pass remaining arguments
