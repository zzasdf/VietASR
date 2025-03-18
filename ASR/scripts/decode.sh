#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

python ./zipformer/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir zipformer/exp \
    --max-duration 1000 \
    --bpe-model data/lang_bpe_2000/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/fbank \
    --use-averaged-model 1 \
    --cuts-name test # specify the cut to decode

