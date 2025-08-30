#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

python ./zipformer_fbank/decode.py \
    --epoch $2\
    --avg $3 \
    --exp-dir zipformer_fbank/exp_iter2_epoch45avg25_ft \
    --max-duration 2000 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/fbank \
    --final-downsample 1 \
    --cuts-name "test"
