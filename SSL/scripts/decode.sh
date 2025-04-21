#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

python ./zipformer_fbank/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir zipformer_fbank/exp-kmeans_ASR_50h-init/exp-epoch-9-tri-stage-50h \
    --max-duration 1000 \
    --bpe-model data/lang_bpe_2000/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/fbank \
    --use-averaged-model 0 \
    --final-downsample 1 \
    --cuts-name all

