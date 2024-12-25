#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH

python ./zipformer_fbank/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir zipformer_fbank/exp-kmeans_ASR_50h-init/exp-epoch-9-tri-stage-50h \
    --max-duration 1000 \
    --mask-before-cnn 1 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/ssl_finetune \
    --use-averaged-model 0 \
    --final-downsample 1 \
    --use-layer-norm 0 \
    --cuts-name all

