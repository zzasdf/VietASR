#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3

bpe_model=data/ssl_finetune/bpe_500/bpe.model # bpe model path

python ./zipformer/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir zipformer/exp-zipformer-50h_phone2 \
    --max-duration 1000 \
    --bpe-model data/ssl_finetune/vie_phonetisaurus_bpe_500/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/ssl_finetune \
    --use-averaged-model 1 \
    --cuts-name phone2 # specify the cut to decode

