#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

bpe_model=data/ssl_finetune/bpe_500/bpe.model # bpe model path

python zipformer/train.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --train-cut 50h \
    --manifest-dir data/ssl_finetune \
    --bpe-model ${bpe_model} \
    --exp-dir zipformer/exp-zipformer-50h \
    --max-duration 1000 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356
