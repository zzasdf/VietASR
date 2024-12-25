#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH

python zipformer/train.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --train-cut 50h_phone \
    --manifest-dir data/ssl_finetune \
    --bpe-model data/ssl_finetune/bpe_500/bpe.model \
    --exp-dir zipformer/exp-zipformer-50h \
    --max-duration 1000 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356
