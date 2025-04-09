#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python zipformer/train.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --train-cuts 50h \
    --manifest-dir data/fbank \
    --bpe-model ${bpe_model} \
    --bpe-model data/lang_bpe_2000/bpe.model \
    --max-duration 1000 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356
