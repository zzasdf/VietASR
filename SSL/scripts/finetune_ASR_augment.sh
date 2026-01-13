#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=2

python SSL/zipformer_fbank/finetune.py \
    --world-size 1\
    --num-epochs 40 \
    --start-epoch 2 \
    --use-fp16 1 \
    --sample-rate 100 \
    --manifest-dir data/fbank_finetune \
    --bpe-model   viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --exp-dir zipformer_fbank/exp7_augment \
    --max-duration 800 \
    --enable-musan 1 \
    --enable-spec-aug 1 \
    --mask-before-cnn 1 \
    --mask-prob 0.65 \
    --mask-channel-prob 0.5 \
    --mask-channel-length 20 \
    --accum-grad 2 \
    --seed 1556 \
    --base-lr 0.005 \
    --max-lr-update 80000 \
    --phase-ratio "(0.05, 0.45, 0.5)" \
    --pretrained-checkpoint-path SSL/zipformer_fbank/exp/epoch-30.pt \
    --pretrained-checkpoint-type SSL\
    --init-encoder-only 1 \
    --num-classes 2000 \
    --use-layer-norm 0 \
    --final-downsample 1 \
    --causal 1 \
    --chunk-size 16,32,64 \
    --left-context-frames 64,128,256 \
    --master-port 12356
