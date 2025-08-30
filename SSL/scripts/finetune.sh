#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python zipformer_fbank/finetune.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --sample-rate 100 \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500/bpe.model \
    --exp-dir zipformer_fbank/exp_iter2_epoch45avg25_ft \
    --max-duration 1000 \
    --enable-musan 0 \
    --enable-spec-aug 0 \
    --mask-before-cnn 1 \
    --mask-prob 0.65 \
    --mask-channel-prob 0.5 \
    --mask-channel-length 20 \
    --accum-grad 1 \
    --base-lr 0.002 \
    --max-lr-update 80000 \
    --phase-ratio "(0.1, 0.4, 0.5)" \
    --pretrained-checkpoint-path zipformer_fbank/exp_iter2/epoch-45-avg-25.pt \
    --final-downsample 1 \
    --master-port 12356

