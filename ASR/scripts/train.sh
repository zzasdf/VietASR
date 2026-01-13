#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,2

python ASR/zipformer/train.py \
    --world-size 2 \
    --num-epochs 50 \
    --start-epoch 1 \
    --use-fp16 1 \
    --train-cuts train \
    --manifest-dir data1/fbank \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --max-duration 600 \
    --enable-musan 0 \
    --exp-dir zipformer/exp_tongdai \
    --spec-aug-time-warp-factor 80 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --base-lr 0.0005 \
    --return-cuts 1 \
    --pretrain-path viet_iter3_pseudo_label/exp/epoch-12.pt \
    --pretrain-type ASR \
    --use-transducer 0 \
    --use-ctc 1 \
    --use-attention-decoder 1 \
    --ctc-loss-scale 0.3 \
    --attention-decoder-loss-scale 0.7 \
    --attention-decoder-num-layers 6 \
    --attention-decoder-dim 512 \
    --attention-decoder-num-heads 8 \
    --master-port 12356 \
