#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH

python zipformer_fbank/pretrain.py \
    --world-size 8 \
    --num-epochs 400 \
    --start-epoch 1 \
    --use-fp16 1 \
    --label-type kmeans_ASR_100h \
    --label-rate 50 \
    --sample-rate 100 \
    --exp-dir zipformer_fbank/exp-kmeans_ASR_100h-all \
    --max-duration 1000 \
    --train-cut large \
    --accum-grad 1 \
    --min-keep-size 200 \
    --mask-before-cnn 1 \
    --max-sample-size 1562 \
    --mask-prob 0.80 \
    --dropout-input 0.1 \
    --dropout-features 0.1 \
    --base-lr 0.045 \
    --save-every-n 15000 \
    --master-port 12356 \
    --manifest-dir data

