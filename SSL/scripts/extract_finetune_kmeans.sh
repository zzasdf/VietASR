#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

python zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py run \
    --task-list tem_data/tem_tem_list21 \
    --model-path tem_data/kmeans_100h_kmeans_ASR_50h_epoch3.pt \
    --pretrained-dir zipformer_fbank/exp-kmeans_ASR_50h-all/epoch-3.pt \
    --exp-dir zipformer_fbank/exp-kmeans_ASR_50h-all/exp-epoch-3-tri-stage-50h \
    --epoch 195 \
    --avg 1 \
    --max-duration 500 \
    --mask-before-cnn 1 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model \
    --checkpoint-type finetune \
    --final-downsample 1 \
    --use-averaged-model 0

