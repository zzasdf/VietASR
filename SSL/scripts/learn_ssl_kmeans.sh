#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

python -m zipformer_fbank.zipformer_layer_feature.extract_kmeans_scripts.learn_kmeans \
    --km-path tem_data/kmeans_100h_kmeans_pretrain_epoch3.pt \
    --n-clusters 500 \
    --max-iter 100 \
    --files data/kmeans_manifest.jsonl.gz \
    --do-training \
    --pretrained-dir zipformer_fbank/exp-kmeans-all/epoch-3.pt \
    --max-duration 1000 \
    --mask-before-cnn 1 \
    --encoder-feature-layer 3 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model