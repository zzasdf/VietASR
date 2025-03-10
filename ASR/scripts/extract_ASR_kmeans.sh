#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0

python zipformer/extract_kmeans.py run \
    --task-list tem_data/kmeans_50h_list_dev \
    --model-path tem_data/kmeans.pt \
    --max-duration 300 \
    --bpe-model data/ssl_finetune/vie_bpe_500/bpe.model \
    --exp-dir ./zipformer/exp-zipformer-50h \
    --epoch 246 \
    --avg 5 \
    --use-averaged-model 0 
