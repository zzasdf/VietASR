#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

python -m zipformer_fbank.zipformer_layer_feature.extract_kmeans_scripts.extract_kmeans run \
    --task-list ${task_file} \
    --model-path ${model_path} \
    --pretrained-dir zipformer_fbank/exp-kmeans-all/epoch-3.pt \
    --max-duration 500 \
    --final-downsample 0 \
    --encoder-feature-layer 3 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model
