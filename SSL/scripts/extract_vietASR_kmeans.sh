#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

# --checkpoint-type specify the type of checkpoint to load, either "finetune" or "ASR" or "pretrain"
# # When checkpoint-type is "pretrain", this argument is the path to the pretrained model, --epoch --avg is not needed
# # When checkpoint-type is "ASR" or "finetuned", this argument is the directory of the ASR model, --epoch --avg is needed
# --pretrained-dir specify the directory of the pretrained model. 
# --epoch specify the epoch of the ASR model
# --avg specify the number of checkpoints to average

python zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
    --task-list tem_data/tem_tem_list21 \
    --model-path tem_data/kmeans_100h_kmeans_ASR_50h_epoch3.pt \
    --pretrained-dir zipformer_fbank/exp-kmeans_ASR_50h-all/exp-epoch-3-tri-stage-50h \
    --epoch 195 \
    --avg 1 \
    --max-duration 500 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model \
    --checkpoint-type finetune \
    --use-averaged-model 1

