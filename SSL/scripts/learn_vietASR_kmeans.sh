#!/bin/bash

# --km-path specify the save path for learned k-means model
# --n-clusters specify the number of k-means clusters
# --files specify the manifest used for k-means training
# --max-duration specify the maximum duration of audio files for each batch during extracting features
# --checkpoint-type specify the type of checkpoint to load
# --bpe-model specify the path to the BPE model, just to decode the output dim for ASR model

# --checkpoint-type specify the type of checkpoint to load, either "finetune" or "ASR" or "pretrain"
# # When checkpoint-type is "pretrain", this argument is the path to the pretrained model, --epoch --avg is not needed
# # When checkpoint-type is "ASR" or "finetuned", this argument is the directory of the ASR model, --epoch --avg is needed
# --pretrained-dir specify the directory of the pretrained model. 
# --epoch specify the epoch of the ASR model
# --avg specify the number of checkpoints to average

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

python -m zipformer_fbank.extract_kmeans_scripts.learn_kmeans \
    --km-path exp/km-iter1 \
    --n-clusters 500 \
    --max-iter 100 \
    --src-dir data/kmeans-111h \
    --do-training \
    --pretrained-dir exp/asr-100h \
    --epoch 60 \
    --avg 10 \
    --max-duration 100 \
    --checkpoint-type ASR \
    --use-averaged-model 1 \
    --bpe-model ../ASR/data/unigram_5000.model \
    "${@}"  # pass remaining arguments
