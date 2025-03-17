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


export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH
python -m zipformer_fbank.extract_kmeans_scripts.learn_kmeans \
    --km-path kmeans.pt \
    --n-clusters 500 \
    --max-iter 100 \
    --files data/kmeans_manifest_100h.jsonl.gz \
    --do-training \
    --pretrained-dir zipformer_fbank/exp-kmeans_ASR_50h-all/exp-epoch-3-tri-stage-50h \
    --epoch 195 \
    --avg 1 \
    --max-duration 1000 \
    --checkpoint-type finetune \
    --use-averaged-model 1 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model