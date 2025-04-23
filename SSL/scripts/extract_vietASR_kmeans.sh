#! /bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

python zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
    --km-path exp/km-iter1 \
    --pretrained-dir ../ASR/zipformer/asr-100h \
    --epoch 60 \
    --avg 10 \
    --use-averaged-model 1 \
    --max-duration 100 \
    --bpe-model ../ASR/data/unigram_5000.model \
    --checkpoint-type ASR \
    --iteration 1 \
    "${@}"  # pass remaining arguments

# --src-dir data/unsupervised \

# --checkpoint-type specify the type of checkpoint to load, either "finetune" or "ASR" or "pretrain"
# # When checkpoint-type is "pretrain", this argument is the path to the pretrained model, --epoch --avg is not needed
# # When checkpoint-type is "ASR" or "finetuned", this argument is the directory of the ASR model, --epoch --avg is needed
# --pretrained-dir specify the directory of the pretrained model. 
# --epoch specify the epoch of the ASR model
# --avg specify the number of checkpoints to average