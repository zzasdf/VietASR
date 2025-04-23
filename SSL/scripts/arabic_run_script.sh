#!/bin/bash

# train ASR model with 44h paired data first.

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

# iteration 1, load ASR model and train kmeans on 33h paired data
./scripts/learn_vietASR_kmeans.sh

# extract kmeans labels of unsupervised data using the model above
./scripts/extract_vietASR_kmeans.sh  --max-duration 500 --num-workers 0 --file data/pretrain-4kh/pretraining_split_1.jsonl.gz --device 1

# pretrain HubertModel with the label extracted
./scripts/run_ssl.sh

# finetune HubertModel with 44h ASR data
./scripts/finetune_tri_stage.sh --max-lr-update 2000

# iteration 2, load ASR model and learn kmeans labels of unsupervised data
./scripts/learn_vietASR_kmeans.sh --max-duration 100 \
    --checkpoint-type finetune \
    --km-path exp/km-iter2 \
    --pretrained-dir exp/finetune-iter1 \
    --epoch 110 \
    --avg 10 \
    --src-dir data/kmeans-111h


# iteration 2, extract labels
./scripts/extract_vietASR_kmeans.sh --iteration 2 \
    --max-duration 100 --num-workers 0 \
    --checkpoint-type finetune \
    --km-path exp/km-iter2 \
    --pretrained-dir exp/finetune-iter1 \
    --epoch 110 \
    --avg 10 \
    --file data/pretrain/pretraining_split_0.jsonl.gz --device 0 

# iteration 2, pretrain
./scripts/run_ssl.sh --exp-dir exp/pretrain-iter2 --num-epochs 20 --iteration 2 --use-fp16 0

# iteration 2, finetune, should not use fp16 because grad_scale goes too small
./scripts/finetune_tri_stage.sh --use-fp16 0 --exp-dir exp/finetune-iter2 --encoder-dir exp/pretrain-iter2 --epoch 20 --avg 5

# iteration 3, learn kmeans model
./scripts/learn_vietASR_kmeans.sh --max-duration 100 \
    --checkpoint-type finetune \
    --km-path exp/km-iter3 \
    --pretrained-dir exp/finetune-iter2 \
    --epoch 70 \
    --avg 10 \
    --use-averaged-model 1 \
    --iteration 3

# iteration 3, extract kmeans model
./scripts/extract_vietASR_kmeans.sh --iteration 3 \
    --max-duration 100 --num-workers 0 \
    --checkpoint-type finetune \
    --km-path exp/km-iter3 \
    --pretrained-dir exp/finetune-iter2 \
    --epoch 70 \
    --avg 10 \
    --file data/pretrain/pretraining_split_0.jsonl.gz --device 0 


# iteration 3, pretrain with previously extracted label, should not use fp16 because grad_scale goes too small
./scripts/run_ssl.sh --use-fp16 0 --max-duration 400 --exp-dir exp/pretrain-iter3 --num-epochs 80 --iteration 3


# iteration 3, finetune, should not use fp16 because grad_scale goes too small
./scripts/finetune_tri_stage.sh --use-fp16 0 --exp-dir exp/finetune-iter3-epoch43 --encoder-dir exp/pretrain-iter3 --epoch 43 --avg 5 --world-size 4


# iteration 4, extract labels
./scripts/extract_vietASR_kmeans.sh --iteration 4 \
    --max-duration 100 --num-workers 0 \
    --checkpoint-type finetune \
    --km-path exp/km-iter4 \
    --pretrained-dir exp/finetune-iter3 \
    --epoch 60 \
    --avg 10 \
    --file data/pretrain/pretraining_split_0.jsonl.gz --device 0 

# iteration 4, pretrain with previously extracted label, should not use fp16 because grad_scale goes too small
./scripts/run_ssl.sh --use-fp16 0 --max-duration 500 --exp-dir exp/pretrain-iter4 --num-epochs 80 --iteration 4

./scripts/run_ssl.sh --max-duration 400 --exp-dir exp/4kh/pretrain-iter1 --num-epochs 50 --iteration 1 --manifest-dir data/pretrain-4kh

# decode ASR model:
./scripts/decode.sh --exp-dir exp/finetune-iter1 --epoch 10 --avg 3 --device 7
