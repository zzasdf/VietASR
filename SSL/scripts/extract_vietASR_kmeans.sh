#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

# --checkpoint-type specify the type of checkpoint to load, either "finetune" or "ASR" or "pretrain"
# # When checkpoint-type is "pretrain", this argument is the path to the pretrained model, --epoch --avg is not needed
# # When checkpoint-type is "ASR" or "finetuned", this argument is the directory of the ASR model, --epoch --avg is needed
# --pretrained-dir specify the directory of the pretrained model. 
# --epoch specify the epoch of the ASR model
# --avg specify the number of checkpoints to average

python zipformer_fbank/extract_kmeans_scripts/extract_kmeans.py \
    --task-list $2 \
    --model-path data/iter1_kmeans.pt \
    --pretrained-dir ../../icefall/egs/dataoceanai-alg/ASR/zipformer/exp_ws1_md1000_lrepochs100_cs1 \
    --epoch 60 \
    --avg 15 \
    --max-duration 1000 \
    --bpe-model ../../icefall/egs/dataoceanai-alg/ASR/data/lang_bpe_500/bpe.model \
    --checkpoint-type ASR \
    --use-averaged-model 1

python /root/busygpu/run.py
