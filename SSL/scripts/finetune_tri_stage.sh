#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# the --epoch argument was to identify pretraining model

export PYTHONPATH=${PWD}/zipformer_fbank:$PYTHONPATH

python zipformer_fbank/finetune_tri_stage.py \
    --world-size 8 \
    --num-epochs 200 \
    --start-epoch 1 \
    --use-fp16 1 \
    --sample-rate 100 \
    --encoder-dir exp/pretrain-iter1 \
    --epoch 10 \
    --avg 3 \
    --bpe-model ../ASR/data/unigram_5000.model \
    --train-file data/asr_45h/mgb2_cuts_train_0.jsonl.gz \
    --manifest-dir data/asr_45h \
    --exp-dir exp/finetune-iter1 \
    --max-duration 300 \
    --enable-musan 0 \
    --enable-spec-aug 0 \
    --mask-before-cnn 1 \
    --mask-prob 0.65 \
    --mask-channel-prob 0.5 \
    --mask-channel-length 20 \
    --accum-grad 1 \
    --seed 1556 \
    --base-lr 0.002 \
    --max-lr-update 30000 \
    --phase-ratio "(0.1, 0.4, 0.5)" \
    --final-downsample 1 \
    --causal 0 \
    --master-port 21497 \
    "${@}"  # pass remaining arguments

