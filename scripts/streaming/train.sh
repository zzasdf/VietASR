#!/bin/bash

# Fine-tune for Streaming ASR with Joint CTC-Transducer
# Optimized hyperparameters for call center domain
#
# Key features:
# - Causal (streaming) mode enabled
# - Joint CTC + Transducer training  
# - Multi-chunk training with full-context
# - Gradual encoder warmup for stable training

export CUDA_VISIBLE_DEVICES=0

EXP_DIR="data4/exp_streaming"
mkdir -p $EXP_DIR

echo "========================================"
echo "Starting Streaming Fine-tuning"
echo "Exp Dir: $EXP_DIR"
echo "========================================"

python SSL/zipformer_fbank/finetune.py \
    --world-size 1 \
    --exp-dir $EXP_DIR \
    --pretrained-checkpoint-path viet_iter3_pseudo_label/exp/epoch-12.pt \
    --pretrained-checkpoint-type ASR \
    --init-encoder-only 0 \
    --manifest-dir data4/fbank \
    --train-cuts vietASR_cuts_train.jsonl.gz \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    \
    --causal True \
    --chunk-size "16,32,64,-1" \
    --left-context-frames "64,128,256,-1" \
    \
    --use-transducer True \
    --use-ctc True \
    --ctc-loss-scale 0.3 \
    \
    --final-downsample 1 \
    --use-layer-norm 0 \
    \
    --base-lr 0.001 \
    --scheduler-type tri_stage \
    --phase-ratio "(0.1, 0.4, 0.5)" \
    --max-lr-update 20000 \
    \
    --num-epochs 30 \
    --max-duration 600 \
    --accum-grad 4 \
    --freeze-encoder-step 2500 \
    --warmup-encoder-step 5000 \
    \
    --enable-spec-aug True \
    --spec-aug-time-warp-factor 80 \
    --enable-musan True \
    \
    --use-fp16 True \
    --tensorboard True \
    2>&1 | tee $EXP_DIR/train.log

echo "Streaming Fine-tuning finished!"
