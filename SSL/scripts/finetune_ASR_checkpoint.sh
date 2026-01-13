#!/bin/bash

# =============================================================================
# Fine-tuning ASR Checkpoint Script - Optimized Version
# =============================================================================
# Điều chỉnh để tiếp tục training với LR thấp hơn giúp model hội tụ tốt hơn
# Sử dụng: bash SSL/scripts/finetune_ASR_checkpoint.sh
# =============================================================================

export CUDA_VISIBLE_DEVICES=2

# -----------------------------------------------------------------------------
# OPTION 1: Resume training với LR giảm (KHUYẾN NGHỊ cho trường hợp plateau)
# -----------------------------------------------------------------------------
python SSL/zipformer_fbank/finetune.py \
    --world-size 1 \
    --num-epochs 20 \
    --start-epoch 1 \
    --use-fp16 1 \
    --manifest-dir fbank \
    --train-cuts vietASR_cuts_train.jsonl.gz \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --exp-dir data4/exp_finetune_v4 \
    --max-duration 800 \
    --accum-grad 2 \
    \
    --enable-spec-aug 1 \
    --mask-prob 0.65 \
    --mask-channel-prob 0.5 \
    --mask-channel-length 20 \
    \
    --base-lr 0.001 \
    --scheduler-type tri_stage \
    --max-lr-update 50000 \
    --phase-ratio "(0.05, 0.15, 0.8)" \
    \
    --pretrained-checkpoint-path data4/exp_finetune_v3/best-valid-loss.pt \
    --pretrained-checkpoint-type finetune \
    --init-encoder-only 0 \
    --use-layer-norm 0 \
    --final-downsample 1 \
    --causal 0 \
    \
    --master-port 12358

# -----------------------------------------------------------------------------
# GIẢI THÍCH CÁC THAY ĐỔI:
# -----------------------------------------------------------------------------
# 1. --base-lr 0.001 (giảm từ 0.002)
#    → Giảm 50% LR giúp model fine-tune chi tiết hơn khi đã gần convergence
#
# 2. --phase-ratio "(0.05, 0.15, 0.8)" (thay vì 0.1, 0.4, 0.5)
#    → Warmup: 5% epochs (1 epoch)
#    → Constant: 15% epochs (3 epochs) 
#    → Decay: 80% epochs (16 epochs) - decay sớm và dài hơn
#
# 3. --num-epochs 20 (giảm từ 30)
#    → Đủ epochs cho fine-tuning phase mới
#
# 4. --exp-dir data4/exp_finetune_v4
#    → Lưu vào thư mục mới để so sánh với v3
#
# 5. --pretrained-checkpoint-path data4/exp_finetune_v3/best-valid-loss.pt
#    → Load từ checkpoint tốt nhất của training trước
# -----------------------------------------------------------------------------
