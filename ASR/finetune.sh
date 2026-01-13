#!/bin/bash
# Fine-tune Zipformer-30M-RNNT với pretrained weights
# Usage: bash finetune.sh [--pretrained-path PATH] [--exp-dir DIR]

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_PRETRAINED="/home/hiennt/VietASR/models/zipformer-30m-rnnt/pretrained.pt"
DEFAULT_EXP_DIR="/home/hiennt/VietASR/data4/exp_zipformer_finetune"
DEFAULT_BPE_MODEL="/home/hiennt/VietASR/models/zipformer-30m-rnnt/bpe.model"
DEFAULT_MANIFEST_DIR="/home/hiennt/VietASR/data4/fbank"

# Parse arguments
PRETRAINED_PATH="$DEFAULT_PRETRAINED"
EXP_DIR="$DEFAULT_EXP_DIR"
BPE_MODEL="$DEFAULT_BPE_MODEL"
MANIFEST_DIR="$DEFAULT_MANIFEST_DIR"
NUM_EPOCHS=10
WORLD_SIZE=1
START_EPOCH=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained-path)
            PRETRAINED_PATH="$2"
            shift 2
            ;;
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --bpe-model)
            BPE_MODEL="$2"
            shift 2
            ;;
        --manifest-dir)
            MANIFEST_DIR="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--pretrained-path PATH] [--exp-dir DIR] [--bpe-model PATH] [--num-epochs N]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH=/workspace/icefall:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "============================================"
echo "Zipformer-30M-RNNT Fine-tuning"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Pretrained model: $PRETRAINED_PATH"
echo "  Experiment dir:   $EXP_DIR"
echo "  BPE model:        $BPE_MODEL"
echo "  Manifest dir:     $MANIFEST_DIR"
echo "  Num epochs:       $NUM_EPOCHS"
echo "  World size:       $WORLD_SIZE"
echo ""

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Error: Pretrained model not found at $PRETRAINED_PATH"
    echo ""
    echo "Please run first:"
    echo "  python ASR/convert_jit_to_pretrained.py \\"
    echo "    --jit-path models/zipformer-30m-rnnt/jit_script.pt \\"
    echo "    --output-path $PRETRAINED_PATH"
    exit 1
fi

# Check BPE model
if [ ! -f "$BPE_MODEL" ]; then
    echo "Error: BPE model not found at $BPE_MODEL"
    exit 1
fi

# Check manifest directory
if [ ! -d "$MANIFEST_DIR" ]; then
    echo "Error: Manifest directory not found at $MANIFEST_DIR"
    echo ""
    echo "Please prepare data first with:"
    echo "  python ASR/local/compute_fbank.py \\"
    echo "    --manifest-dir data/manifests \\"
    echo "    --output-dir $MANIFEST_DIR"
    exit 1
fi

# Create experiment directory
mkdir -p "$EXP_DIR"

# =============================================================================
# Model Architecture Parameters (matching pretrained)
# =============================================================================

# These parameters MUST match the pretrained model architecture
# Zipformer-30M-RNNT configuration:
ENCODER_DIM="384,384,384,384,384"
ENCODER_UNMASKED_DIM="256,256,256,256,256"
FEEDFORWARD_DIM="1024,1024,1024,1024,1024"
NUM_ENCODER_LAYERS="2,2,2,2,2"
NUM_HEADS="4,4,4,4,4"
DECODER_DIM=512
JOINER_DIM=512

# =============================================================================
# Training Hyperparameters
# =============================================================================

# Learning rate (reduced for fine-tuning)
BASE_LR=0.0003  # 10x smaller than from-scratch training
LR_FACTOR=0.1

# Batch size (in seconds of audio)
MAX_DURATION=300

# Transducer parameters
USE_TRANSDUCER=True
USE_CTC=False
PRUNE_RANGE=5

# Checkpointing
SAVE_EVERY_N=1000
KEEP_LAST_K=5

# =============================================================================
# Fine-tuning Command
# =============================================================================

echo "============================================"
echo "Starting fine-tuning..."
echo "============================================"
echo ""

# Note: Assuming SSL/zipformer_fbank/finetune.py exists
# If not, we'll use ASR/zipformer/train.py with modifications

TRAIN_SCRIPT="SSL/zipformer_fbank/finetune.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Warning: $TRAIN_SCRIPT not found"
    echo "Using ASR/zipformer/train.py instead"
    TRAIN_SCRIPT="ASR/zipformer/train.py"
fi

python $TRAIN_SCRIPT \
    --world-size $WORLD_SIZE \
    --num-epochs $NUM_EPOCHS \
    --start-epoch $START_EPOCH \
    --use-fp16 1 \
    --exp-dir "$EXP_DIR" \
    --bpe-model "$BPE_MODEL" \
    --manifest-dir "$MANIFEST_DIR" \
    --max-duration $MAX_DURATION \
    --base-lr $BASE_LR \
    --lr-factor $LR_FACTOR \
    --use-transducer $USE_TRANSDUCER \
    --use-ctc $USE_CTC \
    --prune-range $PRUNE_RANGE \
    --encoder-dim $ENCODER_DIM \
    --encoder-unmasked-dim $ENCODER_UNMASKED_DIM \
    --feedforward-dim $FEEDFORWARD_DIM \
    --num-encoder-layers $NUM_ENCODER_LAYERS \
    --num-heads $NUM_HEADS \
    --decoder-dim $DECODER_DIM \
    --joiner-dim $JOINER_DIM \
    --save-every-n $SAVE_EVERY_N \
    --keep-last-k $KEEP_LAST_K \
    --pretrain-path "$PRETRAINED_PATH" \
    --pretrain-type ASR \
    --use-layer-norm 0 \
    2>&1 | tee "$EXP_DIR/train.log"

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo ""
echo "Logs saved to: $EXP_DIR/train.log"
echo "Checkpoints saved to: $EXP_DIR/"
echo ""
echo "Next steps:"
echo "1. Decode with fine-tuned model:"
echo "   python SSL/zipformer_fbank/decode.py \\"
echo "     --exp-dir $EXP_DIR \\"
echo "     --epoch $NUM_EPOCHS --avg 3 \\"
echo "     --bpe-model $BPE_MODEL \\"
echo "     --manifest-dir $MANIFEST_DIR"
echo ""
echo "2. Export to ONNX/TorchScript:"
echo "   python ASR/zipformer/export.py \\"
echo "     --exp-dir $EXP_DIR \\"
echo "     --epoch $NUM_EPOCHS --avg 3 \\"
echo "     --jit 1"
