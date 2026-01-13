#!/bin/bash

# Script to benchmark streaming ASR model speed (RTF)
# Usage: bash benchmark_streaming.sh [tongdai_hard|tongdai_clean]

DATASET=${1:-tongdaiclean}  # Default to tongdai_hard
export CUDA_VISIBLE_DEVICES=2


# Check CUDA availability first
echo "=== Checking CUDA availability ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'N/A')"
echo "==================================="

# Only proceed if CUDA is available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✓ CUDA is available. Starting benchmark..."
    python SSL/zipformer_fbank/decode_rtfx.py \
        --epoch 30 \
        --avg 10 \
        --exp-dir data4/exp_streaming/ \
        --max-duration 1000 \
        --decoding-method modified_beam_search \
        --manifest-dir fbank/${DATASET}/ \
        --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
        --use-averaged-model 1 \
        --cuts-name tongdai_clean \
        --causal 1 \
        --chunk-size 32 \
        --left-context-frames 128 \
        --use-layer-norm 0 \
        --final-downsample 1 \
        --beam-size 10 \
        
else
    echo "✗ CUDA is NOT available! Model will run on CPU (very slow)."
    echo "Please check:"
    echo "  1. Docker container has GPU access (--gpus all)"
    echo "  2. NVIDIA Docker runtime is installed"
    echo "  3. PyTorch CUDA version matches system CUDA"
    exit 1
fi
