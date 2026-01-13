#!/bin/bash
# Download Zipformer-30M-RNNT-6000h model từ HuggingFace
# Usage: bash download_zipformer_model.sh [output_dir]

set -euo pipefail

# Default model directory
MODEL_DIR="${1:-/home/hiennt/VietASR/models/zipformer-30m-rnnt}"
MODEL_REPO="hynt/Zipformer-30M-RNNT-6000h"

echo "============================================"
echo "Downloading Zipformer-30M-RNNT Model"
echo "Repository: $MODEL_REPO"
echo "Output directory: $MODEL_DIR"
echo "============================================"

# Create directory
mkdir -p "$MODEL_DIR"

# Check if git-lfs is installed
if command -v git-lfs &> /dev/null; then
    echo ""
    echo "Method 1: Using git-lfs (Recommended)"
    echo "----------------------------------------"
    
    git lfs install
    
    # Clone the repository
    if [ -d "$MODEL_DIR/.git" ]; then
        echo "Model directory already exists, pulling latest changes..."
        cd "$MODEL_DIR"
        git pull
    else
        echo "Cloning model repository..."
        git clone https://huggingface.co/$MODEL_REPO "$MODEL_DIR"
    fi
    
elif command -v huggingface-cli &> /dev/null; then
    echo ""
    echo "Method 2: Using huggingface-cli"
    echo "----------------------------------------"
    
    huggingface-cli download $MODEL_REPO \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
        
else
    echo ""
    echo "Method 3: Using wget (manual download)"
    echo "----------------------------------------"
    echo "Neither git-lfs nor huggingface-cli found."
    echo "Please install one of them:"
    echo "  - git-lfs: apt-get install git-lfs"
    echo "  - huggingface-cli: pip install huggingface_hub"
    echo ""
    echo "Alternatively, download manually from:"
    echo "  https://huggingface.co/$MODEL_REPO/tree/main"
    echo ""
    echo "Required files:"
    echo "  - encoder-epoch-20-avg-10.onnx (~92MB)"
    echo "  - encoder-epoch-20-avg-10.int8.onnx (~27MB)"
    echo "  - decoder-epoch-20-avg-10.onnx (~5MB)"
    echo "  - joiner-epoch-20-avg-10.onnx (~4MB)"
    echo "  - bpe.model (~268KB)"
    echo "  - config.json (~23KB)"
    echo "  - jit_script.pt (~107MB)"
    exit 1
fi

echo ""
echo "============================================"
echo "Download complete!"
echo "============================================"
echo ""
echo "Files in $MODEL_DIR:"
ls -lh "$MODEL_DIR"

echo ""
echo "Next steps:"
echo "1. Verify files are downloaded correctly"
echo "2. Run demo: python ASR/demo_sherpa_onnx.py --audio test.wav"
echo "3. For fine-tuning, convert JIT model: python ASR/convert_jit_to_pretrained.py"
