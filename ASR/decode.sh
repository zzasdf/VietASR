#!/bin/bash
# Decode với fine-tuned Zipformer model
# Usage: bash decode.sh [--exp-dir DIR] [--epoch N] [--avg K]

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_EXP_DIR="/home/hiennt/VietASR/data4/exp_zipformer_finetune"
DEFAULT_BPE_MODEL="/home/hiennt/VietASR/models/zipformer-30m-rnnt/bpe.model"
DEFAULT_MANIFEST_DIR="/home/hiennt/VietASR/data4/fbank"

# Parse arguments
EXP_DIR="$DEFAULT_EXP_DIR"
BPE_MODEL="$DEFAULT_BPE_MODEL"
MANIFEST_DIR="$DEFAULT_MANIFEST_DIR"
EPOCH=10
AVG=5
CUTS_NAME="test"
DECODING_METHOD="modified_beam_search"
BEAM_SIZE=10

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --avg)
            AVG="$2"
            shift 2
            ;;
        --cuts-name)
            CUTS_NAME="$2"
            shift 2
            ;;
        --decoding-method)
            DECODING_METHOD="$2"
            shift 2
            ;;
        --beam-size)
            BEAM_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--exp-dir DIR] [--epoch N] [--avg K] [--cuts-name NAME]"
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
echo "Zipformer Decoding"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Experiment dir:   $EXP_DIR"
echo "  BPE model:        $BPE_MODEL"
echo "  Manifest dir:     $MANIFEST_DIR"
echo "  Epoch:            $EPOCH"
echo "  Average:          $AVG"
echo "  Cuts name:        $CUTS_NAME"
echo "  Decoding method:  $DECODING_METHOD"
echo "  Beam size:        $BEAM_SIZE"
echo ""

# Check if checkpoint exists
if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory not found: $EXP_DIR"
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
    exit 1
fi

# =============================================================================
# Decode
# =============================================================================

echo "============================================"
echo "Starting decoding..."
echo "============================================"
echo ""

# Try SSL/zipformer_fbank/decode.py first, fallback to ASR/zipformer/decode.py
DECODE_SCRIPT="SSL/zipformer_fbank/decode.py"
if [ ! -f "$DECODE_SCRIPT" ]; then
    echo "Warning: $DECODE_SCRIPT not found, trying ASR/zipformer/decode.py"
    DECODE_SCRIPT="ASR/zipformer/decode.py"
fi

if [ ! -f "$DECODE_SCRIPT" ]; then
    echo "Error: No decode script found!"
    exit 1
fi

python $DECODE_SCRIPT \
    --epoch $EPOCH \
    --avg $AVG \
    --exp-dir "$EXP_DIR" \
    --bpe-model "$BPE_MODEL" \
    --manifest-dir "$MANIFEST_DIR" \
    --decoding-method "$DECODING_METHOD" \
    --beam-size $BEAM_SIZE \
    --use-layer-norm 0 \
    --cuts-name "$CUTS_NAME" \
    2>&1 | tee "$EXP_DIR/decode-${CUTS_NAME}-epoch${EPOCH}-avg${AVG}.log"

echo ""
echo "============================================"
echo "Decoding completed!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  Log: $EXP_DIR/decode-${CUTS_NAME}-epoch${EPOCH}-avg${AVG}.log"
echo "  Results: $EXP_DIR/recogs-${CUTS_NAME}-epoch${EPOCH}-avg${AVG}.txt"
echo ""

# Display WER if available
if [ -f "$EXP_DIR/recogs-${CUTS_NAME}-epoch${EPOCH}-avg${AVG}.txt" ]; then
    echo "Results summary:"
    tail -20 "$EXP_DIR/decode-${CUTS_NAME}-epoch${EPOCH}-avg${AVG}.log" | grep -E "(WER|CER|%)"
fi
