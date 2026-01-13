#!/bin/bash
# Export fine-tuned Zipformer model to TorchScript/ONNX
# Usage: bash export.sh [--exp-dir DIR] [--epoch N] [--avg K]

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_EXP_DIR="/home/hiennt/VietASR/data4/exp_zipformer_finetune"
DEFAULT_BPE_MODEL="/home/hiennt/VietASR/models/zipformer-30m-rnnt/bpe.model"
DEFAULT_OUTPUT_DIR="/home/hiennt/VietASR/models/zipformer-finetuned"

# Parse arguments
EXP_DIR="$DEFAULT_EXP_DIR"
BPE_MODEL="$DEFAULT_BPE_MODEL"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
EPOCH=10
AVG=3
EXPORT_JIT=1
EXPORT_ONNX=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --bpe-model)
            BPE_MODEL="$2"
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
        --export-jit)
            EXPORT_JIT="$2"
            shift 2
            ;;
        --export-onnx)
            EXPORT_ONNX="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--exp-dir DIR] [--epoch N] [--avg K] [--export-jit 1|0] [--export-onnx 1|0]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

export PYTHONPATH=/workspace/icefall:$PYTHONPATH

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "============================================"
echo "Zipformer Model Export"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Experiment dir:   $EXP_DIR"
echo "  Output dir:       $OUTPUT_DIR"
echo "  BPE model:        $BPE_MODEL"
echo "  Epoch:            $EPOCH"
echo "  Average:          $AVG"
echo "  Export JIT:       $EXPORT_JIT"
echo "  Export ONNX:      $EXPORT_ONNX"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Export Model
# =============================================================================

echo "============================================"
echo "Starting model export..."
echo "============================================"
echo ""

# Find export script
EXPORT_SCRIPT="SSL/zipformer_fbank/export.py"
if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "Warning: $EXPORT_SCRIPT not found, trying ASR/zipformer/export.py"
    EXPORT_SCRIPT="ASR/zipformer/export.py"
fi

if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "Error: No export script found!"
    echo "Please check if SSL/zipformer_fbank/export.py or ASR/zipformer/export.py exists"
    exit 1
fi

# Build tokens file from BPE model if needed
TOKENS_FILE="$OUTPUT_DIR/tokens.txt"
if [ ! -f "$TOKENS_FILE" ]; then
    echo "Creating tokens.txt from BPE model..."
    python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('$BPE_MODEL')
with open('$TOKENS_FILE', 'w') as f:
    for i in range(sp.vocab_size()):
        f.write(f'{sp.id_to_piece(i)}\n')
"
    echo "Tokens file created: $TOKENS_FILE"
fi

# Export command
python $EXPORT_SCRIPT \
    --exp-dir "$EXP_DIR" \
    --tokens "$TOKENS_FILE" \
    --epoch $EPOCH \
    --avg $AVG \
    --jit $EXPORT_JIT \
    --onnx $EXPORT_ONNX \
    --use-layer-norm 0 \
    2>&1 | tee "$OUTPUT_DIR/export.log"

# Copy exported files to output directory if they're in exp-dir
if [ "$EXPORT_JIT" = "1" ]; then
    for file in "$EXP_DIR"/jit_script*.pt; do
        if [ -f "$file" ]; then
            cp "$file" "$OUTPUT_DIR/"
            echo "Copied $(basename $file) to $OUTPUT_DIR/"
        fi
    done
fi

if [ "$EXPORT_ONNX" = "1" ]; then
    for file in "$EXP_DIR"/*.onnx; do
        if [ -f "$file" ]; then
            cp "$file" "$OUTPUT_DIR/"
            echo "Copied $(basename $file) to $OUTPUT_DIR/"
        fi
    done
fi

# Copy BPE model and tokens
cp "$BPE_MODEL" "$OUTPUT_DIR/"
echo "Copied BPE model to $OUTPUT_DIR/"

echo ""
echo "============================================"
echo "Export completed!"
echo "============================================"
echo ""
echo "Exported model files saved to: $OUTPUT_DIR"
echo ""

ls -lh "$OUTPUT_DIR"

echo ""
echo "Usage examples:"
echo ""

if [ "$EXPORT_JIT" = "1" ]; then
    echo "1. Test with TorchScript model:"
    echo "   python ASR/demo_sherpa_onnx.py \\"
    echo "     --model-dir $OUTPUT_DIR \\"
    echo "     --audio test.wav"
    echo ""
fi

if [ "$EXPORT_ONNX" = "1" ]; then
    echo "2. Use ONNX model with sherpa-onnx:"
    echo "   python ASR/demo_sherpa_onnx.py \\"
    echo "     --model-dir $OUTPUT_DIR \\"
    echo "     --audio test.wav"
    echo ""
fi

echo "3. Deploy with sherpa-onnx C++ inference:"
echo "   sherpa-onnx-offline \\"
echo "     --encoder=$OUTPUT_DIR/encoder.onnx \\"
echo "     --decoder=$OUTPUT_DIR/decoder.onnx \\"
echo "     --joiner=$OUTPUT_DIR/joiner.onnx \\"
echo "     --tokens=$OUTPUT_DIR/tokens.txt \\"
echo "     test.wav"
