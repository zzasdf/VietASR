#!/bin/bash

# Script to decode with Word-level N-gram LM using LG Graph
# Updated by Antigravity for Code-switching support

echo "═══════════════════════════════════════════════════════════"
echo "  ASR Decoding with LG Graph (Word-level LM)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Configuration
EXP_DIR="data4/exp_finetune_v3"
BPE_MODEL="/vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"
MANIFEST_DIR="data4/fbank"
LM_FILE="data4/lm/lm_4gram.arpa"
VOCAB_FILE="data4/lm_corpus/vocab.txt"
LANG_DIR="data4/lang"

# Decode parameters
EPOCH=19
AVG=17
BEAM=20.0
MAX_CONTEXTS=8
MAX_STATES=64
NGRAM_LM_SCALE=0.8  # Scale for Graph scores (tune this!)
NBEST_SCALE=0.5

# Ensure Lang Dir exists
mkdir -p $LANG_DIR

echo "📁 Configuration:"
echo "  Model: $EXP_DIR"
echo "  LM: $LM_FILE"
echo "  Lang Dir: $LANG_DIR"
echo ""

# 1. Generate Lexicon & Symbol Table
if [ ! -f "$LANG_DIR/lexicon.txt" ] || [ ! -f "$LANG_DIR/words.txt" ]; then
    echo "⚙️  Generating Lexicon and Symbol Table..."
    
    # Copy tokens.txt (required by icefall Lexicon)
    cp /vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt $LANG_DIR/tokens.txt
    
    docker exec -w /vietasr hiennt_vietasr_gpu_20251121T2348 python utils/generate_lexicon.py \
        --vocab $VOCAB_FILE \
        --bpe-model $BPE_MODEL \
        --output $LANG_DIR/lexicon.txt
else
    echo "✅ Lexicon found."
fi

# 2. Compile LG Graph
if [ ! -f "$LANG_DIR/LG.pt" ]; then
    echo "⚙️  Compiling LG Graph..."
    docker exec -w /vietasr hiennt_vietasr_gpu_20251121T2348 python utils/compile_lg.py \
        --lang-dir $LANG_DIR \
        --lm-path $LM_FILE \
        --output-dir $LANG_DIR
else
    echo "✅ LG Graph found."
fi

echo ""
echo "🚀 Running decoding with fast_beam_search_nbest_LG..."
echo ""

docker exec -w /vietasr hiennt_vietasr_gpu_20251121T2348 python ASR/zipformer/decode.py \
  --epoch $EPOCH \
  --avg $AVG \
  --exp-dir $EXP_DIR \
  --bpe-model $BPE_MODEL \
  --manifest-dir $MANIFEST_DIR \
  --decoding-method fast_beam_search_nbest_LG \
  --lang-dir $LANG_DIR \
  --beam $BEAM \
  --max-contexts $MAX_CONTEXTS \
  --max-states $MAX_STATES \
  --ngram-lm-scale $NGRAM_LM_SCALE \
  --nbest-scale $NBEST_SCALE \
  --num-paths 200 \
  --cuts-name dev

echo ""
echo "✅ Decoding completed!"
echo ""
echo "📊 Check results in:"
echo "  $EXP_DIR/*/recogs-dev-*.txt"
echo "  $EXP_DIR/*/errs-dev-*.txt"
