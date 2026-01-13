#!/bin/bash

# Train large BPE-level 4-gram LM from /data/raw/train
# Run this script OUTSIDE Docker (it will exec into Docker)

set -euo pipefail

DOCKER_NAME="hiennt_vietasr_gpu_20251121T2348"
BPE_MODEL="/vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model"
CORPUS_FILE="data4/lm_corpus/corpus_large.txt"
BPE_CORPUS="data4/lm_corpus/corpus_large_bpe.txt"
OUTPUT_DIR="data4/lang_large"

echo "=== Step 1: Extract and normalize text from /data/raw/train ==="
docker exec -w /vietasr "$DOCKER_NAME" python utils/prepare_large_lm_corpus.py

echo ""
echo "=== Step 2: Tokenize with BPE ==="
docker exec -w /vietasr "$DOCKER_NAME" python -c "
import sentencepiece as spm
from tqdm import tqdm

sp = spm.SentencePieceProcessor()
sp.load('$BPE_MODEL')

print('Tokenizing...')
with open('$CORPUS_FILE', 'r') as f_in, open('$BPE_CORPUS', 'w') as f_out:
    for line in tqdm(f_in):
        line = line.strip()
        if not line: continue
        pieces = sp.encode_as_pieces(line)
        f_out.write(' '.join(pieces) + '\n')
print('Done: $BPE_CORPUS')
"

echo ""
echo "=== Step 3: Train 4-gram LM using icefall ==="
docker exec -w /vietasr "$DOCKER_NAME" mkdir -p "$OUTPUT_DIR"
docker exec -w /vietasr "$DOCKER_NAME" python -c "
import sys
sys.path.insert(0, '/workspace/icefall')
from icefall.shared.make_kn_lm import NgramCounts
import os

print('Training 4-gram KN LM...')
ngram_counts = NgramCounts(4)
ngram_counts.add_raw_counts_from_file('$BPE_CORPUS')
ngram_counts.cal_discounting_constants()
ngram_counts.cal_f()
ngram_counts.cal_bow()

output_lm = '$OUTPUT_DIR/4gram.arpa'
with open(output_lm, 'w', encoding='latin-1') as f:
    ngram_counts.print_as_arpa(fout=f)
    
print(f'LM saved to: {output_lm}')
"

echo ""
echo "=== Training complete! ==="
echo "Corpus: $CORPUS_FILE"
echo "BPE Corpus: $BPE_CORPUS"
echo "LM: $OUTPUT_DIR/4gram.arpa"
