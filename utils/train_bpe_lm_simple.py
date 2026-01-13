#!/usr/bin/env python3
"""
Train BPE-level 4-gram LM from existing corpus
Simplified version using existing corpus
"""

import argparse
import sentencepiece as spm
import sys
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to text corpus")
    parser.add_argument("--bpe-model", required=True, help="Path to bpe.model")
    parser.add_argument("--output", required=True, help="Output tokenized corpus")
    args = parser.parse_args()
    
    # Load BPE model
    print(f"Loading BPE model: {args.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    # Tokenize corpus
    print(f"Tokenizing {args.corpus} -> {args.output}...")
    
    line_count = 0
    with open(args.corpus, 'r', encoding='utf-8') as f_in:
        for _ in f_in:
            line_count += 1
    
    with open(args.corpus, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, total=line_count, desc="Tokenizing"):
            line = line.strip()
            if not line: continue
            pieces = sp.encode_as_pieces(line)
            f_out.write(" ".join(pieces) + "\n")
            
    print(f"Done. Tokenized corpus saved to {args.output}")

if __name__ == "__main__":
    main()
