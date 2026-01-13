#!/usr/bin/env python3
"""
Tokenize corpus into BPE token IDs for LM training.
This is required because ArpaLmScorer expects token IDs, not words.
"""

import argparse
import sentencepiece as spm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tokenize corpus to BPE token IDs")
    parser.add_argument("--bpe-model", type=str, required=True, 
                        help="Path to BPE model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text file (one sentence per line)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file with token IDs")
    args = parser.parse_args()
    
    print(f"Loading BPE model from {args.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    print(f"Tokenizing {args.input}...")
    num_lines = 0
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            text = line.strip()
            if not text:
                continue
            
            # Encode to token IDs
            token_ids = sp.encode(text, out_type=int)
            
            # Write space-separated token IDs
            f_out.write(" ".join(map(str, token_ids)) + "\n")
            num_lines += 1
            
            if num_lines % 50000 == 0:
                print(f"  Processed {num_lines} lines...")
    
    print(f"✅ Done! Tokenized {num_lines} lines to {args.output}")
    
    # Show sample
    print("\n📋 Sample (first 5 lines):")
    with open(args.output, "r") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.strip()}")

if __name__ == "__main__":
    main()
