#!/usr/bin/env python3
"""
Train BPE-level 4-gram LM
1. Tokenize corpus using BPE model
2. Train 4-gram LM using KenLM
"""

import argparse
import sentencepiece as spm
import os
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to normalized text corpus")
    parser.add_argument("--bpe-model", required=True, help="Path to bpe.model")
    parser.add_argument("--output-dir", required=True, help="Output directory for LM")
    parser.add_argument("--ngram", type=int, default=4, help="N-gram order")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Tokenize Corpus
    print(f"Loading BPE model: {args.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    tokenized_corpus = output_dir / "corpus_bpe.txt"
    print(f"Tokenizing {args.corpus} -> {tokenized_corpus}...")
    
    with open(args.corpus, 'r', encoding='utf-8') as f_in, \
         open(tokenized_corpus, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            # Encode as pieces (tokens)
            pieces = sp.encode_as_pieces(line)
            f_out.write(" ".join(pieces) + "\n")
            
    print("Tokenization complete.")
    
    # 2. Train KenLM
    arpa_file = output_dir / f"{args.ngram}gram.arpa"
    print(f"Training {args.ngram}-gram LM -> {arpa_file}...")
    
    # Check for lmplz
    lmplz_cmd = "lmplz"
    # Try to find lmplz in common paths if not in PATH
    if subprocess.call(["which", "lmplz"], stdout=subprocess.DEVNULL) != 0:
        # Fallback to known location in docker or environment
        # Assuming it's in PATH or we can't run it.
        # But wait, we are in a python script. We can try to use os.system
        pass

    cmd = f"lmplz -o {args.ngram} --text {tokenized_corpus} --arpa {arpa_file} --discount_fallback"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        print("Error: lmplz failed. Make sure KenLM is installed and 'lmplz' is in PATH.")
        # Try to use python kenlm if available? No, kenlm python bindings are for querying, not training usually.
        # We might need to use the docker container's lmplz.
        sys.exit(1)
        
    print(f"LM training complete: {arpa_file}")

if __name__ == "__main__":
    main()
