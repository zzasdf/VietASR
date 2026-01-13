#!/usr/bin/env python3
"""
Generate lexicon.txt from vocab.txt and BPE model.
Format: word token1 token2 ...
"""

import argparse
import sentencepiece as spm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate lexicon from vocab and BPE model")
    parser.add_argument("--vocab", required=True, help="Path to vocab.txt (word\tcount)")
    parser.add_argument("--bpe-model", required=True, help="Path to bpe.model")
    parser.add_argument("--output", required=True, help="Output lexicon.txt")
    
    args = parser.parse_args()
    
    print(f"Loading BPE model from {args.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    print(f"Reading vocab from {args.vocab}")
    words = []
    with open(args.vocab, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                words.append(parts[0])
                
    print(f"Generating lexicon for {len(words)} words...")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        # Add <UNK> token
        f.write("<UNK> <unk>\n")
        
        for word in words:
            # Skip empty words
            if not word.strip():
                continue
                
            # Encode word to pieces
            pieces = sp.encode_as_pieces(word)
            
            # Write to file: word piece1 piece2 ...
            f.write(f"{word} {' '.join(pieces)}\n")
            
    print(f"Done. Lexicon written to {args.output}")

    # Generate words.txt (Symbol table)
    words_file = Path(args.output).parent / "words.txt"
    print(f"Generating words.txt to {words_file}...")
    with open(words_file, 'w', encoding='utf-8') as f:
        # k2 requires <eps> at 0
        f.write("<eps> 0\n")
        # Add <UNK>
        f.write("<UNK> 1\n")
        
        # Add other words starting from 2
        for i, word in enumerate(words):
            if not word.strip(): continue
            f.write(f"{word} {i+2}\n")
            
        # Add #0 for disambiguation (required for determinization/minimization usually, but for simple ARPA maybe not strictly if not disambig)
        # But k2 arpa2fst might use it. Let's add it to be safe if needed, or just stick to vocab.
        # For now, basic words.txt is enough.
        
    print(f"Done. Symbol table written to {words_file}")

if __name__ == "__main__":
    main()
