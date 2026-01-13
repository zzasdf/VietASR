#!/usr/bin/env python3
"""
Extract normalized text from Lhotse manifests for Language Model training.
"""

import gzip
import json
import argparse
from pathlib import Path
from typing import List
from collections import Counter


def extract_text_from_manifest(manifest_path: Path) -> List[str]:
    """Extract all text from a manifest file."""
    texts = []
    
    print(f"Reading {manifest_path}...")
    with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} cuts...")
            
            cut = json.loads(line)
            
            # Get text from supervisions
            if 'supervisions' in cut:
                for sup in cut['supervisions']:
                    if 'text' in sup and sup['text']:
                        text = sup['text'].strip()
                        if text:
                            texts.append(text)
    
    print(f"  Total: {len(texts)} utterances")
    return texts


def analyze_corpus(texts: List[str]) -> dict:
    """Analyze corpus statistics."""
    total_words = 0
    word_counter = Counter()
    
    for text in texts:
        words = text.split()
        total_words += len(words)
        word_counter.update(words)
    
    return {
        'num_utterances': len(texts),
        'total_words': total_words,
        'unique_words': len(word_counter),
        'avg_words_per_utt': total_words / len(texts) if texts else 0,
        'vocab_size': len(word_counter)
    }


def main():
    parser = argparse.ArgumentParser(description='Extract text from manifests for LM training')
    parser.add_argument('--manifest-dir', type=Path, default=Path('data4/fbank'),
                        help='Directory containing manifest files')
    parser.add_argument('--output-dir', type=Path, default=Path('data4/lm_corpus'),
                        help='Output directory for text files')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev'],
                        help='Splits to extract (train, dev, test)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_texts = []
    
    for split in args.splits:
        manifest_file = args.manifest_dir / f"vietASR_cuts_{split}.jsonl.gz"
        
        if not manifest_file.exists():
            print(f"Warning: {manifest_file} not found, skipping...")
            continue
        
        # Extract text
        texts = extract_text_from_manifest(manifest_file)
        
        # Save to file
        output_file = args.output_dir / f"{split}.txt"
        print(f"Writing to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Analyze
        stats = analyze_corpus(texts)
        print(f"\n{split.upper()} Statistics:")
        print(f"  Utterances: {stats['num_utterances']:,}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Unique words: {stats['unique_words']:,}")
        print(f"  Avg words/utt: {stats['avg_words_per_utt']:.2f}")
        print()
        
        all_texts.extend(texts)
    
    # Combine all splits
    if len(all_texts) > 0:
        combined_file = args.output_dir / "all.txt"
        print(f"Writing combined corpus to {combined_file}...")
        with open(combined_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n')
        
        # Final statistics
        final_stats = analyze_corpus(all_texts)
        print(f"\nCOMBINED Statistics:")
        print(f"  Utterances: {final_stats['num_utterances']:,}")
        print(f"  Total words: {final_stats['total_words']:,}")
        print(f"  Unique words: {final_stats['unique_words']:,}")
        print(f"  Avg words/utt: {final_stats['avg_words_per_utt']:.2f}")
        
        # Save vocabulary
        vocab_file = args.output_dir / "vocab.txt"
        print(f"\nSaving vocabulary to {vocab_file}...")
        word_counter = Counter()
        for text in all_texts:
            word_counter.update(text.split())
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word, count in word_counter.most_common():
                f.write(f"{word}\t{count}\n")
        
        print(f"Vocabulary size: {len(word_counter):,} words")
    
    print("\n✅ Done!")
    print(f"Output directory: {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()
