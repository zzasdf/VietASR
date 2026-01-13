#!/usr/bin/env python3
# Copyright 2024 (Author: Antigravity)
"""
Analyze pseudo-label quality before filtering.

Provides statistics on:
- Text length distribution
- Vietnamese character ratio
- Duplicate detection
- Potential issues
"""

import argparse
import re
import hashlib
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Analyze pseudo-label quality")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    return parser.parse_args()


VIETNAMESE_CHARS = re.compile(
    r'[aàáạảãăằắặẳẵâầấậẩẫeèéẹẻẽêềếệểễiìíịỉĩoòóọỏõôồốộổỗơờớợởỡuùúụủũưừứựửữyỳýỵỷỹđ\s0-9]',
    re.IGNORECASE
)


def load_transcripts(path):
    transcripts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|', 1)
            if len(parts) == 2:
                transcripts.append((parts[0], parts[1]))
            else:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    transcripts.append((parts[0], parts[1]))
    return transcripts


def analyze(transcripts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    word_counts = []
    char_counts = []
    viet_ratios = []
    text_hashes = []
    issues = []
    
    for audio, text in transcripts:
        words = text.split()
        word_counts.append(len(words))
        char_counts.append(len(text))
        
        text_clean = text.replace(" ", "").lower()
        if len(text_clean) > 0:
            viet_chars = len(VIETNAMESE_CHARS.findall(text.lower()))
            ratio = viet_chars / len(text_clean)
            viet_ratios.append(ratio)
        
        text_hashes.append(hashlib.md5(text.lower().encode()).hexdigest())
        
        # Issue detection
        word_counter = Counter(words)
        if word_counter:
            most_common, count = word_counter.most_common(1)[0]
            if count > 3 and count / len(words) > 0.5:
                issues.append({"type": "repetition", "audio": audio, "text": text[:50]})
    
    # Duplicate analysis
    hash_counts = Counter(text_hashes)
    duplicates = {h: c for h, c in hash_counts.items() if c > 1}
    
    # Print summary
    print("=" * 60)
    print("PSEUDO-LABEL QUALITY ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(transcripts)}")
    print(f"\nWord count distribution:")
    print(f"  Min: {min(word_counts)}, Max: {max(word_counts)}, Mean: {np.mean(word_counts):.1f}")
    print(f"  < 3 words: {sum(1 for w in word_counts if w < 3)} samples")
    print(f"  > 50 words: {sum(1 for w in word_counts if w > 50)} samples")
    
    print(f"\nVietnamese character ratio:")
    print(f"  Mean: {np.mean(viet_ratios):.2%}")
    print(f"  < 90%: {sum(1 for r in viet_ratios if r < 0.9)} samples")
    
    print(f"\nDuplicates: {len(duplicates)} unique texts duplicated")
    print(f"  Total duplicate samples: {sum(duplicates.values()) - len(duplicates)}")
    
    print(f"\nIssues detected: {len(issues)} samples with repetition")
    
    # Save plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(word_counts, bins=50, edgecolor='black')
    axes[0, 0].axvline(3, color='r', linestyle='--', label='Min threshold')
    axes[0, 0].axvline(50, color='r', linestyle='--', label='Max threshold')
    axes[0, 0].set_xlabel('Word Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Word Count Distribution')
    axes[0, 0].legend()
    
    axes[0, 1].hist(viet_ratios, bins=50, edgecolor='black')
    axes[0, 1].axvline(0.9, color='r', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Vietnamese Char Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Vietnamese Character Ratio')
    axes[0, 1].legend()
    
    # Quality score estimation (simple)
    quality_scores = []
    for wc, vr in zip(word_counts, viet_ratios):
        score = 0
        if 3 <= wc <= 50:
            score += 0.5
        if vr >= 0.9:
            score += 0.5
        quality_scores.append(score)
    
    axes[1, 0].hist(quality_scores, bins=3, edgecolor='black')
    axes[1, 0].set_xlabel('Quality Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Quality Score Distribution')
    
    # Text length
    axes[1, 1].hist(char_counts, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Character Count')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Character Count Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pseudo_quality_analysis.png", dpi=150)
    print(f"\nSaved analysis plot to {output_dir / 'pseudo_quality_analysis.png'}")
    
    # Estimate filtered counts
    print("\n" + "=" * 60)
    print("ESTIMATED FILTERING RESULTS")
    print("=" * 60)
    
    passed = sum(1 for wc, vr in zip(word_counts, viet_ratios) if 3 <= wc <= 50 and vr >= 0.9)
    passed_no_dup = passed - (sum(duplicates.values()) - len(duplicates))
    
    print(f"Would pass text quality: {passed} ({passed/len(transcripts)*100:.1f}%)")
    print(f"After dedup: ~{passed_no_dup} ({passed_no_dup/len(transcripts)*100:.1f}%)")
    print(f"\nTarget 10%: {int(len(transcripts)*0.1)} samples")
    print(f"Target 15%: {int(len(transcripts)*0.15)} samples")
    print(f"Target 20%: {int(len(transcripts)*0.2)} samples")


def main():
    args = get_args()
    transcripts = load_transcripts(args.transcripts)
    analyze(transcripts, args.output_dir)


if __name__ == "__main__":
    main()
