#!/usr/bin/env python3
# Copyright 2024 VietASR Project
"""
Stage 3: Language Model Perplexity Scorer

This script calculates perplexity (PPL) for each transcript using KenLM.
High PPL = unnatural/nonsensical text (hallucination type 2)
Very low PPL = suspicious repetition

Input: filtered transcripts from Stage 1-2
Output: transcripts with PPL scores for Stage 5 ranking

Usage:
    python filter_stage3_lm_score.py \
        --input /data/raw/pseudo/call_center/filtered/stage1_2.txt \
        --lm-path /data/lm/lm_4gram.arpa \
        --output /data/raw/pseudo/call_center/filtered/stage3_scored.txt
"""

import argparse
import logging
import math
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: LM Perplexity Scoring"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input transcripts (from Stage 1-2)"
    )
    parser.add_argument(
        "--lm-path",
        type=Path,
        required=True,
        help="Path to ARPA language model file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to output scored transcripts"
    )
    parser.add_argument(
        "--ppl-high-percentile", type=float, default=90,
        help="Remove samples above this PPL percentile (default: 90)"
    )
    parser.add_argument(
        "--ppl-low-percentile", type=float, default=5,
        help="Remove samples below this PPL percentile (default: 5)"
    )
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Try to import kenlm
    try:
        import kenlm
    except ImportError:
        logging.error("kenlm not installed. Please run: pip install kenlm")
        return
    
    logging.info(f"Loading LM from: {args.lm_path}")
    model = kenlm.Model(str(args.lm_path))
    logging.info(f"LM loaded. Order: {model.order}")
    
    logging.info(f"Processing: {args.input}")
    
    # First pass: Calculate PPL for all samples
    samples = []
    ppl_values = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            audio_path = parts[0].strip()
            text = parts[1].strip()
            duration = parts[2].strip()
            
            # Calculate perplexity
            # KenLM returns log10 probability, we convert to perplexity
            log_prob = model.score(text, bos=True, eos=True)
            word_count = len(text.split())
            
            if word_count > 0:
                # PPL = 10^(-log_prob / word_count)
                ppl = math.pow(10, -log_prob / word_count)
            else:
                ppl = float('inf')
            
            samples.append({
                'audio_path': audio_path,
                'text': text,
                'duration': duration,
                'ppl': ppl
            })
            ppl_values.append(ppl)
            
            if line_num % 100000 == 0:
                logging.info(f"Scored {line_num:,} samples...")
    
    logging.info(f"Total samples scored: {len(samples):,}")
    
    # Calculate percentiles
    ppl_values_sorted = sorted(ppl_values)
    n = len(ppl_values_sorted)
    
    low_idx = int(n * args.ppl_low_percentile / 100)
    high_idx = int(n * args.ppl_high_percentile / 100)
    
    ppl_low_threshold = ppl_values_sorted[low_idx] if low_idx < n else 0
    ppl_high_threshold = ppl_values_sorted[high_idx] if high_idx < n else float('inf')
    
    logging.info(f"PPL thresholds: low={ppl_low_threshold:.2f} (P{args.ppl_low_percentile}), "
                 f"high={ppl_high_threshold:.2f} (P{args.ppl_high_percentile})")
    
    # Second pass: Filter and write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    passed = 0
    rejected_low = 0
    rejected_high = 0
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in samples:
            ppl = sample['ppl']
            
            if ppl < ppl_low_threshold:
                rejected_low += 1
                continue
            if ppl > ppl_high_threshold:
                rejected_high += 1
                continue
            
            passed += 1
            # Output format: audio|text|duration|ppl
            f.write(f"{sample['audio_path']}|{sample['text']}|{sample['duration']}|{ppl:.2f}\n")
    
    logging.info("=" * 60)
    logging.info("STAGE 3 COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Input:        {len(samples):,}")
    logging.info(f"Passed:       {passed:,} ({passed/len(samples)*100:.1f}%)")
    logging.info(f"Rejected (low PPL):  {rejected_low:,} (repetitive)")
    logging.info(f"Rejected (high PPL): {rejected_high:,} (nonsensical)")
    logging.info(f"Output:       {args.output}")


if __name__ == "__main__":
    main()
