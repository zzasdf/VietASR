#!/usr/bin/env python3
# Copyright 2024 VietASR Project
"""
Stage 5: Ranking & Top K% Selection

This script ranks all filtered samples by a combined quality score
and selects the top K% highest quality samples.

Score = w1 * (1 - CER) + w2 * (1 / log(PPL)) + w3 * duration_score

Input: Stage 4 output (audio_path|text|duration|ppl|cer)
Output: Final high-quality samples for training

Usage:
    python filter_stage5_ranking.py \
        --input /data/raw/pseudo/call_center/filtered/stage4.txt \
        --output /data/raw/pseudo/call_center/filtered/final_top20.txt \
        --top-k 0.2
"""

import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Stage 5: Ranking & Top K% Selection"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to Stage 4 output (audio_path|text|duration|ppl|cer)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to output final selected samples"
    )
    parser.add_argument(
        "--top-k", type=float, default=0.2,
        help="Percentage of top samples to keep (default: 0.2 = 20%%)"
    )
    # Score weights
    parser.add_argument("--w-cer", type=float, default=0.5,
                        help="Weight for CER score (default: 0.5)")
    parser.add_argument("--w-ppl", type=float, default=0.3,
                        help="Weight for PPL score (default: 0.3)")
    parser.add_argument("--w-duration", type=float, default=0.2,
                        help="Weight for duration score (default: 0.2)")
    
    return parser.parse_args()


def calculate_score(cer: float, ppl: float, duration: float,
                    w_cer: float, w_ppl: float, w_duration: float) -> float:
    """
    Calculate combined quality score.
    
    Higher score = better quality
    """
    # CER score: lower CER = higher score
    cer_score = 1.0 - min(cer, 1.0)
    
    # PPL score: lower PPL = higher score (with log normalization)
    # Typical PPL range: 100 - 100000
    # We use inverse log to normalize
    if ppl > 0:
        ppl_score = 1.0 / (1.0 + math.log10(max(ppl, 1.0)))
    else:
        ppl_score = 0.0
    
    # Duration score: prefer medium-length samples (5-15 seconds)
    optimal_duration = 10.0
    duration_score = 1.0 - min(abs(duration - optimal_duration) / optimal_duration, 1.0)
    
    # Combined score
    score = (w_cer * cer_score + 
             w_ppl * ppl_score + 
             w_duration * duration_score)
    
    return score


def main():
    args = get_args()
    
    logging.info(f"Loading samples from: {args.input}")
    logging.info(f"Weights: CER={args.w_cer}, PPL={args.w_ppl}, Duration={args.w_duration}")
    
    # Load all samples with scores
    samples = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) < 5:
                # If no CER (from Stage 3 directly), use default
                if len(parts) >= 4:
                    audio_path = parts[0].strip()
                    text = parts[1].strip()
                    duration = float(parts[2].strip())
                    ppl = float(parts[3].strip())
                    cer = 0.1  # Default CER if not available
                else:
                    continue
            else:
                audio_path = parts[0].strip()
                text = parts[1].strip()
                duration = float(parts[2].strip())
                ppl = float(parts[3].strip())
                cer = float(parts[4].strip())
            
            score = calculate_score(
                cer, ppl, duration,
                args.w_cer, args.w_ppl, args.w_duration
            )
            
            samples.append({
                'audio_path': audio_path,
                'text': text,
                'duration': duration,
                'ppl': ppl,
                'cer': cer,
                'score': score
            })
    
    logging.info(f"Loaded {len(samples):,} samples")
    
    # Sort by score (descending)
    samples.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top K%
    top_k_count = int(len(samples) * args.top_k)
    top_k_count = max(top_k_count, 1)  # At least 1
    
    selected = samples[:top_k_count]
    
    logging.info(f"Selecting top {args.top_k*100:.0f}%: {top_k_count:,} samples")
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in selected:
            # Output format: audio_path|text|duration (clean format for training)
            f.write(f"{s['audio_path']}|{s['text']}|{s['duration']}\n")
    
    # Also save a version with all scores for analysis
    score_output = args.output.with_suffix('.scores.txt')
    with open(score_output, 'w', encoding='utf-8') as f:
        for s in selected:
            f.write(f"{s['audio_path']}|{s['text']}|{s['duration']}|{s['ppl']:.2f}|{s['cer']:.4f}|{s['score']:.4f}\n")
    
    # Statistics
    logging.info("=" * 60)
    logging.info("STAGE 5 COMPLETE - FINAL OUTPUT")
    logging.info("=" * 60)
    logging.info(f"Total input:    {len(samples):,}")
    logging.info(f"Selected (Top {args.top_k*100:.0f}%): {len(selected):,}")
    logging.info(f"Output:         {args.output}")
    logging.info(f"Scores file:    {score_output}")
    
    if selected:
        scores = [s['score'] for s in selected]
        logging.info(f"\nScore statistics (selected samples):")
        logging.info(f"  Best score:  {max(scores):.4f}")
        logging.info(f"  Worst score: {min(scores):.4f}")
        logging.info(f"  Average:     {sum(scores)/len(scores):.4f}")
        
        total_duration = sum(s['duration'] for s in selected)
        logging.info(f"\nTotal audio duration: {total_duration/3600:.1f} hours")


if __name__ == "__main__":
    main()
