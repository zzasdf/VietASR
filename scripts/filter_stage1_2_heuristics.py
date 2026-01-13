#!/usr/bin/env python3
# Copyright 2024 VietASR Project
"""
Stage 1-2: Heuristics & Repetition Filter for Pseudo-labels

This script filters pseudo-labeled data based on:
- Stage 1: Basic heuristics (duration, word count, Vietnamese ratio)
- Stage 2: Repetition detection (anti-hallucination)

Input: transcripts.txt (format: audio_path|text|duration)
Output: filtered_stage1_2.txt (same format, only passing samples)

Usage:
    python filter_stage1_2_heuristics.py \
        --input /data/raw/pseudo/call_center/transcripts.txt \
        --output /data/raw/pseudo/call_center/filtered/stage1_2.txt \
        --stats-output /data/raw/pseudo/call_center/filtered/stage1_2_stats.json
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Vietnamese character pattern (including digits and spaces)
VIETNAMESE_CHARS = re.compile(
    r'[aàáạảãăằắặẳẵâầấậẩẫeèéẹẻẽêềếệểễiìíịỉĩoòóọỏõôồốộổỗơờớợởỡuùúụủũưừứựửữyỳýỵỷỹđ'
    r'AÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪEÈÉẸẺẼÊỀẾỆỂỄIÌÍỊỈĨOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠUÙÚỤỦŨƯỪỨỰỬỮYỲÝỴỶỸĐ'
    r'\s0-9]'
)

# Common filler words in Vietnamese call center
FILLER_WORDS = {'ừ', 'à', 'uh', 'um', 'hmm', 'ờ', 'ạ', 'dạ', 'vâng', 'vầng', 'ừm', 'hả', 'hở', 'ơ'}


def get_args():
    parser = argparse.ArgumentParser(
        description="Stage 1-2: Heuristics & Repetition Filter"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input transcripts.txt"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to output filtered transcripts"
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Path to save filtering statistics (JSON)"
    )
    # Stage 1 thresholds
    parser.add_argument("--min-duration", type=float, default=2.0,
                        help="Minimum audio duration in seconds (default: 2.0)")
    parser.add_argument("--max-duration", type=float, default=25.0,
                        help="Maximum audio duration in seconds (default: 25.0)")
    parser.add_argument("--min-words", type=int, default=3,
                        help="Minimum word count (default: 3)")
    parser.add_argument("--max-words", type=int, default=50,
                        help="Maximum word count (default: 50)")
    parser.add_argument("--min-viet-ratio", type=float, default=0.9,
                        help="Minimum Vietnamese character ratio (default: 0.9)")
    # Stage 2 thresholds
    parser.add_argument("--max-repeat-ratio", type=float, default=0.5,
                        help="Max ratio of most common word to total words (default: 0.5)")
    parser.add_argument("--max-repeat-count", type=int, default=3,
                        help="Max occurrences of most common word before flagging (default: 3)")
    
    return parser.parse_args()


def check_vietnamese_ratio(text: str) -> float:
    """Calculate ratio of Vietnamese characters in text."""
    text_clean = text.replace(" ", "")
    if len(text_clean) == 0:
        return 0.0
    viet_chars = len(VIETNAMESE_CHARS.findall(text.lower()))
    return viet_chars / len(text_clean)


def check_repetition(words: List[str], max_ratio: float, max_count: int) -> Tuple[bool, str]:
    """
    Check if text has suspicious repetition (hallucination indicator).
    
    Returns:
        (is_valid, reason)
    """
    if len(words) == 0:
        return False, "empty"
    
    word_counts = Counter(words)
    most_common_word, count = word_counts.most_common(1)[0]
    ratio = count / len(words)
    
    # Check repetition
    if count > max_count and ratio > max_ratio:
        return False, f"repetition:{most_common_word}x{count}"
    
    # Check if only filler words
    unique_words = set(w.lower() for w in words)
    if unique_words.issubset(FILLER_WORDS):
        return False, "filler_only"
    
    return True, "ok"


def filter_sample(
    audio_path: str,
    text: str,
    duration: float,
    args: argparse.Namespace
) -> Tuple[bool, str]:
    """
    Apply Stage 1-2 filters to a sample.
    
    Returns:
        (passed, reason)
    """
    # Stage 1: Duration check
    if duration < args.min_duration:
        return False, f"duration_short:{duration:.1f}s"
    if duration > args.max_duration:
        return False, f"duration_long:{duration:.1f}s"
    
    # Stage 1: Text quality
    text = text.strip()
    if not text:
        return False, "empty_text"
    
    words = text.split()
    word_count = len(words)
    
    if word_count < args.min_words:
        return False, f"words_few:{word_count}"
    if word_count > args.max_words:
        return False, f"words_many:{word_count}"
    
    # Stage 1: Vietnamese ratio
    viet_ratio = check_vietnamese_ratio(text)
    if viet_ratio < args.min_viet_ratio:
        return False, f"viet_ratio_low:{viet_ratio:.2f}"
    
    # Stage 2: Repetition check
    passed, reason = check_repetition(words, args.max_repeat_ratio, args.max_repeat_count)
    if not passed:
        return False, reason
    
    return True, "passed"


def main():
    args = get_args()
    
    logging.info(f"Stage 1-2 Filtering: {args.input}")
    logging.info(f"Thresholds: duration=[{args.min_duration}, {args.max_duration}], "
                 f"words=[{args.min_words}, {args.max_words}], viet_ratio>{args.min_viet_ratio}")
    
    # Statistics tracking
    stats = {
        "total": 0,
        "passed": 0,
        "rejected": 0,
        "reject_reasons": Counter(),
        "examples_rejected": []
    }
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) < 3:
                stats["reject_reasons"]["parse_error"] += 1
                continue
            
            audio_path = parts[0].strip()
            text = parts[1].strip()
            try:
                duration = float(parts[2].strip())
            except ValueError:
                duration = 0.0
            
            stats["total"] += 1
            
            passed, reason = filter_sample(audio_path, text, duration, args)
            
            if passed:
                stats["passed"] += 1
                fout.write(f"{audio_path}|{text}|{duration}\n")
            else:
                stats["rejected"] += 1
                stats["reject_reasons"][reason.split(":")[0]] += 1
                
                # Keep some examples for inspection
                if len(stats["examples_rejected"]) < 20:
                    stats["examples_rejected"].append({
                        "audio": audio_path,
                        "text": text[:80] + "..." if len(text) > 80 else text,
                        "reason": reason
                    })
            
            # Progress logging
            if line_num % 100000 == 0:
                logging.info(f"Processed {line_num:,} samples, "
                            f"passed: {stats['passed']:,} ({stats['passed']/stats['total']*100:.1f}%)")
    
    # Final stats
    pass_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
    logging.info("=" * 60)
    logging.info("FILTERING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total input:  {stats['total']:,}")
    logging.info(f"Passed:       {stats['passed']:,} ({pass_rate:.1f}%)")
    logging.info(f"Rejected:     {stats['rejected']:,} ({100-pass_rate:.1f}%)")
    logging.info(f"Output:       {args.output}")
    
    logging.info("\nRejection breakdown:")
    for reason, count in stats["reject_reasons"].most_common():
        pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
        logging.info(f"  {reason}: {count:,} ({pct:.1f}%)")
    
    # Save stats to JSON
    if args.stats_output:
        stats["reject_reasons"] = dict(stats["reject_reasons"])
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logging.info(f"\nStats saved to: {args.stats_output}")


if __name__ == "__main__":
    main()
