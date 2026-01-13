#!/usr/bin/env python3
# Copyright 2024 VietASR Project
"""
Stage 4: Cross-Model CER Filter

This script compares pseudo-labels (from Whisper) with Zipformer hypotheses
to filter out samples where the two models disagree significantly.

High CER = Models disagree = Likely errors (deletion, insertion, substitution)

Input: Stage 3 output (audio_path|text|duration|ppl)
       Zipformer decode results (recogs-*.txt)
Output: Filtered samples with CER scores

Usage:
    python filter_stage4_cross_model.py \
        --input /data/raw/pseudo/call_center/filtered/stage3.txt \
        --recogs /path/to/recogs-candidates.txt \
        --output /data/raw/pseudo/call_center/filtered/stage4.txt \
        --max-cer 0.15
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Stage 4: Cross-Model CER Filter"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to Stage 3 output (audio_path|text|duration|ppl)"
    )
    parser.add_argument(
        "--recogs",
        type=Path,
        required=True,
        help="Path to Zipformer decode results (recogs-*.txt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to output filtered transcripts"
    )
    parser.add_argument(
        "--max-cer", type=float, default=0.15,
        help="Maximum CER to keep (default: 0.15 = 15%%)"
    )
    
    return parser.parse_args()


def calculate_cer(ref: str, hyp: str) -> float:
    """Calculate Character Error Rate between reference and hypothesis."""
    import editdistance
    
    ref = ref.strip().lower().replace(" ", "")
    hyp = hyp.strip().lower().replace(" ", "")
    
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    
    distance = editdistance.eval(ref, hyp)
    return distance / len(ref)


def load_recogs(recogs_path: Path) -> Dict[str, str]:
    """
    Load Zipformer decode results.
    
    Expected format (icefall recogs format):
        cut_id    ref_text    hyp_text
    
    Returns: Dict[cut_id -> hyp_text]
    """
    recogs = {}
    
    with open(recogs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                cut_id = parts[0].strip()
                hyp_text = parts[2].strip()  # Third column is hypothesis
                recogs[cut_id] = hyp_text
            elif len(parts) == 2:
                # Sometimes format is: cut_id \t hyp_text
                cut_id = parts[0].strip()
                hyp_text = parts[1].strip()
                recogs[cut_id] = hyp_text
    
    return recogs


def extract_cut_id(audio_path: str) -> str:
    """Extract cut ID from audio path for matching with recogs."""
    # Get filename without extension
    filename = Path(audio_path).stem
    return filename


def main():
    args = get_args()
    
    # Check for editdistance
    try:
        import editdistance
    except ImportError:
        logging.error("editdistance not installed. Please run: pip install editdistance")
        return
    
    logging.info(f"Loading recogs from: {args.recogs}")
    recogs = load_recogs(args.recogs)
    logging.info(f"Loaded {len(recogs)} Zipformer hypotheses")
    
    if len(recogs) == 0:
        logging.error("No recogs loaded! Check the file format.")
        return
    
    logging.info(f"Processing input: {args.input}")
    logging.info(f"Max CER threshold: {args.max_cer}")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    passed = 0
    rejected_cer = 0
    no_match = 0
    
    cer_values = []
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) < 4:
                continue
            
            audio_path = parts[0].strip()
            pseudo_label = parts[1].strip()
            duration = parts[2].strip()
            ppl = parts[3].strip()
            
            total += 1
            cut_id = extract_cut_id(audio_path)
            
            if cut_id not in recogs:
                no_match += 1
                continue
            
            zipformer_hyp = recogs[cut_id]
            cer = calculate_cer(pseudo_label, zipformer_hyp)
            cer_values.append(cer)
            
            if cer <= args.max_cer:
                passed += 1
                # Output: audio|text|duration|ppl|cer
                fout.write(f"{audio_path}|{pseudo_label}|{duration}|{ppl}|{cer:.4f}\n")
            else:
                rejected_cer += 1
            
            if total % 50000 == 0:
                logging.info(f"Processed {total:,}, passed: {passed:,}")
    
    # Statistics
    logging.info("=" * 60)
    logging.info("STAGE 4 COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total input:      {total:,}")
    logging.info(f"No match in recogs: {no_match:,}")
    logging.info(f"Rejected (high CER): {rejected_cer:,}")
    logging.info(f"Passed:           {passed:,} ({passed/total*100:.1f}%)")
    logging.info(f"Output:           {args.output}")
    
    if cer_values:
        avg_cer = sum(cer_values) / len(cer_values)
        logging.info(f"\nCER statistics:")
        logging.info(f"  Average CER: {avg_cer:.4f}")
        logging.info(f"  Min CER: {min(cer_values):.4f}")
        logging.info(f"  Max CER: {max(cer_values):.4f}")


if __name__ == "__main__":
    main()
