#!/usr/bin/env python3
# Copyright 2024 (Author: Antigravity)
"""
Filter pseudo-labels based on existing Whisper WER results.

Input: wer.txt with format: path|pseudo|whisper_hyp|wer|duration
Output: candidates.txt with format: path|text|duration (same as transcripts.txt)

Based on Inter-ASR agreement approach from IDIAP INTERSPEECH 2025.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple


def get_args():
    parser = argparse.ArgumentParser(
        description="Filter pseudo-labels by Whisper WER threshold"
    )
    parser.add_argument(
        "--wer-file",
        type=Path,
        required=True,
        help="Path to wer.txt (format: path|pseudo|whisper_hyp|wer|duration)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for filtered transcripts",
    )
    parser.add_argument(
        "--max-wer",
        type=float,
        default=0.3,
        help="Maximum WER threshold. Default: 0.3 (30%%)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum audio duration. Default: 1.5s",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=25.0,
        help="Maximum audio duration. Default: 25.0s",
    )
    parser.add_argument(
        "--inspect-num",
        type=int,
        default=20,
        help="Number of samples to show for inspection",
    )
    return parser.parse_args()


def parse_wer_file(path: Path) -> List[Tuple[str, str, str, float, float]]:
    """
    Parse wer.txt file.
    Format: path|pseudo|whisper_hyp|wer|duration
    
    Returns: List of (path, pseudo, whisper_hyp, wer, duration)
    """
    samples = []
    errors = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 5:
                audio_path = parts[0].strip()
                pseudo = parts[1].strip()
                whisper_hyp = parts[2].strip()
                
                # Handle ERROR cases
                if parts[3].strip() == "ERROR":
                    errors += 1
                    continue
                
                try:
                    wer = float(parts[3].strip())
                    duration = float(parts[4].strip())
                except ValueError:
                    errors += 1
                    continue
                
                samples.append((audio_path, pseudo, whisper_hyp, wer, duration))
    
    logging.info(f"Parsed {len(samples)} valid samples, {errors} errors")
    return samples


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    logging.info(f"Loading WER file: {args.wer_file}")
    samples = parse_wer_file(args.wer_file)
    
    # Statistics
    stats = {
        "total": len(samples),
        "passed_wer": 0,
        "passed_duration": 0,
        "passed_all": 0,
        "rejected_wer": 0,
        "rejected_duration": 0,
    }
    
    # WER distribution
    wer_bins = {
        "0%": 0,
        "1-10%": 0,
        "11-20%": 0,
        "21-30%": 0,
        "31-50%": 0,
        ">50%": 0,
    }
    
    passed_samples = []
    rejected_examples = []
    
    logging.info(f"Filtering with WER ≤ {args.max_wer:.0%}, duration [{args.min_duration}-{args.max_duration}]s")
    
    for audio_path, pseudo, whisper_hyp, wer, duration in samples:
        # WER distribution
        if wer == 0:
            wer_bins["0%"] += 1
        elif wer <= 0.1:
            wer_bins["1-10%"] += 1
        elif wer <= 0.2:
            wer_bins["11-20%"] += 1
        elif wer <= 0.3:
            wer_bins["21-30%"] += 1
        elif wer <= 0.5:
            wer_bins["31-50%"] += 1
        else:
            wer_bins[">50%"] += 1
        
        # Filter by WER
        if wer > args.max_wer:
            stats["rejected_wer"] += 1
            if len(rejected_examples) < args.inspect_num:
                rejected_examples.append({
                    "audio": audio_path,
                    "pseudo": pseudo[:50],
                    "whisper": whisper_hyp[:50],
                    "wer": wer,
                    "reason": "wer"
                })
            continue
        
        stats["passed_wer"] += 1
        
        # Filter by duration
        if duration < args.min_duration or duration > args.max_duration:
            stats["rejected_duration"] += 1
            continue
        
        stats["passed_duration"] += 1
        stats["passed_all"] += 1
        
        # Keep sample with WER score for sorting later
        passed_samples.append((audio_path, pseudo, duration, wer))
    
    # Sort by WER (lower is better)
    passed_samples.sort(key=lambda x: x[3])
    
    # Save output
    logging.info(f"Saving {len(passed_samples)} samples to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for audio_path, pseudo, duration, _ in passed_samples:
            f.write(f"{audio_path}|{pseudo}|{duration}\n")
    
    # Print statistics
    logging.info("\n" + "=" * 60)
    logging.info("WHISPER WER FILTERING STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total input: {stats['total']}")
    logging.info(f"\nWER Distribution:")
    for bin_name, count in wer_bins.items():
        pct = count / stats['total'] * 100
        bar = "█" * int(pct / 2)
        logging.info(f"  {bin_name:8s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    logging.info(f"\nFiltering Results:")
    logging.info(f"  Passed WER (≤{args.max_wer:.0%}): {stats['passed_wer']} ({stats['passed_wer']/stats['total']*100:.1f}%)")
    logging.info(f"  Passed duration: {stats['passed_duration']}")
    logging.info(f"  Final output: {stats['passed_all']} ({stats['passed_all']/stats['total']*100:.1f}%)")
    
    # Show rejected examples
    if rejected_examples:
        logging.info("\n" + "=" * 60)
        logging.info("REJECTED SAMPLES (WER too high)")
        logging.info("=" * 60)
        for ex in rejected_examples[:10]:
            logging.info(f"\n[REJECTED] WER={ex['wer']:.1%}")
            logging.info(f"  Pseudo:  {ex['pseudo']}...")
            logging.info(f"  Whisper: {ex['whisper']}...")
    
    logging.info("\nDone!")


if __name__ == "__main__":
    main()
