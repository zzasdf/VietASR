#!/usr/bin/env python3
"""
Data preparation script for fine-tuning with VLSP 2025 quality-focused strategies.

Implements:
- Conservative filtering (SoFarSoGood approach)
- Text normalization (hynguyenthien approach)
- Duration distribution balancing
"""

import argparse
import gzip
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List, Dict

from lhotse import RecordingSet, SupervisionSet, CutSet, load_manifest_lazy
from lhotse.recipes.utils import manifests_exist

# Vietnamese tone marks
VIETNAMESE_TONES = ['à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
                    'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
                    'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
                    'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
                    'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
                    'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',
                    'đ']


def has_vietnamese_tone(text: str) -> bool:
    """Check if text contains Vietnamese tone marks"""
    return any(tone in text.lower() for tone in VIETNAMESE_TONES)


def has_latin_or_emoji(text: str) -> bool:
    """Check if text contains non-Vietnamese Latin or emoji"""
    # Allow Vietnamese, numbers, spaces, basic punctuation
    allowed = re.compile(r'^[a-zA-ZÀ-ỹĐđ0-9\s\.,!?\-\'"]+$')
    return not allowed.match(text)


def simple_number_to_words(text: str) -> str:
    """
    Simple Vietnamese number normalization.
    For production, use num2words or vnnum2words library.
    """
    # Basic digit replacement (very simplified)
    replacements = {
        '0': 'không',
        '1': 'một',
        '2': 'hai',
        '3': 'ba',
        '4': 'bốn',
        '5': 'năm',
        '6': 'sáu',
        '7': 'bảy',
        '8': 'tám',
        '9': 'chín',
    }
    
    # Replace standalone digits
    result = text
    for digit, word in replacements.items():
        # Only replace standalone digits (not in larger numbers for now)
        result = re.sub(rf'\b{digit}\b', word, result)
    
    # Handle common patterns
    result = re.sub(r'(\d+)%', r'\1 phần trăm', result)
    result = re.sub(r'(\d+)km', r'\1 ki lô mét', result)
    
    return result


def filter_supervision_quality(text: str, 
                              min_duration: float,
                              max_duration: float,
                              duration: float) -> tuple[bool, str]:
    """
    Apply VLSP quality filters to a supervision.
    
    Returns: (is_valid, processed_text)
    """
    if not text or not text.strip():
        return False, ""
    
    text = text.strip()
    
    # 1. Check duration
    if duration < min_duration or duration > max_duration:
        return False, ""
    
    # 2. Require Vietnamese tone marks (SoFarSoGood)
    if not has_vietnamese_tone(text):
        return False, ""
    
    # 3. Check for Latin/emoji (SoFarSoGood)
    if has_latin_or_emoji(text):
        return False, ""
    
    # 4. Text length < 85 words (SoFarSoGood)
    word_count = len(text.split())
    if word_count > 85:
        return False, ""
    
    # 5. Normalize numbers (hynguyenthien)
    normalized_text = simple_number_to_words(text)
    
    return True, normalized_text


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare high-quality fine-tuning manifests with VLSP strategies"
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/data/manifest_k2ssl"),
        help="Input manifest directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests_finetune"),
        help="Output manifest directory"
    )
    
    parser.add_argument(
        "--call-center",
        type=str,
        default="call_center,call_center_ibs_report,tongdai,tongdai_02112022,tongdai_14112025,tongdai_25112024",
        help="Comma-separated list of call center datasets"
    )
    
    parser.add_argument(
        "--regional",
        type=str,
        default="ctv,ctv_09_12,ctv_12_2021,dong_nam_bo,regions_63,regions.bac_trung_bo_112024,bac_trung_bo",
        help="Comma-separated list of regional datasets"
    )
    
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum utterance duration in seconds"
    )
    
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum utterance duration in seconds"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Train/dev split ratio"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )
    
    return parser.parse_args()


def load_and_filter_dataset(dataset_path: Path, 
                            min_dur: float, 
                            max_dur: float) -> tuple[list, list, dict]:
    """
    Load and filter a dataset.
    
    Returns: (recordings, supervisions, stats)
    """
    # Find files
    rec_files = list(dataset_path.glob("recordings_*.jsonl.gz"))
    sup_files = [f for f in dataset_path.glob("supervisions_*.jsonl.gz") 
                 if "kmean" not in f.name]
    
    if not rec_files or not sup_files:
        return [], [], {}
    
    stats = {
        'total': 0,
        'passed_duration': 0,
        'passed_tone': 0,
        'passed_latin': 0,
        'passed_length': 0,
        'final_valid': 0
    }
    
    recordings = []
    supervisions = []
    
    # Load supervisions
    with gzip.open(sup_files[0], 'rt', encoding='utf-8') as f:
        for line in f:
            sup_dict = json.loads(line.strip())
            stats['total'] += 1
            
            text = sup_dict.get('text', '')
            duration = sup_dict.get('duration', 0)
            
            # Apply quality filters
            is_valid, normalized_text = filter_supervision_quality(
                text, min_dur, max_dur, duration
            )
            
            if is_valid:
                stats['final_valid'] += 1
                sup_dict['text'] = normalized_text
                supervisions.append(sup_dict)
    
    # Load corresponding recordings
    if supervisions:
        valid_rec_ids = {sup['recording_id'] for sup in supervisions}
        with gzip.open(rec_files[0], 'rt', encoding='utf-8') as f:
            for line in f:
                rec_dict = json.loads(line.strip())
                if rec_dict['id'] in valid_rec_ids:
                    recordings.append(rec_dict)
    
    return recordings, supervisions, stats


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dataset list
    call_center_list = [d.strip() for d in args.call_center.split(',')]
    regional_list = [d.strip() for d in args.regional.split(',')]
    
    all_recordings = []
    all_supervisions = []
    
    overall_stats = {
        'call_center': {'total': 0, 'valid': 0},
        'regional': {'total': 0, 'valid': 0}
    }
    
    # Process Call Center datasets
    logging.info("Processing Call Center datasets...")
    for dataset_name in call_center_list:
        dataset_path = args.manifest_dir / dataset_name
        if not dataset_path.exists():
            logging.warning(f"Dataset not found: {dataset_path}")
            continue
        
        logging.info(f"  Loading {dataset_name}...")
        recs, sups, stats = load_and_filter_dataset(
            dataset_path, args.min_duration, args.max_duration
        )
        
        if sups:
            all_recordings.extend(recs)
            all_supervisions.extend(sups)
            overall_stats['call_center']['total'] += stats['total']
            overall_stats['call_center']['valid'] += stats['final_valid']
            logging.info(f"    Kept {stats['final_valid']:,}/{stats['total']:,} cuts "
                        f"({stats['final_valid']/stats['total']*100:.1f}%)")
    
    # Process Regional datasets
    logging.info("Processing Regional datasets...")
    for dataset_name in regional_list:
        dataset_path = args.manifest_dir / dataset_name
        if not dataset_path.exists():
            logging.warning(f"Dataset not found: {dataset_path}")
            continue
        
        logging.info(f"  Loading {dataset_name}...")
        recs, sups, stats = load_and_filter_dataset(
            dataset_path, args.min_duration, args.max_duration
        )
        
        if sups:
            all_recordings.extend(recs)
            all_supervisions.extend(sups)
            overall_stats['regional']['total'] += stats['total']
            overall_stats['regional']['valid'] += stats['final_valid']
            logging.info(f"    Kept {stats['final_valid']:,}/{stats['total']:,} cuts "
                        f"({stats['final_valid']/stats['total']*100:.1f}%)")
    
    # Summary
    logging.info("=" * 80)
    logging.info("QUALITY FILTERING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Call Center: {overall_stats['call_center']['valid']:,} valid / "
                f"{overall_stats['call_center']['total']:,} total "
                f"({overall_stats['call_center']['valid']/overall_stats['call_center']['total']*100:.1f}%)")
    logging.info(f"Regional:    {overall_stats['regional']['valid']:,} valid / "
                f"{overall_stats['regional']['total']:,} total "
                f"({overall_stats['regional']['valid']/overall_stats['regional']['total']*100:.1f}%)")
    
    total_valid = overall_stats['call_center']['valid'] + overall_stats['regional']['valid']
    logging.info(f"Total:       {total_valid:,} high-quality cuts")
    
    # Create Lhotse manifests
    logging.info("Creating Lhotse manifests...")
    recording_set = RecordingSet.from_recordings([
        lhotse.Recording.from_dict(r) for r in all_recordings
    ])
    supervision_set = SupervisionSet.from_segments([
        lhotse.SupervisionSegment.from_dict(s) for s in all_supervisions
    ])
    
    # Shuffle and split
    random.seed(args.seed)
    indices = list(range(len(supervision_set)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * args.train_split)
    train_indices = set(indices[:split_idx])
    
    train_recordings = []
    train_supervisions = []
    dev_recordings = []
    dev_supervisions = []
    
    for i, (rec, sup) in enumerate(zip(recording_set, supervision_set)):
        if i in train_indices:
            train_recordings.append(rec)
            train_supervisions.append(sup)
        else:
            dev_recordings.append(rec)
            dev_supervisions.append(sup)
    
    # Save
    train_rec_set = RecordingSet.from_recordings(train_recordings)
    train_sup_set = SupervisionSet.from_segments(train_supervisions)
    dev_rec_set = RecordingSet.from_recordings(dev_recordings)
    dev_sup_set = SupervisionSet.from_segments(dev_supervisions)
    
    train_rec_set.to_file(output_dir / "vietASR_recordings_train.jsonl.gz")
    train_sup_set.to_file(output_dir / "vietASR_supervisions_train.jsonl.gz")
    dev_rec_set.to_file(output_dir / "vietASR_recordings_dev.jsonl.gz")
    dev_sup_set.to_file(output_dir / "vietASR_supervisions_dev.jsonl.gz")
    
    logging.info(f"Train: {len(train_supervisions):,} cuts")
    logging.info(f"Dev:   {len(dev_supervisions):,} cuts")
    logging.info(f"Saved to {output_dir}")
    logging.info("Done!")


if __name__ == "__main__":
    # Import lhotse here to avoid import errors if not installed
    import lhotse
    main()
