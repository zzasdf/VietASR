#!/usr/bin/env python3
# Copyright 2024 (Author: Antigravity)
"""
Multi-criteria filtering script for pseudo-labeled data.

Based on papers:
- "Improved Noisy Student Training for ASR" (Google, INTERSPEECH 2020)
- "slimIPL: Language-Model-Free Iterative Pseudo-Labeling" (Meta, INTERSPEECH 2022)
- "Efficient Data Selection for Domain Adaptation of ASR" (IDIAP, INTERSPEECH 2025)

Filtering criteria:
1. Duration: 2-25 seconds
2. Text quality: word count, Vietnamese character ratio
3. WER against reference (if available)
4. Deduplicate transcripts
"""

import argparse
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import jiwer
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Multi-criteria filtering for pseudo-labeled ASR data."
    )
    parser.add_argument(
        "--transcripts-in",
        type=Path,
        required=True,
        help="Path to input transcripts.txt (format: audio_path|transcript)",
    )
    parser.add_argument(
        "--transcripts-out",
        type=Path,
        required=True,
        help="Path to save filtered transcripts.",
    )
    # Removed --audio-dir since duration is in transcripts.txt
    parser.add_argument(
        "--reference-transcripts",
        type=Path,
        help="Path to human-labeled transcripts for WER comparison (optional).",
    )
    
    # Duration thresholds
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum audio duration in seconds. Default: 2.0",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=25.0,
        help="Maximum audio duration in seconds. Default: 25.0",
    )
    
    # Text quality thresholds
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum word count. Default: 3",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50,
        help="Maximum word count. Default: 50",
    )
    parser.add_argument(
        "--min-viet-ratio",
        type=float,
        default=0.9,
        help="Minimum ratio of Vietnamese characters. Default: 0.9",
    )
    
    # WER threshold (for reference comparison)
    parser.add_argument(
        "--max-wer",
        type=float,
        default=0.3,
        help="Maximum WER against reference (if provided). Default: 0.3 (30%%)",
    )
    
    # Deduplication
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate transcripts.",
    )
    
    # Target retention
    parser.add_argument(
        "--target-retention",
        type=float,
        default=0.2,
        help="Target percentage to keep (0.1 = 10%%, 0.2 = 20%%). Default: 0.2",
    )
    
    parser.add_argument(
        "--inspect-num",
        type=int,
        default=10,
        help="Number of rejected samples to show. Default: 10",
    )
    
    return parser.parse_args()


# Vietnamese character pattern
VIETNAMESE_CHARS = re.compile(
    r'[aàáạảãăằắặẳẵâầấậẩẫeèéẹẻẽêềếệểễiìíịỉĩoòóọỏõôồốộổỗơờớợởỡuùúụủũưừứựửữyỳýỵỷỹđ'
    r'AÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪEÈÉẸẺẼÊỀẾỆỂỄIÌÍỊỈĨOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠUÙÚỤỦŨƯỪỨỰỬỮYỲÝỴỶỸĐ'
    r'\s0-9]'
)


def check_text_quality(text: str, min_words: int, max_words: int, min_viet_ratio: float) -> Tuple[bool, str]:
    """
    Check text quality based on multiple criteria.
    
    Returns:
        (passed: bool, reason: str)
    """
    if not text or not text.strip():
        return False, "empty_text"
    
    # Word count check
    words = text.strip().split()
    word_count = len(words)
    
    if word_count < min_words:
        return False, f"too_few_words:{word_count}"
    
    if word_count > max_words:
        return False, f"too_many_words:{word_count}"
    
    # Vietnamese character ratio
    text_clean = text.replace(" ", "")
    if len(text_clean) > 0:
        viet_chars = len(VIETNAMESE_CHARS.findall(text.lower()))
        total_chars = len(text_clean)
        viet_ratio = viet_chars / total_chars
        
        if viet_ratio < min_viet_ratio:
            return False, f"low_viet_ratio:{viet_ratio:.2f}"
    
    # Check for repeated words (hallucination detection)
    word_counts = Counter(words)
    most_common_word, most_common_count = word_counts.most_common(1)[0]
    if most_common_count > 3 and most_common_count / word_count > 0.5:
        return False, f"repeated_word:{most_common_word}x{most_common_count}"
    
    # Check for filler-only transcripts
    filler_words = {'ừ', 'à', 'uh', 'um', 'hmm', 'ờ', 'ạ', 'dạ', 'vâng'}
    if set(w.lower() for w in words).issubset(filler_words):
        return False, "filler_only"
    
    return True, "passed"


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds."""
    try:
        import torchaudio
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return None


def load_transcripts(path: Path) -> List[Tuple[str, str, float]]:
    """Load transcripts from file. Format: audio_path|transcript|duration"""
    transcripts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                # Format: path | text | duration
                audio_path = parts[0].strip()
                text = parts[1].strip()
                try:
                    duration = float(parts[2].strip())
                except ValueError:
                    duration = 0.0
                transcripts.append((audio_path, text, duration))
            elif len(parts) == 2:
                # Fallback: path | text (no duration)
                audio_path, text = parts
                transcripts.append((audio_path.strip(), text.strip(), 0.0))
    return transcripts


def load_reference_transcripts(path: Path) -> Dict[str, str]:
    """Load reference transcripts into a dict keyed by audio filename."""
    refs = {}
    for audio_path, text, _ in load_transcripts(path):
        filename = Path(audio_path).stem
        refs[filename] = text
    return refs


def calculate_wer(ref: str, hyp: str) -> float:
    """Calculate WER between reference and hypothesis."""
    if not ref or not hyp:
        return 1.0
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    try:
        return jiwer.wer(
            ref, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
    except Exception:
        return 1.0


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    logging.info(f"Loading transcripts from {args.transcripts_in}")
    transcripts = load_transcripts(args.transcripts_in)
    logging.info(f"Loaded {len(transcripts)} samples.")
    
    # Load reference transcripts if provided
    ref_transcripts = {}
    if args.reference_transcripts:
        logging.info(f"Loading reference transcripts from {args.reference_transcripts}")
        ref_transcripts = load_reference_transcripts(args.reference_transcripts)
        logging.info(f"Loaded {len(ref_transcripts)} reference samples.")
    
    # Statistics
    stats = {
        "total": len(transcripts),
        "passed_text": 0,
        "passed_duration": 0,
        "passed_wer": 0,
        "passed_all": 0,
        "rejected": {
            "text_quality": [],
            "duration": [],
            "wer": [],
            "duplicate": []
        }
    }
    
    passed_samples = []
    sample_scores = []  # For ranking
    seen_texts = set()  # For deduplication
    
    logging.info("Filtering samples...")
    for audio_path, text, duration in tqdm(transcripts):
        sample_passed = True
        sample_score = 0.0
        reject_reasons = []
        
        # 1. Text quality check
        text_passed, reason = check_text_quality(
            text, args.min_words, args.max_words, args.min_viet_ratio
        )
        
        if not text_passed:
            sample_passed = False
            reject_reasons.append(f"text:{reason}")
            if len(stats["rejected"]["text_quality"]) < args.inspect_num:
                stats["rejected"]["text_quality"].append({
                    "audio": audio_path,
                    "text": text[:100],
                    "reason": reason
                })
        else:
            stats["passed_text"] += 1
            sample_score += 1.0
        
        # 2. Duration check (use duration from file)
        if sample_passed and duration > 0:
            if duration < args.min_duration or duration > args.max_duration:
                sample_passed = False
                reject_reasons.append(f"duration:{duration:.1f}s")
                if len(stats["rejected"]["duration"]) < args.inspect_num:
                    stats["rejected"]["duration"].append({
                        "audio": audio_path,
                        "duration": duration
                    })
            else:
                stats["passed_duration"] += 1
                # Prefer medium-length samples
                optimal_duration = 10.0
                duration_score = 1.0 - abs(duration - optimal_duration) / optimal_duration
                sample_score += max(0, duration_score)
        
        # 3. WER check against reference (if provided)
        if ref_transcripts and sample_passed:
            filename = Path(audio_path).stem
            if filename in ref_transcripts:
                ref_text = ref_transcripts[filename]
                wer = calculate_wer(ref_text, text)
                
                if wer > args.max_wer:
                    sample_passed = False
                    reject_reasons.append(f"wer:{wer:.2%}")
                    if len(stats["rejected"]["wer"]) < args.inspect_num:
                        stats["rejected"]["wer"].append({
                            "audio": audio_path,
                            "pseudo": text[:50],
                            "ref": ref_text[:50],
                            "wer": wer
                        })
                else:
                    stats["passed_wer"] += 1
                    # Lower WER = higher score
                    sample_score += (1.0 - wer)
        
        # 4. Deduplication
        if args.deduplicate and sample_passed:
            text_hash = hashlib.md5(text.lower().encode()).hexdigest()
            if text_hash in seen_texts:
                sample_passed = False
                reject_reasons.append("duplicate")
                if len(stats["rejected"]["duplicate"]) < args.inspect_num:
                    stats["rejected"]["duplicate"].append({
                        "audio": audio_path,
                        "text": text[:50]
                    })
            else:
                seen_texts.add(text_hash)
        
        if sample_passed:
            stats["passed_all"] += 1
            passed_samples.append((audio_path, text, duration, sample_score))
    
    # Sort by score and apply target retention
    passed_samples.sort(key=lambda x: x[3], reverse=True)
    
    target_count = int(stats["total"] * args.target_retention)
    target_count = max(target_count, min(len(passed_samples), 100))  # At least 100 or all passed
    
    final_samples = passed_samples[:target_count]
    
    # Save results
    logging.info(f"Saving {len(final_samples)} samples to {args.transcripts_out}")
    args.transcripts_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.transcripts_out, 'w', encoding='utf-8') as f:
        for audio_path, text, duration, _ in final_samples:
            f.write(f"{audio_path}|{text}|{duration}\n")
    
    # Print statistics
    logging.info("\n" + "=" * 60)
    logging.info("FILTERING STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total input samples: {stats['total']}")
    logging.info(f"Passed text quality: {stats['passed_text']} ({stats['passed_text']/stats['total']*100:.1f}%)")
    logging.info(f"Passed duration: {stats['passed_duration']}")
    if ref_transcripts:
        logging.info(f"Passed WER check: {stats['passed_wer']}")
    logging.info(f"Passed all criteria: {stats['passed_all']} ({stats['passed_all']/stats['total']*100:.1f}%)")
    logging.info(f"Target retention: {args.target_retention*100:.0f}%")
    logging.info(f"Final output: {len(final_samples)} ({len(final_samples)/stats['total']*100:.1f}%)")
    
    # Print rejection examples
    if stats["rejected"]["text_quality"]:
        logging.info("\n--- Text Quality Rejections ---")
        for ex in stats["rejected"]["text_quality"][:5]:
            logging.info(f"  {ex['audio']}: {ex['reason']} | {ex['text'][:50]}...")
    
    if stats["rejected"]["wer"]:
        logging.info("\n--- WER Rejections ---")
        for ex in stats["rejected"]["wer"][:5]:
            logging.info(f"  {ex['audio']}: WER={ex['wer']:.2%}")
            logging.info(f"    Pseudo: {ex['pseudo']}")
            logging.info(f"    Ref:    {ex['ref']}")
    
    logging.info("\nDone!")


if __name__ == "__main__":
    main()
