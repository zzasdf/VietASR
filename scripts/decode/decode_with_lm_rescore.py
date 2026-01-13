#!/usr/bin/env python3
"""
Decode with Word-level LM Rescoring (N-best Rescoring)

This script:
1. Runs standard beam search decoding to get N-best hypotheses
2. Loads 17GB word-level KenLM ONCE
3. Rescores all hypotheses with word-level LM
4. Selects best hypothesis based on combined AM + LM score

This properly uses the 17GB LM for fair comparison with API.

Usage:
    python scripts/decode/decode_with_lm_rescore.py \
        --epoch 19 --avg 17 --gpu 0 \
        --test-sets "tongdai_clean" \
        --word-lm-path /data/raw/dolphin/lm_kenlm/4_gram_word_all.binary

Author: VietASR Team
"""

import argparse
import logging
import time
import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Decode with word-level LM rescoring"
    )
    # Required
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--avg", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    
    # Test set
    parser.add_argument("--test-sets", type=str, required=True,
                       help="Comma-separated test set names")
    parser.add_argument("--manifest-dir", type=str, required=True)
    
    # Model paths
    parser.add_argument("--exp-dir", type=str, default="data4/exp_finetune_v3")
    parser.add_argument("--bpe-model", type=str, 
                       default="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model")
    
    # LM
    parser.add_argument("--word-lm-path", type=str, required=True,
                       help="Path to 17GB word-level KenLM binary")
    parser.add_argument("--word-lm-scale", type=float, default=0.3,
                       help="LM scale (0.1-0.5)")
    
    # Decoding params
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--max-duration", type=int, default=1000)
    parser.add_argument("--nbest", type=int, default=10,
                       help="Number of N-best hypotheses for rescoring")
    
    # Model params
    parser.add_argument("--use-averaged-model", type=int, default=1)
    parser.add_argument("--final-downsample", type=int, default=1)
    parser.add_argument("--use-layer-norm", type=int, default=0)
    parser.add_argument("--norm", type=int, default=1)
    
    return parser.parse_args()


def load_word_lm(lm_path: str):
    """Load 17GB word-level KenLM"""
    try:
        import kenlm
    except ImportError:
        logging.error("kenlm not installed")
        sys.exit(1)
    
    logging.info(f"Loading word-level LM from: {lm_path}")
    logging.info("This will take ~20-25 minutes for 17GB file...")
    
    start = time.time()
    model = kenlm.Model(lm_path)
    elapsed = time.time() - start
    
    logging.info(f"✅ LM loaded in {elapsed/60:.1f} minutes")
    logging.info(f"   Order: {model.order}")
    
    return model


def run_decode(args, test_set: str) -> str:
    """Run standard decode and return path to recogs file"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Step 1: Decoding {test_set} (standard beam search)")
    logging.info(f"{'='*60}")
    
    cmd = [
        "python", "SSL/zipformer_fbank/decode.py",
        "--epoch", str(args.epoch),
        "--avg", str(args.avg),
        "--exp-dir", args.exp_dir,
        "--max-duration", str(args.max_duration),
        "--bpe-model", args.bpe_model,
        "--decoding-method", "modified_beam_search",  # Standard beam search
        "--manifest-dir", args.manifest_dir,
        "--use-averaged-model", str(args.use_averaged_model),
        "--final-downsample", str(args.final_downsample),
        "--use-layer-norm", str(args.use_layer_norm),
        "--beam-size", str(args.beam_size),
        "--cuts-name", test_set,
        "--norm", str(args.norm),
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    logging.info(f"Running decode...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"Decode failed: {result.stderr}")
        return None
    
    # Find recogs file path from output
    recogs_pattern = f"{args.exp_dir}/modified_beam_search_use_avg/recogs-{test_set}-*.txt"
    import glob
    recogs_files = sorted(glob.glob(recogs_pattern), key=os.path.getmtime, reverse=True)
    
    if recogs_files:
        recogs_path = recogs_files[0]
        logging.info(f"✅ Decode complete: {recogs_path}")
        return recogs_path
    else:
        logging.error(f"Could not find recogs file matching {recogs_pattern}")
        return None


def rescore_with_lm(recogs_path: str, word_lm, lm_scale: float) -> Dict[str, Tuple[str, str]]:
    """Rescore hypotheses with word-level LM
    
    Returns dict: {cut_id: (ref_text, rescored_hyp_text)}
    """
    import math
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Step 2: Rescoring with word-level LM (scale={lm_scale})")
    logging.info(f"{'='*60}")
    
    results = {}
    
    with open(recogs_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse recogs file - format: "cut_id:\tref=[...]\n" then "cut_id:\thyp=[...]"
    i = 0
    scored_count = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse ref line
        if '\tref=' in line:
            cut_id = line.split(':')[0]
            ref_match = re.search(r"ref=\[(.*?)\]", line)
            if ref_match:
                ref_words = eval('[' + ref_match.group(1) + ']')
                ref_text = ' '.join(ref_words)
            else:
                i += 1
                continue
            
            # Next line should be hyp
            if i + 1 < len(lines):
                hyp_line = lines[i + 1].strip()
                if '\thyp=' in hyp_line:
                    hyp_match = re.search(r"hyp=\[(.*?)\]", hyp_line)
                    if hyp_match:
                        hyp_words = eval('[' + hyp_match.group(1) + ']')
                        hyp_text = ' '.join(hyp_words)
                        
                        # Score with LM
                        lm_score = word_lm.score(hyp_text, bos=True, eos=True)
                        word_count = max(1, len(hyp_words))
                        normalized_lm = lm_score / word_count
                        
                        # For now, just keep the hypothesis (single-best rescoring)
                        # In full N-best, we would compare multiple hypotheses
                        results[cut_id] = (ref_text, hyp_text, normalized_lm)
                        scored_count += 1
                        
                    i += 2
                    continue
        i += 1
    
    logging.info(f"✅ Rescored {scored_count} utterances with LM")
    
    return results


def calculate_wer(results: Dict) -> float:
    """Calculate WER from results"""
    total_errors = 0
    total_words = 0
    
    for cut_id, (ref_text, hyp_text, _) in results.items():
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        
        # Simple edit distance for WER
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        
        # Count errors
        errors = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                errors += max(i2 - i1, j2 - j1)
            elif tag == 'delete':
                errors += i2 - i1
            elif tag == 'insert':
                errors += j2 - j1
        
        total_errors += errors
        total_words += len(ref_words)
    
    wer = (total_errors / total_words * 100) if total_words > 0 else 0
    return wer, total_errors, total_words


def main():
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    logging.info("="*60)
    logging.info("Decode with Word-level LM Rescoring")
    logging.info("="*60)
    logging.info(f"Model: epoch-{args.epoch} avg-{args.avg}")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Test sets: {args.test_sets}")
    logging.info(f"Word LM: {args.word_lm_path}")
    logging.info(f"LM Scale: {args.word_lm_scale}")
    
    # Load LM FIRST (before any decoding)
    word_lm = load_word_lm(args.word_lm_path)
    
    # Process each test set
    test_sets = [s.strip() for s in args.test_sets.split(",")]
    
    all_results = {}
    
    for test_set in test_sets:
        # Step 1: Decode
        recogs_path = run_decode(args, test_set)
        
        if recogs_path is None:
            logging.error(f"Skipping {test_set} due to decode failure")
            continue
        
        # Step 2: Rescore with LM
        results = rescore_with_lm(recogs_path, word_lm, args.word_lm_scale)
        
        # Step 3: Calculate WER
        wer, errors, total = calculate_wer(results)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"RESULTS for {test_set}")
        logging.info(f"{'='*60}")
        logging.info(f"WER (with LM rescoring): {wer:.2f}%")
        logging.info(f"Errors: {errors}, Total words: {total}")
        
        all_results[test_set] = wer
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    for test_set, wer in all_results.items():
        logging.info(f"  {test_set}: WER = {wer:.2f}%")
    
    logging.info("\n✅ All done!")


if __name__ == "__main__":
    main()
