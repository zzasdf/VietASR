#!/usr/bin/env python3
# Copyright 2024 (Author: Antigravity)
"""
Pseudo-labeling script for VietASR.
Replaces original labels with model hypotheses for cuts with WER in a specified range.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import jiwer
from lhotse import load_manifest_lazy, CutSet
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Replace labels with model hypotheses for cuts with WER in specified range."
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        required=True,
        help="Path to the input CutSet manifest.",
    )
    parser.add_argument(
        "--decode-results",
        type=Path,
        required=True,
        help="Path to decode results file (format: cut_id hypothesis_text).",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        required=True,
        help="Path to save the output manifest with updated labels.",
    )
    parser.add_argument(
        "--wer-low",
        type=float,
        default=0.1,
        help="Lower WER threshold. Cuts with WER below this are kept as-is. Default: 0.1",
    )
    parser.add_argument(
        "--wer-high",
        type=float,
        default=0.5,
        help="Upper WER threshold. Cuts with WER above this are removed. Default: 0.5",
    )
    parser.add_argument(
        "--remove-high-wer",
        action="store_true",
        help="If True, remove cuts with WER > wer-high instead of keeping them.",
    )
    parser.add_argument(
        "--inspect-num",
        type=int,
        default=0,
        help="Number of relabeled examples to print for inspection. Default: 0",
    )
    return parser.parse_args()


def load_hypotheses(path: Path) -> Dict[str, str]:
    """Load decode results into a dictionary."""
    hyps = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                hyps[parts[0]] = parts[1]
            elif len(parts) == 1:
                hyps[parts[0]] = ""
    return hyps


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info(f"Loading hypotheses from {args.decode_results}")
    hyps = load_hypotheses(args.decode_results)
    logging.info(f"Loaded {len(hyps)} hypotheses.")

    logging.info(f"Loading manifest from {args.manifest_in}")
    cuts = load_manifest_lazy(args.manifest_in)

    updated_cuts = []
    stats = {
        "total": 0,
        "kept_original": 0,  # WER < wer_low
        "relabeled": 0,      # wer_low <= WER <= wer_high
        "removed": 0,        # WER > wer_high (if remove_high_wer)
        "kept_high_wer": 0,  # WER > wer_high (if not remove_high_wer)
        "not_found": 0,
    }

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

    relabeled_examples = []

    logging.info("Processing cuts...")
    for cut in tqdm(cuts):
        stats["total"] += 1

        if cut.id not in hyps:
            stats["not_found"] += 1
            updated_cuts.append(cut)
            continue

        if not cut.supervisions:
            updated_cuts.append(cut)
            continue

        ref = cut.supervisions[0].text or ""
        hyp = hyps[cut.id]

        if not ref:
            # No reference text, keep as-is
            updated_cuts.append(cut)
            continue

        wer = jiwer.wer(
            ref,
            hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        )

        if wer < args.wer_low:
            # Good quality label, keep original
            stats["kept_original"] += 1
            updated_cuts.append(cut)
        elif wer <= args.wer_high:
            # Medium quality, relabel with hypothesis
            old_text = cut.supervisions[0].text
            cut.supervisions[0].text = hyp
            stats["relabeled"] += 1
            
            if len(relabeled_examples) < args.inspect_num:
                relabeled_examples.append({
                    "id": cut.id,
                    "wer": wer,
                    "old": old_text,
                    "new": hyp,
                })
            
            updated_cuts.append(cut)
        else:
            # High WER
            if args.remove_high_wer:
                stats["removed"] += 1
            else:
                stats["kept_high_wer"] += 1
                updated_cuts.append(cut)

    # Print statistics
    logging.info("=" * 50)
    logging.info("PSEUDO-LABELING STATISTICS")
    logging.info("=" * 50)
    logging.info(f"Total cuts processed: {stats['total']}")
    logging.info(f"Kept original (WER < {args.wer_low}): {stats['kept_original']}")
    logging.info(f"Relabeled ({args.wer_low} <= WER <= {args.wer_high}): {stats['relabeled']}")
    if args.remove_high_wer:
        logging.info(f"Removed (WER > {args.wer_high}): {stats['removed']}")
    else:
        logging.info(f"Kept high WER (WER > {args.wer_high}): {stats['kept_high_wer']}")
    logging.info(f"Not found in decode results: {stats['not_found']}")
    logging.info(f"Final output cuts: {len(updated_cuts)}")

    # Print relabeled examples
    if relabeled_examples:
        logging.info("\n" + "=" * 50)
        logging.info("RELABELED EXAMPLES")
        logging.info("=" * 50)
        for ex in relabeled_examples:
            logging.info(f"\n[RELABELED] Cut ID: {ex['id']} (WER: {ex['wer']:.2f})")
            logging.info(f"  Old: {ex['old']}")
            logging.info(f"  New: {ex['new']}")

    # Save output
    logging.info(f"\nSaving to {args.manifest_out}")
    cutset = CutSet.from_cuts(updated_cuts)
    cutset.to_file(args.manifest_out)
    logging.info("Done!")


if __name__ == "__main__":
    main()
