#!/usr/bin/env python
"""combine_manifests.py

Utility to combine Lhotse manifests from raw datasets (already processed with convert_data.py)
and existing manifests under `/storage/asr/hiennt/VietASR/manifests/`.
It also applies the hybrid Vietnamese tone normalization (open‑syllable = old style, closed‑syllable = new style)
to any existing supervisions, deduplicates recording IDs, and creates train/dev splits
(99% train, 1% dev). No test split is generated as per user request.

Usage example::

    python combine_manifests.py \
        --raw-manifest-dir /data/manifest \
        --existing-manifest-dir /storage/asr/hiennt/VietASR/manifests \
        --output-dir /data/manifest \
        --train-ratio 0.99
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Set

from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse import load_manifest

# Import the hybrid normalizer we created earlier
import sys
sys.path.append(str(Path(__file__).parent / "utils"))
from normalize_hybrid import normalize_hybrid_tone


def load_manifests(manifest_dir: Path) -> Tuple[RecordingSet, SupervisionSet]:
    """Load all recordings_*.jsonl.gz and supervisions_*.jsonl.gz in a directory.
    Returns a concatenated RecordingSet and SupervisionSet.
    """
    recordings: List[RecordingSet] = []
    supervisions: List[SupervisionSet] = []
    for rec_file in manifest_dir.glob("recordings_*.jsonl.gz"):
        recordings.append(load_manifest(rec_file))
    for sup_file in manifest_dir.glob("supervisions_*.jsonl.gz"):
        supervisions.append(load_manifest(sup_file))
    if recordings:
        rec_set = RecordingSet.from_recordings([r for rs in recordings for r in rs])
    else:
        rec_set = RecordingSet.from_recordings([])
    if supervisions:
        sup_set = SupervisionSet.from_segments([s for ss in supervisions for s in ss])
    else:
        sup_set = SupervisionSet.from_segments([])
    return rec_set, sup_set


def normalize_existing_supervisions(sup_set: SupervisionSet) -> SupervisionSet:
    """Apply hybrid tone normalization to texts of an existing SupervisionSet.
    The text is first lower‑cased then passed through ``normalize_hybrid_tone``.
    """
    normalized = []
    for seg in sup_set:
        new_text = normalize_hybrid_tone(seg.text.lower())
        normalized.append(seg._replace(text=new_text))
    return SupervisionSet.from_segments(normalized)


def deduplicate_ids(
    rec_set: RecordingSet, sup_set: SupervisionSet
) -> Tuple[RecordingSet, SupervisionSet]:
    """Remove duplicate recording IDs across the combined sets.
    If a duplicate is found, the first occurrence is kept.
    """
    seen: Set[str] = set()
    uniq_recordings = []
    for rec in rec_set:
        if rec.id not in seen:
            seen.add(rec.id)
            uniq_recordings.append(rec)
    uniq_supervisions = []
    for sup in sup_set:
        if sup.recording_id in seen:
            uniq_supervisions.append(sup)
    return RecordingSet.from_recordings(uniq_recordings), SupervisionSet.from_segments(uniq_supervisions)


def split_supervisions(
    sup_set: SupervisionSet, train_ratio: float
) -> Tuple[SupervisionSet, SupervisionSet]:
    """Randomly split supervisions into train/dev according to ``train_ratio``.
    The split is performed on the supervision level; recordings are kept as‑is.
    """
    sup_list = list(sup_set)
    random.shuffle(sup_list)
    split_idx = int(len(sup_list) * train_ratio)
    train_sup = SupervisionSet.from_segments(sup_list[:split_idx])
    dev_sup = SupervisionSet.from_segments(sup_list[split_idx:])
    return train_sup, dev_sup


def main():
    parser = argparse.ArgumentParser(description="Combine Lhotse manifests with hybrid normalization")
    parser.add_argument("--raw-manifest-dir", type=Path, required=True, help="Directory containing manifests created by convert_data.py")
    parser.add_argument("--existing-manifest-dir", type=Path, required=True, help="Directory with pre‑existing manifests (e.g., /storage/asr/hiennt/VietASR/manifests)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where combined manifests will be written")
    parser.add_argument("--train-ratio", type=float, default=0.99, help="Proportion of data for training (dev gets the rest)")
    args = parser.parse_args()

    # Load raw manifests (already normalized)
    raw_rec, raw_sup = load_manifests(args.raw_manifest_dir)

    # Load existing manifests and apply hybrid normalization
    exist_rec, exist_sup = load_manifests(args.existing_manifest_dir)
    exist_sup = normalize_existing_supervisions(exist_sup)

    # Combine
    combined_rec = RecordingSet.from_recordings(list(raw_rec) + list(exist_rec))
    combined_sup = SupervisionSet.from_segments(list(raw_sup) + list(exist_sup))

    # Deduplicate IDs
    combined_rec, combined_sup = deduplicate_ids(combined_rec, combined_sup)

    # Split into train / dev
    train_sup, dev_sup = split_supervisions(combined_sup, args.train_ratio)
    # Recordings are shared; we keep the same recording set for both splits
    # Save manifests
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Train
    combined_rec.to_file(args.output_dir / "recordings_combined_train.jsonl.gz")
    train_sup.to_file(args.output_dir / "supervisions_combined_train.jsonl.gz")
    # Dev
    combined_rec.to_file(args.output_dir / "recordings_combined_dev.jsonl.gz")
    dev_sup.to_file(args.output_dir / "supervisions_combined_dev.jsonl.gz")

    print("✅ Combined manifests written to", args.output_dir)
    print("   Train supervisions:", len(train_sup))
    print("   Dev supervisions:", len(dev_sup))

if __name__ == "__main__":
    main()
