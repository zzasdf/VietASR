#!/usr/bin/env python3
"""
Tạo dev set đại diện và loại bỏ dev cuts khỏi train set.
Đảm bảo không có data leakage giữa train và dev.
"""

import argparse
import logging
from pathlib import Path
from lhotse import load_manifest_lazy, CutSet
import random

logging.basicConfig(level=logging.INFO)

def create_dev_and_clean_train(
    fbank_dir: Path, 
    train_input: Path,
    dev_output: Path, 
    train_output: Path,
    samples_per_dataset: int = 100, 
    seed: int = 42
):
    """
    1. Tạo dev set bằng cách lấy mẫu từ mỗi dataset
    2. Loại bỏ dev cuts khỏi train set
    """
    random.seed(seed)
    
    datasets = [
        "regions_63",
        "regions.bac_trung_bo", 
        "regions.bac_trung_bo_112024",
        "regions.dong_nam_bo",
        "tongdai",
        "tongdai_02112022",
        "tongdai_25112024"
    ]
    
    all_dev_cuts = []
    dev_cut_ids = set()
    
    # Step 1: Sample dev cuts from each dataset
    logging.info("=== Step 1: Sampling dev cuts ===")
    for dataset in datasets:
        manifest_path = fbank_dir / dataset / f"vietASR_cuts_{dataset}.jsonl.gz"
        
        if not manifest_path.exists():
            logging.warning(f"Manifest not found: {manifest_path}")
            continue
        
        logging.info(f"Sampling from {dataset}...")
        cuts = load_manifest_lazy(manifest_path)
        
        # Collect all cuts
        all_cuts = list(cuts)
        total = len(all_cuts)
        
        # Sample
        n_samples = min(samples_per_dataset, total)
        sampled = random.sample(all_cuts, n_samples)
        
        logging.info(f"  {dataset}: {n_samples}/{total} cuts")
        all_dev_cuts.extend(sampled)
        
        # Track dev cut IDs
        for cut in sampled:
            dev_cut_ids.add(cut.id)
    
    # Shuffle and save dev set
    random.shuffle(all_dev_cuts)
    dev_set = CutSet.from_cuts(all_dev_cuts)
    dev_set.to_file(dev_output)
    logging.info(f"Created dev set with {len(all_dev_cuts)} cuts -> {dev_output}")
    
    # Step 2: Remove dev cuts from train set
    logging.info("=== Step 2: Removing dev cuts from train ===")
    train_cuts = load_manifest_lazy(train_input)
    
    clean_train_cuts = []
    removed_count = 0
    total_train = 0
    
    for cut in train_cuts:
        total_train += 1
        if cut.id in dev_cut_ids:
            removed_count += 1
        else:
            clean_train_cuts.append(cut)
    
    clean_train_set = CutSet.from_cuts(clean_train_cuts)
    clean_train_set.to_file(train_output)
    
    logging.info(f"Original train: {total_train} cuts")
    logging.info(f"Removed {removed_count} dev cuts from train")
    logging.info(f"Clean train: {len(clean_train_cuts)} cuts -> {train_output}")
    
    return len(all_dev_cuts), len(clean_train_cuts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbank-dir", type=Path, default=Path("fbank"))
    parser.add_argument("--train-input", type=Path, default=Path("fbank/combined_train_filtered.jsonl.gz"),
                        help="Input train manifest")
    parser.add_argument("--dev-output", type=Path, default=Path("fbank/vietASR_cuts_dev.jsonl.gz"))
    parser.add_argument("--train-output", type=Path, default=Path("fbank/vietASR_cuts_train.jsonl.gz"),
                        help="Output clean train manifest (without dev cuts)")
    parser.add_argument("--samples-per-dataset", type=int, default=100,
                        help="Number of samples from each dataset for dev")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    create_dev_and_clean_train(
        args.fbank_dir, 
        args.train_input,
        args.dev_output, 
        args.train_output,
        args.samples_per_dataset, 
        args.seed
    )

if __name__ == "__main__":
    main()
