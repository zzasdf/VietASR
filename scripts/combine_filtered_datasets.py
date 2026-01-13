#!/usr/bin/env python3
# Script: combine_filtered_datasets.py
# Gộp các tập đã lọc + Oversample vùng miền

import argparse
from pathlib import Path
from lhotse import load_manifest_lazy, CutSet
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbank-dir", type=Path, default=Path("fbank"))
    parser.add_argument("--output", type=Path, default=Path("fbank/combined_train_filtered.jsonl.gz"))
    parser.add_argument("--oversample-regions", type=int, default=2, 
                        help="Số lần nhân dữ liệu vùng miền")
    args = parser.parse_args()

    # Định nghĩa các tập và loại
    datasets = {
        # Tập tổng đài (không oversample)
        "tongdai": {"type": "call_center", "oversample": 1},
        "tongdai_02112022": {"type": "call_center", "oversample": 1},
        "tongdai_25112024": {"type": "call_center", "oversample": 1},
        # Tập vùng miền (oversample)
        "regions_63": {"type": "regional", "oversample": args.oversample_regions},
        "regions.bac_trung_bo": {"type": "regional", "oversample": args.oversample_regions},
        "regions.bac_trung_bo_112024": {"type": "regional", "oversample": args.oversample_regions},
        "regions.dong_nam_bo": {"type": "regional", "oversample": args.oversample_regions},
    }

    all_cuts = []
    stats = {}

    for dataset_name, config in datasets.items():
        manifest_path = args.fbank_dir / dataset_name / f"vietASR_cuts_{dataset_name}_filtered.jsonl.gz"
        
        if not manifest_path.exists():
            logging.warning(f"[SKIP] {dataset_name}: File not found - {manifest_path}")
            continue
        
        logging.info(f"Loading: {dataset_name}")
        cuts = load_manifest_lazy(manifest_path)
        
        # Đếm số mẫu gốc
        cut_list = list(cuts)
        original_count = len(cut_list)
        
        # Oversample nếu cần
        oversample = config["oversample"]
        if oversample > 1:
            logging.info(f"  Oversampling {dataset_name} x{oversample}")
            for _ in range(oversample):
                all_cuts.extend(cut_list)
            final_count = original_count * oversample
        else:
            all_cuts.extend(cut_list)
            final_count = original_count
        
        stats[dataset_name] = {
            "type": config["type"],
            "original": original_count,
            "final": final_count,
            "oversample": oversample
        }
        logging.info(f"  {dataset_name}: {original_count} -> {final_count} cuts")

    # Tạo CutSet và lưu
    logging.info(f"\nTotal cuts: {len(all_cuts)}")
    combined = CutSet.from_cuts(all_cuts)
    
    logging.info(f"Saving to: {args.output}")
    combined.to_file(args.output)

    # In thống kê
    print("\n" + "=" * 60)
    print("THỐNG KÊ CUỐI CÙNG")
    print("=" * 60)
    
    total_call_center = sum(s["final"] for s in stats.values() if s["type"] == "call_center")
    total_regional = sum(s["final"] for s in stats.values() if s["type"] == "regional")
    
    print(f"\nCall Center: {total_call_center:,} cuts")
    for name, s in stats.items():
        if s["type"] == "call_center":
            print(f"  - {name}: {s['original']:,} cuts")
    
    print(f"\nVùng Miền (x{args.oversample_regions}): {total_regional:,} cuts")
    for name, s in stats.items():
        if s["type"] == "regional":
            print(f"  - {name}: {s['original']:,} x{s['oversample']} = {s['final']:,} cuts")
    
    print(f"\nTỔNG CỘNG: {len(all_cuts):,} cuts")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
