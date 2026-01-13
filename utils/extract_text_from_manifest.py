#!/usr/bin/env python3
"""
Trích xuất text từ Lhotse manifests để train BPE model.
Support cả recordings+supervisions và cuts manifest.
"""

import argparse
from pathlib import Path
from lhotse import load_manifest_lazy, CutSet
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO
)


def extract_from_supervisions(manifest_path: Path, output_file: Path):
    """Trích xuất text từ supervisions manifest."""
    logging.info(f"Loading supervisions from {manifest_path}")
    supervisions = load_manifest_lazy(manifest_path)
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for sup in supervisions:
            if hasattr(sup, 'text') and sup.text:
                f.write(sup.text.strip() + '\n')
                count += 1
                if count % 10000 == 0:
                    logging.info(f"Processed {count} supervisions...")
    
    logging.info(f"✅ Extracted {count} texts to {output_file}")


def extract_from_cuts(manifest_path: Path, output_file: Path):
    """Trích xuất text từ cuts manifest."""
    logging.info(f"Loading cuts from {manifest_path}")
    cuts = load_manifest_lazy(manifest_path)
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for cut in cuts:
            # Cuts có thể có nhiều supervisions
            for sup in cut.supervisions:
                if hasattr(sup, 'text') and sup.text:
                    f.write(sup.text.strip() + '\n')
                    count += 1
                    if count % 10000 == 0:
                        logging.info(f"Processed {count} utterances...")
    
    logging.info(f"✅ Extracted {count} texts to {output_file}")


def merge_text_files(input_files: list, output_file: Path):
    """Gộp nhiều file text thành một."""
    logging.info(f"Merging {len(input_files)} files into {output_file}")
    
    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            if not Path(input_file).exists():
                logging.warning(f"⚠️  File not found: {input_file}")
                continue
                
            logging.info(f"  Reading {input_file}")
            with open(input_file, 'r', encoding='utf-8') as in_f:
                lines = 0
                for line in in_f:
                    out_f.write(line)
                    lines += 1
                total_lines += lines
                logging.info(f"    Added {lines} lines")
    
    logging.info(f"✅ Merged {total_lines} total lines to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from Lhotse manifests for BPE training"
    )
    
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory containing manifests"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["train", "dev"],
        help="Dataset names to extract (e.g., train dev)"
    )
    
    parser.add_argument(
        "--manifest-type",
        type=str,
        choices=["cuts", "supervisions"],
        default="cuts",
        help="Type of manifest: 'cuts' or 'supervisions'"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/text"),
        help="Output directory for text files"
    )
    
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all extracted texts into a single file"
    )
    
    args = parser.parse_args()
    
    # Tạo output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    
    # Trích xuất từng dataset
    for dataset in args.datasets:
        if args.manifest_type == "cuts":
            manifest_file = args.manifest_dir / f"vietASR_cuts_{dataset}.jsonl.gz"
            output_file = args.output_dir / f"text_{dataset}.txt"
            
            if not manifest_file.exists():
                logging.warning(f"⚠️  Manifest not found: {manifest_file}")
                continue
            
            extract_from_cuts(manifest_file, output_file)
            
        else:  # supervisions
            manifest_file = args.manifest_dir / f"supervisions_{dataset}.jsonl.gz"
            output_file = args.output_dir / f"text_{dataset}.txt"
            
            if not manifest_file.exists():
                logging.warning(f"⚠️  Manifest not found: {manifest_file}")
                continue
            
            extract_from_supervisions(manifest_file, output_file)
        
        extracted_files.append(output_file)
    
    # Gộp tất cả nếu cần
    if args.merge and extracted_files:
        merged_file = args.output_dir / "all_text.txt"
        merge_text_files(extracted_files, merged_file)
        logging.info(f"\n📄 Use this file for BPE training: {merged_file}")
    
    logging.info("\n" + "="*70)
    logging.info("✅ DONE!")
    logging.info("="*70)


if __name__ == "__main__":
    main()