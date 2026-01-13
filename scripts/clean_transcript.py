#!/usr/bin/env python3
"""
Script lọc và clean transcript:
1. Xóa sample chỉ có "111" (hoặc rỗng sau khi clean)
2. Xóa đuôi "111", "ck" ở cuối text

Usage:
    python scripts/clean_transcript.py \
        --input /path/to/transcripts.txt \
        --output /path/to/transcripts_clean.txt
"""

import argparse
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def clean_text(text: str) -> str:
    """Xóa đuôi 111, ck ở cuối text"""
    text = text.strip()
    
    # Lặp lại cho đến khi không còn đuôi cần xóa
    while True:
        old_text = text
        # Xóa đuôi 111
        text = re.sub(r'\s+111\s*$', '', text)
        # Xóa đuôi ck
        text = re.sub(r'\s+ck\s*$', '', text, flags=re.IGNORECASE)
        
        if text == old_text:
            break
    
    return text.strip()


def should_remove(text: str) -> bool:
    """Kiểm tra xem sample có cần xóa không"""
    cleaned = text.strip()
    
    # Xóa nếu chỉ có 111 (hoặc các biến thể)
    if re.match(r'^111\s*$', cleaned):
        return True
    
    # Xóa nếu text rỗng
    if not cleaned:
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Clean transcript: remove 111-only samples, trim trailing 111/ck")
    parser.add_argument("--input", required=True, help="Input transcript file (path|text|duration)")
    parser.add_argument("--output", required=True, help="Output cleaned transcript file")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    removed = 0
    modified = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            parts = line.split('|')
            
            if len(parts) < 3:
                continue
            
            audio_path = parts[0]
            text = parts[1]
            duration = parts[2]
            rest = parts[3:] if len(parts) > 3 else []
            
            # Kiểm tra xem cần xóa sample không
            if should_remove(text):
                removed += 1
                continue
            
            # Clean text (xóa đuôi 111, ck)
            cleaned_text = clean_text(text)
            
            # Kiểm tra sau khi clean có rỗng không
            if not cleaned_text:
                removed += 1
                continue
            
            if cleaned_text != text.strip():
                modified += 1
            
            # Ghi ra file
            output_line = f"{audio_path}|{cleaned_text}|{duration}"
            if rest:
                output_line += "|" + "|".join(rest)
            fout.write(output_line + "\n")
    
    kept = total - removed
    
    logging.info("=" * 50)
    logging.info("CLEAN TRANSCRIPT COMPLETE")
    logging.info("=" * 50)
    logging.info(f"Total input:      {total:,}")
    logging.info(f"Removed (111):    {removed:,} ({removed/total*100:.1f}%)")
    logging.info(f"Modified (trim):  {modified:,}")
    logging.info(f"Kept:             {kept:,} ({kept/total*100:.1f}%)")
    logging.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
