#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import json
import gzip
import sys
import os
import shutil

# Add current directory to path to import normalization scripts
sys.path.append(os.getcwd())
try:
    from normalize_transcripts import normalize_tone_and_typos, normalize_numbers, normalize_punctuation
except ImportError:
    print("Error: Could not import normalization functions. Make sure normalize_transcripts.py is in the current directory.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def normalize_text(text):
    if not text:
        return text
    text = normalize_tone_and_typos(text)
    text = normalize_numbers(text)
    text = normalize_punctuation(text)
    return text

def process_cuts(manifest_dir, cuts_name="test"):
    manifest_dir = Path(manifest_dir)
    cuts_file = manifest_dir / f"vietASR_cuts_{cuts_name}.jsonl.gz"
    
    if not cuts_file.exists():
        logging.warning(f"Cuts file not found: {cuts_file}")
        return

    logging.info(f"Processing {cuts_file}...")
    
    # Backup
    backup_file = manifest_dir / f"vietASR_cuts_{cuts_name}.jsonl.gz.bak_norm"
    if not backup_file.exists():
        shutil.copy(cuts_file, backup_file)
        logging.info(f"Backed up to {backup_file}")

    temp_file = manifest_dir / f"vietASR_cuts_{cuts_name}.jsonl.gz.tmp"
    
    try:
        modified_count = 0
        total_count = 0
        
        with gzip.open(cuts_file, 'rt', encoding='utf-8') as f_in, \
             gzip.open(temp_file, 'wt', encoding='utf-8') as f_out:
            
            for line in f_in:
                total_count += 1
                try:
                    data = json.loads(line)
                    # Handle both CutSet (list of cuts) and individual cuts (one JSON per line)
                    # Lhotse usually stores one cut per line in jsonl.gz
                    
                    if 'supervisions' in data:
                        for sup in data['supervisions']:
                            original_text = sup.get('text', '')
                            if original_text:
                                normalized_text = normalize_text(original_text)
                                if normalized_text != original_text:
                                    sup['text'] = normalized_text
                                    modified_count += 1
                    
                    json.dump(data, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line {total_count}")
                    continue
        
        # Replace original with temp
        shutil.move(temp_file, cuts_file)
        logging.info(f"Normalized {modified_count} cuts out of {total_count} in {cuts_file}")
        
    except Exception as e:
        logging.error(f"Failed to process {cuts_file}: {e}")
        if temp_file.exists():
            os.remove(temp_file)
        # Restore backup if failed
        if backup_file.exists():
            shutil.copy(backup_file, cuts_file)
            logging.info("Restored backup due to error.")

def main():
    parser = argparse.ArgumentParser(description="Normalize text in Lhotse cuts")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing cuts")
    parser.add_argument("--cuts-name", type=str, default="test", help="Name of the cuts file (e.g., test for vietASR_cuts_test.jsonl.gz)")
    
    args = parser.parse_args()
    
    process_cuts(args.data_dir, args.cuts_name)

if __name__ == "__main__":
    main()
