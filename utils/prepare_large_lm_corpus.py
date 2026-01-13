#!/usr/bin/env python3
"""
Extract and normalize text from /data/raw/train for LM training
Run this INSIDE Docker container
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

# ==========================================
# NORMALIZATION LOGIC (from normalize_transcripts.py)
# ==========================================

DICT_MAP = {}
mappings = [
    ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
    ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
    ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
]

for src, tgt in mappings:
    DICT_MAP[src] = tgt
    DICT_MAP[src.capitalize()] = tgt.capitalize()
    DICT_MAP[src.upper()] = tgt.upper()
    if src.lower().startswith('o'):
        DICT_MAP['0' + src[1:]] = tgt.capitalize()
        DICT_MAP['0' + src[1:].upper()] = tgt.upper()

def normalize_tone_and_typos(text):
    for k, v in DICT_MAP.items():
        text = text.replace(k, v)
    return text

def normalize_punctuation(text):
    import string
    punctuations = string.punctuation
    extended_punctuations = [
        """, """, "…", "–", "—", "'", "'", "´", "`", 
        "«", "»", "•", "–", "…", "−"
    ]
    for char in punctuations:
        text = text.replace(char, " ")
    for char in extended_punctuations:
        text = text.replace(char, " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def convert_block(n, is_last=False, is_first=False):
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    hundreds = n // 100
    tens = (n % 100) // 10
    units = n % 10
    s = ""
    if hundreds > 0 or (not is_first):
        s += digits[hundreds] + " trăm "
    if tens == 0 and units == 0:
        return s.strip()
    if tens == 0:
        if hundreds > 0 or not is_first:
            s += "linh " + digits[units]
        else:
            s += digits[units]
    elif tens == 1:
        s += "mười"
        if units == 1: s += " một"
        elif units == 5: s += " lăm"
        elif units > 0: s += " " + digits[units]
    else:
        s += digits[tens] + " mươi"
        if units == 1: s += " mốt"
        elif units == 4: s += " tư"
        elif units == 5: s += " lăm"
        elif units > 0: s += " " + digits[units]
    return s.strip()

def convert_large_number(n):
    if n == 0: return "không"
    blocks = []
    temp_n = n
    while temp_n > 0:
        blocks.append(temp_n % 1000)
        temp_n //= 1000
    suffixes = ["", " nghìn", " triệu", " tỷ", " nghìn tỷ"]
    result = []
    for i, block in enumerate(blocks):
        if block == 0: continue
        block_text = convert_block(block, is_last=(i==0), is_first=(i==len(blocks)-1))
        if i > 0:
            block_text += suffixes[i]
        result.append(block_text)
    return " ".join(reversed(result)).strip()

def normalize_numbers(text):
    def replace_func(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            return convert_large_number(num)
        except:
            return num_str
    return re.sub(r'\d+', replace_func, text)

def normalize_line(text):
    text = normalize_tone_and_typos(text)
    text = normalize_numbers(text)
    text = normalize_punctuation(text)
    return text.lower()

# ==========================================
# MAIN PROCESSING
# ==========================================

def main():
    raw_dir = Path("/data/raw/train")
    output_file = Path("data4/lm_corpus/corpus_large.txt")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {raw_dir} for transcripts.txt...")
    
    # Find all transcripts.txt recursively
    transcript_files = []
    for root, dirs, files in os.walk(raw_dir):
        if "transcripts.txt" in files:
            transcript_files.append(Path(root) / "transcripts.txt")
        if "prompts.txt" in files:  # VIVOS format
            transcript_files.append(Path(root) / "prompts.txt")
    
    print(f"Found {len(transcript_files)} transcript files.")
    
    total_lines = 0
    processed_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for tf in tqdm(transcript_files, desc="Processing files"):
            try:
                with open(tf, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line: continue
                        
                        # Handle ID|TEXT or ID TEXT format
                        text = ""
                        if "|" in line:
                            parts = line.split("|")
                            if len(parts) >= 2:
                                text = parts[1]
                        else:
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2:
                                text = parts[1]
                            else:
                                text = line
                                
                        if text:
                            norm_text = normalize_line(text)
                            if norm_text:
                                f_out.write(norm_text + "\n")
                                processed_lines += 1
                        total_lines += 1
            except Exception as e:
                print(f"Error reading {tf}: {e}")
                
    print(f"Done.")
    print(f"Total lines scanned: {total_lines}")
    print(f"Lines written to corpus: {processed_lines}")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
