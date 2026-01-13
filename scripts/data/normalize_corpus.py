import sys
import os
import re

# Normalization logic (Same as normalize_transcripts.py)
DICT_MAP = {}
mappings = [
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
]
for src, tgt in mappings:
    DICT_MAP[src] = tgt
    DICT_MAP[src.capitalize()] = tgt.capitalize()
    DICT_MAP[src.upper()] = tgt.upper()
    if tgt.lower().startswith('o'):
        src_0_lower = '0' + tgt[1:]
        DICT_MAP[src_0_lower] = tgt.capitalize()
        src_0_upper = '0' + tgt[1:].upper()
        DICT_MAP[src_0_upper] = tgt.upper()

def normalize_tone_and_typos(text):
    for k, v in DICT_MAP.items():
        text = text.replace(k, v)
    # Specific fixes
    text = text.replace("a lô", "alo")
    text = text.replace("A lô", "Alo")
    text = text.replace("A Lô", "Alo")
    return text

def normalize_punctuation(text):
    import string
    punctuations = string.punctuation
    extended_punctuations = [
        "”", "“", "…", "–", "—", "’", "‘", "´", "`", 
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python normalize_corpus.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".txt", "_normalized.txt")
    
    print(f"Processing {input_file} -> {output_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            # Normalize
            text = normalize_tone_and_typos(line)
            text = normalize_numbers(text)
            text = normalize_punctuation(text)
            
            if text:
                f_out.write(f"{text}\n")
            
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} lines...", end='\r')
                
    print(f"\nDone. Processed {count} lines.")

if __name__ == "__main__":
    main()
