import sys
import os
import re

# Comprehensive Dictionary map for Vietnamese tone normalization
# Normalizing to "New Style" (tone on second vowel for oa, oe, uy)
# òa -> oà, óa -> oá, ỏa -> oả, õa -> oã, ọa -> oạ
# òe -> oè, óe -> oé, ỏe -> oẻ, õe -> oẽ, ọe -> oẹ
# ùy -> uỳ, úy -> uý, ủy -> uỷ, ũy -> uỹ, ụy -> uỵ

DICT_MAP = {}

# Define the mappings
# Source (Old Style/Typo) -> Target (New Style)
# format: (char_with_tone_on_1st, char_with_tone_on_2nd)
mappings = [
    # oa
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    # oe
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    # uy
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
]

# Word-level tone position mappings (whole words, not substrings)
WORD_TONE_MAPPINGS = {
    # Reverse mappings (new style -> old style for WER matching)
    "khỏe": "khoẻ", "thuỷ": "thủy", "tuỵ": "tụy", "uỷ": "ủy",
    "mộta": "một a",
    # uy variants
    "quý": "qúy", "thuý": "thúy", "suý": "súy", "luý": "lúy", 
    "tuý": "túy", "huý": "húy", "nguý": "ngúy", "thuỵ": "thụy",
    # oe variants
    "khoẻ": "khỏe", "loẻ": "lỏe", "ngoẻ": "ngỏe", "hoẻ": "hỏe", "toẻ": "tỏe",
    # oa variants  
    "hoạ": "họa", "toà": "tòa",
}

# Generate uppercase and 0 variants
for src, tgt in mappings:
    # Lowercase
    DICT_MAP[src] = tgt
    
    # Uppercase first letter (Sentence case)
    # src: òa -> Òa, tgt: oà -> Oà
    src_cap = src.capitalize()
    tgt_cap = tgt.capitalize()
    DICT_MAP[src_cap] = tgt_cap
    
    # Uppercase all (ALL CAPS)
    # src: òa -> ÒA, tgt: oà -> OÀ
    src_upper = src.upper()
    tgt_upper = tgt.upper()
    DICT_MAP[src_upper] = tgt_upper
    
    # '0' variants (Typos from OCR or input)
    # 0à -> Oà, 0À -> OÀ (mapping 0-typos to New Style)
    # We map 0-typos to the TARGET (New Style) directly
    # e.g. 0à -> Oà (New Style)
    
    # We need to handle 0-typos for both Old and New style inputs if they exist
    # But primarily we want to fix "0à" -> "Oà"
    
    if src.lower().startswith('ò') or src.lower().startswith('ó') or src.lower().startswith('ỏ') or src.lower().startswith('õ') or src.lower().startswith('ọ'):
         # src starts with vowel with tone, e.g. òa. '0' replacement is tricky here because tone is on the char.
         # Actually, 0-typo usually replaces 'o' or 'O'.
         # If src is "òa", 0-typo is "0a" (no tone on 0) + tone? No.
         # Usually it's like "0à" where 0 replaces o.
         # But "òa" starts with "ò". "0" cannot hold a tone in standard input.
         # So "0à" is likely "0" + "à".
         pass
         
    # Let's handle the explicit 0-typos based on the target (New Style) which starts with 'o'
    # tgt is like "oà". 0-typo would be "0à".
    if tgt.lower().startswith('o'):
        # 0à -> Oà
        src_0_lower = '0' + tgt[1:]
        DICT_MAP[src_0_lower] = tgt_cap
        
        # 0À -> OÀ
        src_0_upper = '0' + tgt[1:].upper()
        DICT_MAP[src_0_upper] = tgt_upper

# Add specific '0' cases if missed or for 'u' if needed (though '0' usually looks like 'O')
# The user specifically listed 0à, 0á... so we keep the logic generic or hardcode if needed.
# The loop above covers Oà, Oá... and 0à, 0á... for 'oa', 'oe'.

# Manual additions for any other edge cases if needed
# (The loop covers the user's provided list and expands it)


def normalize_punctuation(text):
    """
    Normalize punctuation:
    1. Replace common punctuation and special characters with space.
    2. Collapse multiple spaces into one.
    3. Trim leading/trailing spaces.
    """
    # Standard punctuation from string.punctuation
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    import string
    punctuations = string.punctuation
    
    # Extended list of characters to remove/replace with space
    extended_punctuations = [
        "”", "“", "…", "–", "—", "’", "‘", "´", "`", 
        "«", "»", "•", "–", "…", "−"
    ]
    
    # Replace all with space
    for char in punctuations:
        text = text.replace(char, " ")
    
    for char in extended_punctuations:
        text = text.replace(char, " ")
        
    # Normalize whitespace (collapse multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_tone_and_typos(text):
    # Apply substring mappings (oa, oe, uy)
    for k, v in DICT_MAP.items():
        text = text.replace(k, v)
    
    # Apply word-level mappings (exact word match only)
    words = text.split()
    result = []
    for word in words:
        word_lower = word.lower()
        if word_lower in WORD_TONE_MAPPINGS:
            # Preserve original case pattern
            if word.isupper():
                result.append(WORD_TONE_MAPPINGS[word_lower].upper())
            elif word[0].isupper():
                result.append(WORD_TONE_MAPPINGS[word_lower].capitalize())
            else:
                result.append(WORD_TONE_MAPPINGS[word_lower])
        else:
            result.append(word)
    return " ".join(result)

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
        print("Usage: python normalize_transcripts.py <input_file> [output_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".txt", "_normalized.txt")
    
    print(f"Processing {input_file} -> {output_file}")
    
    total_count = 0
    modified_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_count += 1
            line = line.strip()
            if not line:
                continue
            
            # Check for pipe separator
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    # Assume format: ID|TEXT|DURATION or ID|TEXT
                    # We only normalize the TEXT part (index 1)
                    utt_id = parts[0]
                    text = parts[1]
                    original_text = text
                    rest = parts[2:] # Duration etc.
                    
                    # 1. Normalize tone and typos
                    text = normalize_tone_and_typos(text)
                    
                    # 2. Normalize numbers
                    text = normalize_numbers(text)
                    
                    # 3. Normalize punctuation
                    text = normalize_punctuation(text)
                    
                    if text != original_text:
                        modified_count += 1
                    
                    # Reconstruct line
                    new_line = f"{utt_id}|{text}"
                    if rest:
                        new_line += "|" + "|".join(rest)
                    
                    f_out.write(f"{new_line}\n")
                else:
                    # Malformed pipe line? Just write it back or try to normalize
                    f_out.write(f"{line}\n")
            else:
                # Assume standard Kaldi format: ID TEXT
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    original_text = text
                    
                    # 1. Normalize tone and typos
                    text = normalize_tone_and_typos(text)
                    
                    # 2. Normalize numbers
                    text = normalize_numbers(text)
                    
                    # 3. Normalize punctuation (remove special chars, fix spaces)
                    text = normalize_punctuation(text)
                    
                    if text != original_text:
                        modified_count += 1
                    
                    # Only write if text is not empty after normalization
                    if text:
                        f_out.write(f"{utt_id} {text}\n")
                else:
                    # Handle lines without ID (just text?) or malformed
                    # If it's just text, normalize it too
                    text = line
                    original_text = text
                    
                    text = normalize_tone_and_typos(text)
                    text = normalize_numbers(text)
                    text = normalize_punctuation(text)
                    
                    if text != original_text:
                        modified_count += 1
                        
                    if text:
                        f_out.write(f"{text}\n")

    print(f"Done. Processed {total_count} lines.")
    print(f"Modified {modified_count} lines.")

if __name__ == "__main__":
    main()
