import gzip
import json
import re
import sys
import os

def read_group(group):
    read_digit = [" không", " một", " hai", " ba", " bốn", " năm", " sáu", " bảy", " tám", " chín"]
    temp = ""
    if group == "000":
        return ""
    
    # Tram
    temp += read_digit[int(group[0])] + " trăm"
    
    # Chuc
    if group[1] == "0":
        if group[2] == "0":
            return temp
        else:
            temp += " linh" + read_digit[int(group[2])]
            return temp
    elif group[1] == "1":
        temp += " mười"
    else:
        temp += read_digit[int(group[1])] + " mươi"
        
    # Don vi
    if group[2] == "1":
        if group[1] == "0" or group[1] == "1":
            temp += " một"
        else:
            temp += " mốt"
    elif group[2] == "5":
        if group[1] == "0":
            temp += " năm"
        else:
            temp += " lăm"
    elif group[2] == "4":
        if group[1] == "0" or group[1] == "1":
            temp += " bốn"
        else:
            temp += " tư"
    elif group[2] == "0":
        pass
    else:
        temp += read_digit[int(group[2])]
        
    return temp

def num2text_vietnamese(n):
    if n == 0:
        return "không"
    
    s = str(n)
    groups = []
    
    # Pad to multiple of 3
    while len(s) % 3 != 0:
        s = "0" + s
        
    # Split into groups of 3
    for i in range(0, len(s), 3):
        groups.append(s[i:i+3])
        
    # Reverse to process from lowest to highest
    groups = groups[::-1]
    
    result = ""
    suffixes = ["", " nghìn", " triệu", " tỷ", " nghìn tỷ", " triệu tỷ"]
    
    for i in range(len(groups)):
        group_text = read_group(groups[i])
        if group_text:
            # Handle special case for first group (highest value) if it has leading zeros
            if i == len(groups) - 1:
                # Remove " không trăm" or " không trăm linh" if it's the start
                # But our read_group always outputs " không trăm..." for 0xx
                # We need to clean it up later or handle it here.
                pass
            
            result = group_text + suffixes[i] + result
            
    # Clean up result
    result = result.strip()
    
    # Fix initial "không trăm..." if it's at the start of the sentence (the number)
    # Example: 015 -> không trăm mười lăm -> mười lăm
    # But wait, we padded with 0s. 
    # Let's do a simpler post-processing.
    
    # Remove leading "không trăm linh " or "không trăm " if they are at the start
    # and there are following words.
    
    # Actually, a better way is to implement a robust library-like logic.
    # But for now, let's use a simpler recursive approach or just fix the string.
    
    # Let's try a simpler implementation for standard numbers.
    return convert_number(n)

def convert_number(n):
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    
    if n < 10:
        return digits[n]
    
    if n < 20:
        if n == 10: return "mười"
        if n == 15: return "mười lăm"
        if n == 11: return "mười một" # Standard
        return "mười " + digits[n%10]
        
    if n < 100:
        tens = n // 10
        unit = n % 10
        s = digits[tens] + " mươi"
        if unit == 0:
            return s
        if unit == 1:
            return s + " mốt"
        if unit == 4:
            return s + " tư"
        if unit == 5:
            return s + " lăm"
        return s + " " + digits[unit]
        
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        s = digits[hundreds] + " trăm"
        if remainder == 0:
            return s
        if remainder < 10:
            return s + " linh " + digits[remainder]
        return s + " " + convert_number(remainder)
        
    if n < 1000000:
        thousands = n // 1000
        remainder = n % 1000
        s = convert_number(thousands) + " nghìn"
        if remainder == 0:
            return s
        if remainder < 100:
            return s + " không trăm " + convert_number(remainder).replace("linh", "lẻ", 1) # "lẻ" is common for <100 in thousands
        return s + " " + convert_number(remainder)
        
    if n < 1000000000:
        millions = n // 1000000
        remainder = n % 1000000
        s = convert_number(millions) + " triệu"
        if remainder == 0:
            return s
        if remainder < 100000: # e.g. 1 000 050 -> một triệu không trăm năm mươi (nghìn...) - wait, Vietnamese reading is complex here.
            # Simplified: just append remainder with full reading?
            # 1,000,100 -> một triệu một trăm
            # 1,000,010 -> một triệu không trăm mười
            # 1,000,001 -> một triệu không trăm linh một
            # We need to pad the remainder reading?
            
            # Let's use the block approach which is safer.
            pass
            
    # Fallback to a simpler block-based approach for large numbers
    return convert_large_number(n)

def convert_large_number(n):
    if n == 0: return "không"
    
    blocks = []
    while n > 0:
        blocks.append(n % 1000)
        n //= 1000
        
    suffixes = ["", " nghìn", " triệu", " tỷ", " nghìn tỷ"]
    result = []
    
    for i, block in enumerate(blocks):
        if block == 0: continue
        
        block_text = convert_block(block, is_last=(i==0), is_first=(i==len(blocks)-1))
        
        # Add suffix
        if i > 0:
            block_text += suffixes[i]
            
        result.append(block_text)
        
    return " ".join(reversed(result)).strip()

def convert_block(n, is_last=False, is_first=False):
    # n is 0-999
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    
    hundreds = n // 100
    tens = (n % 100) // 10
    units = n % 10
    
    s = ""
    
    # Hundreds
    if hundreds > 0 or (not is_first): # Read hundreds if it's not the very first block (e.g. 1000 -> 1 nghìn, not 1 nghìn 0 trăm)
        # Actually for 1050 -> một nghìn không trăm năm mươi. So if not first block, we always read hundreds.
        s += digits[hundreds] + " trăm "
    
    # Tens and Units
    if tens == 0 and units == 0:
        if not is_first and hundreds == 0: # e.g. 1000 -> block 000. 
             pass # Handled by main loop skipping 0 blocks? No, main loop skips 0 blocks.
             # But what about 1 000 000? Main loop handles it.
             # What about 1 000 050? Block 050.
             pass
        return s.strip()
    
    if tens == 0:
        if hundreds > 0 or not is_first: # 105 -> một trăm linh năm. 1005 -> một nghìn không trăm linh năm.
            s += "linh " + digits[units]
        else: # 5 -> năm (just units)
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

def normalize_text(text):
    # Find all numbers
    # We look for sequences of digits
    # We should also handle floating points if necessary, but let's stick to integers first as they are most common
    # Regex for integer: \d+
    
    def replace_func(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            return convert_large_number(num)
        except:
            return num_str
            
    # Replace standalone numbers or numbers inside text
    # Use word boundaries? Or just replace any digit sequence?
    # "110" -> "một trăm mười"
    # "29.000000" -> This is in the ID, not text. Text usually has "thứ 3", "2023".
    # Let's assume integers for now.
    
    return re.sub(r'\d+', replace_func, text)

def main():
    input_file = "/storage/asr/data/manifest_k2ssl/bac_trung_bo/supervisions_bac_trung_bo.jsonl.gz"
    output_file = "/storage/asr/data/manifest_k2ssl/bac_trung_bo/supervisions_bac_trung_bo_normalized.jsonl.gz"
    
    print(f"Processing {input_file}...")
    
    count = 0
    modified_count = 0
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
        
        for line in f_in:
            count += 1
            try:
                data = json.loads(line)
                original_text = data.get('text', '')
                
                # Normalize
                normalized_text = normalize_text(original_text)
                
                if normalized_text != original_text:
                    modified_count += 1
                    data['text'] = normalized_text
                    
                # Write back
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')
                
            except json.JSONDecodeError:
                print(f"Error decoding line {count}: {line}")
                continue
                
    print(f"Done. Processed {count} lines.")
    print(f"Modified {modified_count} lines.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    # Test cases
    test_nums = [1, 5, 10, 11, 15, 20, 21, 24, 25, 100, 101, 105, 110, 115, 120, 121, 1000, 1001, 1010, 1100, 2023, 1000000, 1000050]
    # for n in test_nums:
    #     print(f"{n}: {convert_large_number(n)}")
        
    main()
