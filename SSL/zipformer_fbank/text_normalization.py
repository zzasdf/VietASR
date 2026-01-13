"""
Vietnamese Text Normalization Mappings for ASR WER Calculation
Reads mappings from vi2en.txt to match other models' scoring method.

This module:
1. Loads mappings from vi2en.txt at import time
2. Separates FILLER_WORDS (empty target) from WORD_MAPPINGS
3. Provides normalize_text() and remove_filler_words() functions
"""

from typing import List, Dict, Set
from pathlib import Path
import os
import unicodedata

# Find vi2en.txt file
def _find_vi2en_file():
    """Find vi2en.txt relative to this file or in common locations"""
    possible_paths = [
        Path(__file__).parent.parent.parent / "vi2en.txt",  # /VietASR/vi2en.txt
        Path(__file__).parent.parent.parent.parent / "vi2en.txt",
        Path("/vietasr/vi2en.txt"),
        Path("vi2en.txt"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def _load_mappings_from_file(filepath: Path):
    """
    Load mappings from vi2en.txt
    Format: source|target (target can be empty for filler words)
    """
    filler_words = set()
    phrase_mappings = {}  # Multi-word source
    word_mappings = {}    # Single-word source
    
    if not filepath or not filepath.exists():
        print(f"Warning: vi2en.txt not found, using default mappings")
        return _get_default_mappings()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            parts = line.split('|', 1)
            if len(parts) != 2:
                continue
                
            source = parts[0].strip().lower()
            target = parts[1].strip().lower()
            
            if not source:
                continue
            
            # Skip dangerous patterns that match too broadly
            if source in ['g ', 'r ', 's n', 'd i ', 'e n', 'theo d', 'x n', 'en nờ', 'e nờ']:
                continue
            
            # Empty target = filler word to delete
            if not target:
                # Only add single words as fillers
                if ' ' not in source and len(source) <= 3:
                    filler_words.add(source)
                elif ' ' in source:
                    # Multi-word phrase with empty target = delete entire phrase
                    phrase_mappings[source] = ""
            else:
                # Multi-word source = phrase mapping
                if ' ' in source:
                    phrase_mappings[source] = target
                else:
                    word_mappings[source] = target
    
    return filler_words, phrase_mappings, word_mappings


def _get_default_mappings():
    """Default mappings if vi2en.txt not found"""
    filler_words = {"á", "à", "ạ", "ấy", "ơ", "ờ", "ờm", "ư", "ừ", "ừm", "ý", "ỳ"}
    phrase_mappings = {
        "quy rờ": "qr",
        "quy rò": "qr", 
        "việt nam": "vietnam",
        "ki lô mét": "km",
    }
    word_mappings = {
        "hông": "không",
        "vầng": "vâng",
        "nà": "là",
        "bẩy": "bảy",
    }
    return filler_words, phrase_mappings, word_mappings


# Load mappings at module import
_vi2en_path = _find_vi2en_file()
FILLER_WORDS, PHRASE_MAPPINGS, WORD_MAPPINGS = _load_mappings_from_file(_vi2en_path)

# Print stats on import
if _vi2en_path:
    print(f"Loaded normalization from: {_vi2en_path}")
    print(f"  - Filler words: {len(FILLER_WORDS)}")
    print(f"  - Phrase mappings: {len(PHRASE_MAPPINGS)}")
    print(f"  - Word mappings: {len(WORD_MAPPINGS)}")


def normalize_text(text: str, use_extended: bool = False) -> str:
    """
    Apply text normalizations from vi2en.txt:
    1. Unicode NFC normalization (fix NFD vs NFC mismatch)
    2. Lowercase
    3. Phrase-level mappings (exact word sequence match)
    4. Word-level mappings
    5. Remove filler words
    
    Args:
        text: Input text to normalize
        use_extended: Unused, kept for backward compatibility
    """
    # Step 1: Unicode NFC normalization - critical for Vietnamese text
    # This fixes cases where "là" (NFD) != "là" (NFC) even though they look identical
    text = unicodedata.normalize('NFC', text)
    
    text = text.lower()
    words = text.split()
    
    # Apply phrase mappings with EXACT word sequence matching
    # This prevents "em xi" matching inside "em xin"
    i = 0
    result_words = []
    
    while i < len(words):
        matched = False
        
        # Try matching phrases from longest to shortest (up to 6 words)
        for phrase_len in range(min(6, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + phrase_len])
            if phrase in PHRASE_MAPPINGS:
                replacement = PHRASE_MAPPINGS[phrase]
                if replacement:  # Non-empty replacement
                    # Split replacement and add each word
                    result_words.extend(replacement.split())
                # If empty replacement, just skip (delete phrase)
                i += phrase_len
                matched = True
                break
        
        if not matched:
            # No phrase match, try word mapping
            word = words[i]
            if word in WORD_MAPPINGS:
                replacement = WORD_MAPPINGS[word]
                if replacement:
                    result_words.append(replacement)
                # If empty, word is deleted
            elif word in FILLER_WORDS:
                pass  # Skip filler words
            else:
                result_words.append(word)
            i += 1
    
    return " ".join(result_words)


def remove_filler_words(words: List[str]) -> List[str]:
    """Remove Vietnamese filler words from a list of words for WER calculation."""
    return [w for w in words if w.lower() not in FILLER_WORDS]
