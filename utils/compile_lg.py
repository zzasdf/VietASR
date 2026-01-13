#!/usr/bin/env python3
"""
Compile Lexicon (L) and Language Model (G) into a decoding graph (LG).
"""

import argparse
import logging
import os
from pathlib import Path

import k2
import torch
from icefall.lexicon import Lexicon

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        required=True,
        help="Directory containing lexicon.txt",
    )
    parser.add_argument(
        "--lm-path",
        type=Path,
        required=True,
        help="Path to lm_4gram.arpa",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save LG.pt",
    )
    return parser

def compile_lg(lang_dir: Path, lm_path: Path, output_dir: Path):
    device = torch.device("cpu")
    
    # 1. Load Lexicon
    logging.info(f"Loading lexicon from {lang_dir}")
    # Ensure lexicon.txt exists
    if not (lang_dir / "lexicon.txt").exists():
        raise FileNotFoundError(f"{lang_dir}/lexicon.txt not found")
        
    lexicon = Lexicon(lang_dir)
    
    # 2. Compile L (Lexicon FST)
    logging.info("Compiling L.pt...")
    L = lexicon.L
    torch.save(L.as_dict(), output_dir / "L.pt")
    
    # 3. Compile G (Grammar FST from ARPA)
    logging.info(f"Compiling G from {lm_path}...")
    
    # Check if we can use k2's internal tools or need to use shell
    # For simplicity and robustness in this environment, we use k2.Fsa.from_arpa if available
    # or we construct it via standard method.
    
    # Actually, icefall usually has a specific flow. 
    # Let's try to read ARPA directly.
    
    with open(lm_path) as f:
        arpa_content = f.read()
        
    # Note: k2 doesn't have a direct python 'from_arpa' that takes a file path easily in all versions.
    # But usually we use a shell command `python3 -m k2.bin.arpa2fst`
    # Let's try to run the shell command for robustness as it's the standard way in k2 recipes.
    
    arpa_output_path = output_dir / "G.fst.txt"
    
    # We will use os.system to call the k2 binary tool if possible, or use a python binding if I can find it.
    # Let's assume we can use the python binding for L, but for G from ARPA, 
    # the standard is often `python3 -m k2.bin.arpa2fst`.
    
    # However, to keep it pure python if possible:
    # There isn't a simple one-liner in k2 python API to load ARPA to FSA without external tools often.
    # Let's use the shell command approach which is safer.
    
    cmd = f"python3 -m k2.bin.arpa2fst --read-symbol-table={lang_dir}/words.txt --keep-order=false {lm_path} {output_dir}/G.pt"
    logging.info(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("Failed to compile G.pt")
        
    # 4. Load G
    logging.info("Loading G.pt...")
    G = k2.Fsa.from_dict(torch.load(output_dir / "G.pt", map_location=device))
    
    # 5. Compose L and G -> LG
    logging.info("Composing L and G...")
    LG = k2.compose(L, G)
    
    # 6. Connect and Sort
    logging.info("Connecting and sorting LG...")
    LG = k2.connect(LG)
    LG = k2.arc_sort(LG)
    
    # 7. Save LG
    output_file = output_dir / "LG.pt"
    logging.info(f"Saving LG to {output_file}")
    torch.save(LG.as_dict(), output_file)
    
    logging.info("Done!")

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    logging.getLogger().setLevel(logging.INFO)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    compile_lg(args.lang_dir, args.lm_path, args.output_dir)

if __name__ == "__main__":
    main()
