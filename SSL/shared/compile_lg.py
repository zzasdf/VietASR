#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import k2
import torch
from icefall.lexicon import Lexicon
from icefall.utils import str2bool

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        help="Directory containing tokens.txt",
    )
    parser.add_argument(
        "--lm-dir",
        type=Path,
        help="Directory containing lm_4gram.arpa",
    )
    return parser

def compile_lg(lang_dir: Path, lm_dir: Path):
    lexicon = Lexicon(lang_dir)
    device = torch.device("cpu")
    
    logging.info("Loading L_disambig.pt")
    if (lang_dir / "L_disambig.pt").exists():
        L = k2.Fsa.from_dict(torch.load(lang_dir / "L_disambig.pt", map_location=device))
    else:
        # If L_disambig.pt doesn't exist, we might need to create it or use L.pt
        # For now assuming L.pt exists or we can create it from lexicon
        logging.warning("L_disambig.pt not found, trying to load L.pt")
        L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt", map_location=device))

    logging.info("Loading G.fst")
    # We assume G is already compiled to FST or we need to compile it from ARPA
    # Usually we compile ARPA to G.fst (or G.pt)
    # If G.pt doesn't exist, we should create it from ARPA
    
    lm_path = lm_dir / "lm_4gram.arpa"
    if not lm_path.exists():
         raise FileNotFoundError(f"{lm_path} not found")

    # Use k2 to compile ARPA to G
    # This requires 'arpa2fst' binary or python binding if available.
    # Since we are in python, we might check if we can use k2's arpa loading if available, 
    # but usually we use shell commands for this.
    # However, let's try to use a python approach if possible or assume G.pt creation is separate.
    # Wait, the user asked to "compile HLG", so I should probably do the whole pipeline.
    
    # Let's use a simplified approach:
    # 1. Load L
    # 2. Compile G from ARPA (using shell command inside python if needed)
    # 3. Compose L and G
    
    pass

if __name__ == "__main__":
    # This is a placeholder. I need to check if I can run arpa2fst or similar.
    # I will write a better script in the next step after verifying tools.
    pass
