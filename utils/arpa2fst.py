#!/usr/bin/env python3
"""
Convert ARPA LM to FST (G.fst) using kaldifst
Usage: python utils/arpa2fst.py --arpa <arpa_file> --fst <fst_file> --read-symbol-table <words.txt>
"""

import argparse
import sys
import math
import logging
from collections import defaultdict

try:
    import kaldifst
except ImportError:
    print("Error: kaldifst not installed. Please install it first.")
    sys.exit(1)

def load_symbol_table(filename):
    sym2id = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sym2id[parts[0]] = int(parts[1])
    return sym2id

def parse_arpa(arpa_file):
    ngrams = defaultdict(dict)
    with open(arpa_file, 'r', encoding='utf-8') as f:
        section = None
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('\\data\\'): continue
            if line.startswith('\\end\\'): break
            if line.startswith('\\') and line.endswith('-grams:'):
                section = int(line[1:].split('-')[0])
                continue
            if section is not None:
                parts = line.split()
                try:
                    prob = float(parts[0])
                    if len(parts) > 1:
                        # Check for backoff
                        try:
                            backoff = float(parts[-1])
                            words = tuple(parts[1:-1])
                        except ValueError:
                            backoff = 0.0
                            words = tuple(parts[1:])
                        ngrams[section][words] = (prob, backoff)
                except ValueError:
                    continue
    return ngrams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arpa", required=True, help="Input ARPA file")
    parser.add_argument("--fst", required=True, help="Output FST file")
    parser.add_argument("--read-symbol-table", required=True, help="Symbol table (words.txt)")
    args = parser.parse_args()

    sym2id = load_symbol_table(args.read_symbol_table)
    ngrams = parse_arpa(args.arpa)
    
    # Build FST
    # State mapping: history -> state_id
    # <s> is start state. </s> is end state (or transition to it).
    # In standard G.fst:
    # Start state corresponds to <s> history.
    # We need to handle <s> and </s> explicitly.
    
    # Simplified construction:
    # States are histories.
    # 1-gram history: () (empty) -> usually represents <s> context if we treat <s> as history?
    # No, standard:
    # State for <s> is start.
    # State for </s> is final? No, </s> is a symbol.
    
    # Let's follow Kaldi's arpa2fst logic roughly.
    # But simpler:
    # States are tuples of words (history).
    # Max order N.
    # State is (w_{i-N+2}, ..., w_i).
    
    # Collect all histories
    histories = set()
    histories.add(("<s>",)) # Start state history
    
    max_order = max(ngrams.keys())
    
    for order in range(1, max_order + 1):
        for ngram in ngrams[order]:
            # ngram is (w1, ..., wn)
            # History for this ngram is (w1, ..., wn-1)
            # Target state history is (w2, ..., wn) (if order < N)
            
            # We need states for all prefixes that appear as histories.
            if order == 1:
                # 1-gram (w). History is empty? Or <s>?
                # Usually unigram state is the backoff state for everything.
                histories.add(()) 
            else:
                histories.add(ngram[:-1])
                
    # Also add states for complete n-grams if they are histories for higher orders
    # But we only have up to max_order.
    # So for max_order n-grams, we don't need a state for the full n-gram unless it's a backoff state?
    # Actually, for 3-gram LM:
    # 3-gram (u, v, w). Transition from state (u, v) to state (v, w).
    # So we need states for all (n-1)-grams.
    
    # Let's refine:
    # States are histories of length 0 to N-1.
    # 0-length: () -> Unigram state (backoff for 1-grams)
    # 1-length: (w)
    # ...
    # N-1 length.
    
    # Collect all (n-1) suffixes of n-grams
    states = set()
    states.add(()) # Unigram state
    states.add(("<s>",)) # Start state
    
    for order in range(1, max_order + 1):
        for ngram in ngrams[order]:
            # ngram: w1 ... wn
            # It represents a transition.
            # From state: w1 ... wn-1
            # To state: w2 ... wn
            
            # We need to ensure state w1...wn-1 exists.
            # And state w2...wn exists (or backoff).
            
            if len(ngram) < max_order:
                states.add(ngram)
            else:
                # For max order, the target state is the suffix of length N-1
                states.add(ngram[1:])
            
            states.add(ngram[:-1])

    # Map states to IDs
    # Start state is ("<s>",)
    state2id = {}
    # Ensure start state is 0? Not required but good.
    if ("<s>",) in states:
        state2id[("<s>",)] = 0
    else:
        # Should not happen if <s> is in ARPA
        state2id[("<s>",)] = 0
        
    idx = 1
    for s in states:
        if s != ("<s>",):
            state2id[s] = idx
            idx += 1
            
    fst = kaldifst.StdVectorFst()
    for _ in range(len(state2id)):
        fst.add_state()
        
    fst.start = state2id[("<s>",)]
    
    # Add arcs
    # For each ngram (w1...wn) with prob p, backoff b:
    # Source state: (w1...wn-1)
    # Label: wn
    # Cost: -p * ln(10)
    # Dest state: (w2...wn) if exists, else backoff.
    
    # Also add backoff arcs.
    # For each state (w1...wk), backoff to (w2...wk).
    # Cost: -b * ln(10). Label: epsilon (0).
    
    # Wait, backoff b is stored with the ngram that CREATED the state.
    # i.e. backoff for state (w1...wk) comes from entry (w1...wk) in ARPA.
    
    scale = math.log(10)
    
    for order in range(1, max_order + 1):
        for ngram, (prob, backoff) in ngrams[order].items():
            # 1. Add the arc for this ngram
            src_hist = ngram[:-1]
            word = ngram[-1]
            
            if src_hist not in state2id: continue # Should not happen
            src = state2id[src_hist]
            
            # Dest state
            if len(ngram) < max_order:
                dest_hist = ngram
            else:
                dest_hist = ngram[1:]
                
            if dest_hist in state2id:
                dest = state2id[dest_hist]
            else:
                # Fallback? Should exist if we collected correctly.
                # If not, backoff to shorter suffix
                temp = dest_hist[1:]
                while temp not in state2id and len(temp) > 0:
                    temp = temp[1:]
                dest = state2id[temp]
                
            word_id = sym2id.get(word, sym2id.get("<unk>", 0))
            cost = -prob * scale
            
            fst.add_arc(src, kaldifst.StdArc(word_id, word_id, cost, dest))
            
            # 2. Add backoff arc FROM this state (if it is a state)
            if ngram in state2id:
                # Backoff state is suffix
                backoff_hist = ngram[1:]
                if backoff_hist in state2id:
                    backoff_dest = state2id[backoff_hist]
                    backoff_cost = -backoff * scale
                    # Epsilon arc
                    fst.add_arc(state2id[ngram], kaldifst.StdArc(0, 0, backoff_cost, backoff_dest))

    # Handle </s>
    # Usually </s> is a transition to a final state or just a symbol.
    # In ARPA, </s> appears as a word.
    # If we treat it as a word, we transition to a state ending in </s>.
    # That state should be final.
    
    # Let's iterate states and see which ones end in </s>
    for hist, sid in state2id.items():
        if len(hist) > 0 and hist[-1] == "</s>":
            fst.set_final(sid, 0.0)
            
    # Also handle unigram state backoff to empty?
    # State () backoff? No.
    
    # Write FST
    fst.write(args.fst)
    print(f"Written FST to {args.fst}")

if __name__ == "__main__":
    main()
