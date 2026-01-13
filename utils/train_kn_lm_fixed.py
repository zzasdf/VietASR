#!/usr/bin/env python3
"""
Simple script to train word-level n-gram LM using modified make_kn_lm.py
with better error handling for sparse data.
"""

import sys
import os

# Add path for make_kn_lm
sys.path.insert(0, 'SSL/shared')

# Monkey-patch to handle sparse n-gram data
import SSL.shared.make_kn_lm as kn_module

# Save original cal_discounting_constants
original_cal_discounting = kn_module.NgramCounts.cal_discounting_constants

def cal_discounting_constants_fixed(self):
    """Fixed version with fallback for sparse data"""
    self.d = [0]  # For unigram
    for n in range(1, self.ngram_order):
        this_order_counts = self.counts[n]
        n1 = 0
        n2 = 0
        for hist, counts_for_hist in this_order_counts.items():
            from collections import Counter
            stat = Counter(counts_for_hist.word_to_count.values())
            n1 += stat[1]
            n2 += stat[2]
        
        # Fallback: if n1 and n2 are both 0, use a small default discount
        if n1 + 2 * n2 == 0:
            print(f"Warning: Sparse data at ngram order {n+1}, using default discount 0.5", file=sys.stderr)
            self.d.append(0.5)
        else:
            # Original formula
            self.d.append(max(0.1, n1 * 1.0) / (n1 + 2 * n2))

# Apply monkey patch
kn_module.NgramCounts.cal_discounting_constants = cal_discounting_constants_fixed

# Now run the original script
if __name__ == "__main__":
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ngram-order", type=int, default=3)
    parser.add_argument("-text", type=str, required=True)
    parser.add_argument("-lm", type=str, required=True)
    parser.add_argument("-verbose", type=int, default=1)
    args = parser.parse_args()
    
    # Create NgramCounts
    ngram_counts = kn_module.NgramCounts(args.ngram_order)
    
    # Add counts from file
    if os.path.isfile(args.text):
        ngram_counts.add_raw_counts_from_file(args.text)
    else:
        print(f"Error: File {args.text} not found", file=sys.stderr)
        sys.exit(1)
    
    # Calculate discounting constants (with fix)
    ngram_counts.cal_discounting_constants()
    print(f"Discounting constants: {ngram_counts.d}", file=sys.stderr)
    
    # Calculate probabilities
    ngram_counts.cal_f()
    ngram_counts.cal_bow()
    
    # Write ARPA
    with open(args.lm, "w", encoding="latin-1") as f:
        ngram_counts.print_as_arpa(fout=f)
    
    print(f"✅ LM written to {args.lm}", file=sys.stderr)
