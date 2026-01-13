#!/usr/bin/env python3
"""
Phân tích WER distribution để chọn threshold tối ưu
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import jiwer

def parse_recogs_file(path):
    """Parse recogs file and compute WER for each cut."""
    results = []
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if ':' in lines[i] and 'ref=' in lines[i]:
            cut_id = lines[i].split(':')[0].strip()
            ref_line = lines[i]
            
            if i + 1 < len(lines) and 'hyp=' in lines[i + 1]:
                hyp_line = lines[i + 1]
                
                # Extract ref and hyp
                try:
                    ref_start = ref_line.find("ref=[")
                    ref_end = ref_line.rfind("]")
                    ref_str = ref_line[ref_start+5:ref_end]
                    ref_words = [w.strip("' ") for w in ref_str.split(',')]
                    
                    hyp_start = hyp_line.find("hyp=[")
                    hyp_end = hyp_line.rfind("]")
                    hyp_str = hyp_line[hyp_start+5:hyp_end]
                    hyp_words = [w.strip("' ") for w in hyp_str.split(',')]
                    
                    ref_text = ' '.join(ref_words)
                    hyp_text = ' '.join(hyp_words)
                    
                    if ref_text.strip():
                        wer = jiwer.wer(ref_text, hyp_text)
                        results.append({
                            'cut_id': cut_id,
                            'wer': wer,
                            'ref': ref_text,
                            'hyp': hyp_text,
                            'ref_len': len(ref_words)
                        })
                except:
                    pass
                
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return results

def analyze_wer_distribution(results):
    """Analyze WER distribution."""
    wer_buckets = defaultdict(int)
    
    for r in results:
        wer = r['wer']
        if wer == 0:
            bucket = "0%"
        elif wer <= 0.1:
            bucket = "1-10%"
        elif wer <= 0.2:
            bucket = "11-20%"
        elif wer <= 0.3:
            bucket = "21-30%"
        elif wer <= 0.4:
            bucket = "31-40%"
        elif wer <= 0.5:
            bucket = "41-50%"
        elif wer <= 0.6:
            bucket = "51-60%"
        elif wer <= 0.7:
            bucket = "61-70%"
        elif wer <= 0.8:
            bucket = "71-80%"
        elif wer <= 0.9:
            bucket = "81-90%"
        else:
            bucket = "91-100%+"
        
        wer_buckets[bucket] += 1
    
    return wer_buckets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recogs-dir", type=Path, 
                        default=Path("data4/exp/modified_beam_search_use_avg"))
    parser.add_argument("--show-worst", type=int, default=10,
                        help="Show N worst examples")
    args = parser.parse_args()
    
    # Find all recogs files (exclude test)
    recogs_files = list(args.recogs_dir.glob("recogs-*.txt"))
    recogs_files = [f for f in recogs_files if "test" not in f.name]
    
    print("=" * 70)
    print("PHÂN TÍCH WER DISTRIBUTION")
    print("=" * 70)
    
    all_results = []
    
    for recogs_file in sorted(recogs_files):
        dataset_name = recogs_file.name.split("-")[1]
        print(f"\n--- {dataset_name} ---")
        
        results = parse_recogs_file(recogs_file)
        all_results.extend(results)
        
        if not results:
            print("  No results parsed")
            continue
        
        # WER distribution
        buckets = analyze_wer_distribution(results)
        total = len(results)
        
        print(f"  Total cuts: {total}")
        print(f"  WER Distribution:")
        
        bucket_order = ["0%", "1-10%", "11-20%", "21-30%", "31-40%", 
                        "41-50%", "51-60%", "61-70%", "71-80%", "81-90%", "91-100%+"]
        
        for bucket in bucket_order:
            count = buckets.get(bucket, 0)
            pct = count * 100 / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {bucket:>10}: {count:>6} ({pct:>5.1f}%) {bar}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("TỔNG HỢP TẤT CẢ CÁC TẬP")
    print("=" * 70)
    
    buckets = analyze_wer_distribution(all_results)
    total = len(all_results)
    
    print(f"Total cuts: {total}")
    print(f"\nWER Distribution:")
    
    cumulative = 0
    for bucket in bucket_order:
        count = buckets.get(bucket, 0)
        cumulative += count
        pct = count * 100 / total if total > 0 else 0
        cum_pct = cumulative * 100 / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>10}: {count:>6} ({pct:>5.1f}%) [Cumulative: {cum_pct:>5.1f}%] {bar}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("KHUYẾN NGHỊ THRESHOLD")
    print("=" * 70)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for thresh in thresholds:
        kept = sum(1 for r in all_results if r['wer'] <= thresh)
        removed = total - kept
        kept_pct = kept * 100 / total if total > 0 else 0
        print(f"  WER <= {thresh*100:.0f}%: Giữ {kept:,} cuts ({kept_pct:.1f}%), Xóa {removed:,}")
    
    # Show worst examples
    print("\n" + "=" * 70)
    print(f"TOP {args.show_worst} MẪU WER CAO NHẤT")
    print("=" * 70)
    
    sorted_results = sorted(all_results, key=lambda x: -x['wer'])
    for i, r in enumerate(sorted_results[:args.show_worst]):
        print(f"\n[{i+1}] {r['cut_id']} - WER: {r['wer']*100:.1f}%")
        print(f"  REF: {r['ref'][:100]}...")
        print(f"  HYP: {r['hyp'][:100]}...")

if __name__ == "__main__":
    main()
