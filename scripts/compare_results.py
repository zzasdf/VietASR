#!/usr/bin/env python3
"""
So sánh kết quả từ 2 models (có thể từ 2 repos khác nhau).

Workflow:
1. Chạy model A (VietASR repo) → export ra CSV
2. Chạy model B (Team's repo) → export ra CSV cùng format
3. Chạy script này để so sánh

CSV Format (cần giống nhau cho cả 2 models):
    audio_path,duration_sec,duration_bin,reference,hypothesis,latency_sec,rtf

Usage:
    python scripts/compare_results.py \
        --model-a-csv /path/to/model_a_results.csv \
        --model-a-name "Zipformer_VietASR" \
        --model-b-csv /path/to/model_b_results.csv \
        --model-b-name "Team_API_Model" \
        --output-dir results/comparison
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import sys


# Duration bins
DURATION_BINS = ["0-4s", "4-8s", "8-12s", "12-16s", "16-20s", "20s+"]


def calculate_wer(reference: str, hypothesis: str) -> Tuple[int, int]:
    """Calculate word error rate components."""
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()
    
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n], len(ref_words)


@dataclass
class SampleResult:
    audio_path: str
    duration: float
    duration_bin: str
    reference: str
    hypothesis: str
    latency: float
    rtf: float


@dataclass
class BinStats:
    count: int = 0
    total_duration: float = 0.0
    total_latency: float = 0.0
    total_words: int = 0
    total_errors: int = 0
    
    @property
    def rtf(self) -> float:
        return self.total_latency / self.total_duration if self.total_duration > 0 else 0
    
    @property
    def wer(self) -> float:
        return (self.total_errors / self.total_words * 100) if self.total_words > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.count if self.count > 0 else 0


def load_csv_results(csv_path: str) -> Dict[str, List[SampleResult]]:
    """Load results from CSV and group by duration bin."""
    bins = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = SampleResult(
                audio_path=row['audio_path'],
                duration=float(row['duration_sec']),
                duration_bin=row['duration_bin'],
                reference=row['reference'],
                hypothesis=row['hypothesis'],
                latency=float(row['latency_sec']),
                rtf=float(row['rtf']),
            )
            bins[sample.duration_bin].append(sample)
    
    return bins


def compute_stats(samples: List[SampleResult]) -> BinStats:
    """Compute statistics for a list of samples."""
    stats = BinStats()
    
    for sample in samples:
        stats.count += 1
        stats.total_duration += sample.duration
        stats.total_latency += sample.latency
        
        errors, words = calculate_wer(sample.reference, sample.hypothesis)
        stats.total_errors += errors
        stats.total_words += words
    
    return stats


def print_comparison_table(
    model_a_bins: Dict[str, List[SampleResult]],
    model_b_bins: Dict[str, List[SampleResult]],
    model_a_name: str,
    model_b_name: str,
):
    """Print comparison table."""
    
    print("\n" + "=" * 120)
    print(f"COMPARISON: {model_a_name} vs {model_b_name}")
    print("=" * 120)
    
    # WER Comparison
    print("\n--- WER (%) by Duration Bin ---")
    print(f"{'Duration':<12} | {model_a_name:>20} | {model_b_name:>20} | {'Diff':>12} | {'Winner':>15}")
    print("-" * 90)
    
    for bin_name in DURATION_BINS:
        a_samples = model_a_bins.get(bin_name, [])
        b_samples = model_b_bins.get(bin_name, [])
        
        a_stats = compute_stats(a_samples) if a_samples else None
        b_stats = compute_stats(b_samples) if b_samples else None
        
        a_wer = f"{a_stats.wer:.2f}" if a_stats and a_stats.count > 0 else "-"
        b_wer = f"{b_stats.wer:.2f}" if b_stats and b_stats.count > 0 else "-"
        
        if a_stats and b_stats and a_stats.count > 0 and b_stats.count > 0:
            diff = b_stats.wer - a_stats.wer  # Positive = A is better
            diff_str = f"{diff:+.2f}"
            winner = model_a_name if diff > 0 else model_b_name if diff < 0 else "TIE"
        else:
            diff_str = "-"
            winner = "-"
        
        print(f"{bin_name:<12} | {a_wer:>20} | {b_wer:>20} | {diff_str:>12} | {winner:>15}")
    
    # RTF Comparison
    print("\n--- RTF (Real Time Factor) by Duration Bin ---")
    print(f"{'Duration':<12} | {model_a_name:>20} | {model_b_name:>20} | {'Ratio':>12} | {'Faster':>15}")
    print("-" * 90)
    
    for bin_name in DURATION_BINS:
        a_samples = model_a_bins.get(bin_name, [])
        b_samples = model_b_bins.get(bin_name, [])
        
        a_stats = compute_stats(a_samples) if a_samples else None
        b_stats = compute_stats(b_samples) if b_samples else None
        
        a_rtf = f"{a_stats.rtf:.4f}" if a_stats and a_stats.count > 0 else "-"
        b_rtf = f"{b_stats.rtf:.4f}" if b_stats and b_stats.count > 0 else "-"
        
        if a_stats and b_stats and a_stats.count > 0 and b_stats.count > 0 and a_stats.rtf > 0:
            ratio = b_stats.rtf / a_stats.rtf  # >1 = A is faster
            ratio_str = f"{ratio:.2f}x"
            faster = model_a_name if ratio > 1 else model_b_name if ratio < 1 else "TIE"
        else:
            ratio_str = "-"
            faster = "-"
        
        print(f"{bin_name:<12} | {a_rtf:>20} | {b_rtf:>20} | {ratio_str:>12} | {faster:>15}")
    
    # Latency Comparison
    print("\n--- Average Latency (seconds) by Duration Bin ---")
    print(f"{'Duration':<12} | {model_a_name:>20} | {model_b_name:>20} | {'Count A':>10} | {'Count B':>10}")
    print("-" * 90)
    
    for bin_name in DURATION_BINS:
        a_samples = model_a_bins.get(bin_name, [])
        b_samples = model_b_bins.get(bin_name, [])
        
        a_stats = compute_stats(a_samples) if a_samples else None
        b_stats = compute_stats(b_samples) if b_samples else None
        
        a_lat = f"{a_stats.avg_latency:.4f}s" if a_stats and a_stats.count > 0 else "-"
        b_lat = f"{b_stats.avg_latency:.4f}s" if b_stats and b_stats.count > 0 else "-"
        
        a_count = a_stats.count if a_stats else 0
        b_count = b_stats.count if b_stats else 0
        
        print(f"{bin_name:<12} | {a_lat:>20} | {b_lat:>20} | {a_count:>10} | {b_count:>10}")
    
    print("=" * 120)


def save_comparison_report(
    model_a_bins: Dict[str, List[SampleResult]],
    model_b_bins: Dict[str, List[SampleResult]],
    model_a_name: str,
    model_b_name: str,
    output_dir: Path,
):
    """Save comparison to JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "bins": {}
    }
    
    for bin_name in DURATION_BINS:
        a_stats = compute_stats(model_a_bins.get(bin_name, []))
        b_stats = compute_stats(model_b_bins.get(bin_name, []))
        
        report["bins"][bin_name] = {
            "model_a": {
                "count": a_stats.count,
                "wer": a_stats.wer,
                "rtf": a_stats.rtf,
                "avg_latency": a_stats.avg_latency,
            },
            "model_b": {
                "count": b_stats.count,
                "wer": b_stats.wer,
                "rtf": b_stats.rtf,
                "avg_latency": b_stats.avg_latency,
            }
        }
    
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Markdown report
    with open(output_dir / "comparison.md", "w") as f:
        f.write(f"# Model Comparison: {model_a_name} vs {model_b_name}\n\n")
        
        f.write("## WER by Duration\n\n")
        f.write(f"| Duration | {model_a_name} | {model_b_name} | Diff |\n")
        f.write("|----------|--------------|--------------|------|\n")
        
        for bin_name in DURATION_BINS:
            a_stats = compute_stats(model_a_bins.get(bin_name, []))
            b_stats = compute_stats(model_b_bins.get(bin_name, []))
            diff = b_stats.wer - a_stats.wer if a_stats.count > 0 and b_stats.count > 0 else 0
            f.write(f"| {bin_name} | {a_stats.wer:.2f}% | {b_stats.wer:.2f}% | {diff:+.2f} |\n")
        
        f.write("\n## RTF by Duration\n\n")
        f.write(f"| Duration | {model_a_name} | {model_b_name} |\n")
        f.write("|----------|--------------|---------------|\n")
        
        for bin_name in DURATION_BINS:
            a_stats = compute_stats(model_a_bins.get(bin_name, []))
            b_stats = compute_stats(model_b_bins.get(bin_name, []))
            f.write(f"| {bin_name} | {a_stats.rtf:.4f} | {b_stats.rtf:.4f} |\n")
    
    print(f"\nReport saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare results from two ASR models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-a-csv", type=str, required=True, help="CSV results from model A")
    parser.add_argument("--model-a-name", type=str, default="Model_A", help="Name for model A")
    parser.add_argument("--model-b-csv", type=str, required=True, help="CSV results from model B")
    parser.add_argument("--model-b-name", type=str, default="Model_B", help="Name for model B")
    parser.add_argument("--output-dir", type=str, default="results/comparison", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Loading Model A results: {args.model_a_csv}")
    model_a_bins = load_csv_results(args.model_a_csv)
    
    print(f"Loading Model B results: {args.model_b_csv}")
    model_b_bins = load_csv_results(args.model_b_csv)
    
    print_comparison_table(model_a_bins, model_b_bins, args.model_a_name, args.model_b_name)
    
    save_comparison_report(
        model_a_bins, model_b_bins,
        args.model_a_name, args.model_b_name,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
