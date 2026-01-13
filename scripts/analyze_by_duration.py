#!/usr/bin/env python3
"""
Phân tích WER và Latency theo độ dài audio.

Script này:
1. Load audio từ thư mục test
2. Phân loại theo duration bins: 0-4s, 4-8s, 8-12s, 12-16s, 16-20s, 20s+
3. Chạy inference và đo latency cho từng sample
4. Tính WER và RTF theo từng bin

Usage:
    python scripts/analyze_by_duration.py \
        --test-dir /data/raw/test/tongdai_clean \
        --exp-dir /vietasr/data4/exp_finetune_v3 \
        --epoch 19 \
        --avg 17 \
        --output-dir results/duration_analysis

"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import kaldifeat
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

# Add paths
sys.path.insert(0, "/vietasr/SSL/zipformer_fbank")
sys.path.insert(0, "/vietasr/scripts/data")  # For normalize_transcripts

from icefall.utils import make_pad_mask
from icefall.checkpoint import average_checkpoints_with_averaged_model, load_checkpoint
from finetune import add_model_arguments, get_params, get_model
from beam_search import modified_beam_search, greedy_search_batch

# Import Vietnamese text normalization (uses vi2en.txt)
from text_normalization import normalize_text

# Import normalize_transcripts functions (from decode.sh pipeline)
try:
    from normalize_transcripts import (
        normalize_tone_and_typos,
        normalize_numbers,
        normalize_punctuation
    )
    HAS_NORMALIZE_TRANSCRIPTS = True
except ImportError:
    HAS_NORMALIZE_TRANSCRIPTS = False
    logging.warning("Could not import normalize_transcripts, using text_normalization only")


def full_normalize(text: str) -> str:
    """
    Full normalization pipeline matching decode.sh order:
    1. normalize_tone_and_typos
    2. normalize_numbers  
    3. normalize_punctuation
    4. normalize_text (vi2en.txt) - for scoring consistency only
    """
    if HAS_NORMALIZE_TRANSCRIPTS:
        # decode.sh order
        text = normalize_tone_and_typos(text)
        text = normalize_numbers(text)
        text = normalize_punctuation(text)
    
    # vi2en.txt for scoring consistency (lowercase + mappings)
    text = normalize_text(text)
    
    return text


LOG_EPS = math.log(1e-10)

# Duration bins (seconds)
DURATION_BINS = [
    (0, 4, "0-4s"),
    (4, 8, "4-8s"),
    (8, 12, "8-12s"),
    (12, 16, "12-16s"),
    (16, 20, "16-20s"),
    (20, float('inf'), "20s+"),
]


@dataclass
class SampleResult:
    """Result for a single audio sample."""
    audio_path: str
    duration: float
    duration_bin: str
    reference: str
    hypothesis: str
    latency_seconds: float
    rtf: float
    wer: float = 0.0


@dataclass  
class BinStats:
    """Statistics for a duration bin."""
    bin_name: str
    count: int = 0
    total_duration: float = 0.0
    total_latency: float = 0.0
    total_words: int = 0
    total_errors: int = 0
    samples: List[SampleResult] = field(default_factory=list)
    
    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.count if self.count > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.count if self.count > 0 else 0
    
    @property
    def rtf(self) -> float:
        return self.total_latency / self.total_duration if self.total_duration > 0 else 0
    
    @property
    def wer(self) -> float:
        return (self.total_errors / self.total_words * 100) if self.total_words > 0 else 0


def get_duration_bin(duration: float) -> str:
    """Get the bin name for a given duration."""
    for min_d, max_d, name in DURATION_BINS:
        if min_d <= duration < max_d:
            return name
    return DURATION_BINS[-1][2]  # Default to last bin


def calculate_wer(reference: str, hypothesis: str) -> Tuple[int, int]:
    """
    Calculate word error rate components.
    Applies full_normalize (decode.sh pipeline) to both texts for fair comparison.
    
    Returns:
        (num_errors, num_words)
    """
    # Full normalization matching decode.sh pipeline
    ref_norm = full_normalize(reference)
    hyp_norm = full_normalize(hypothesis)
    
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    
    # Simple Levenshtein distance
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


def load_test_data(test_dir: str) -> List[Tuple[str, str]]:
    """
    Load test data from directory.
    
    Expects structure:
        test_dir/
            wavs/
            transcripts.txt  (format: audio_path|text|duration or audio_path|text)
    
    Actual format in transcripts.txt:
        /full/path/to/audio.wav|transcript text here|duration_in_seconds
        
    Returns:
        List of (audio_path, reference_text)
    """
    test_dir = Path(test_dir)
    samples = []
    
    # Try different transcript file names (in order of priority)
    transcript_files = ["transcripts.txt", "transcript.txt", "text.txt", "meta.txt"]
    transcript_file = None
    for tf in transcript_files:
        if (test_dir / tf).exists():
            transcript_file = test_dir / tf
            logging.info(f"Found transcript file: {transcript_file}")
            break
    
    if not transcript_file:
        logging.error(f"No transcript file found in {test_dir}")
        logging.error(f"Tried: {transcript_files}")
        return samples
    
    # Parse transcript file
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Split by pipe '|'
            parts = line.split('|')
            
            if len(parts) >= 2:
                # Format: path|text or path|text|duration
                audio_path = parts[0].strip()
                text = parts[1].strip()
                
                # =============================================
                # ESPnet filtering logic (from convert_data.py)
                # =============================================
                
                # Filter 1: Skip samples containing '111' (invalid/test samples)
                if '111' in text:
                    logging.debug(f"Skipping sample with '111': {audio_path}")
                    continue
                
                # Filter 2: Clean text - remove ' ck' and ' spkt' suffixes
                text = text.replace(' ck', '').replace(' spkt', '')
                text = text.strip()
                
                # Skip empty text after cleaning
                if not text:
                    logging.debug(f"Skipping empty text after cleaning: {audio_path}")
                    continue
                
                # =============================================
                
                # Handle case where path in file doesn't match actual location
                # The file might have a different base path
                if not os.path.exists(audio_path):
                    # Try relative to test_dir/wavs/
                    filename = Path(audio_path).name
                    wav_dir = test_dir / "wavs"
                    if wav_dir.exists():
                        alt_path = wav_dir / filename
                        if alt_path.exists():
                            audio_path = str(alt_path)
                        else:
                            logging.debug(f"Audio not found: {audio_path}")
                            continue
                    else:
                        logging.debug(f"Audio not found: {audio_path}")
                        continue
                
                samples.append((audio_path, text))
            else:
                logging.warning(f"Line {line_num}: Invalid format (expected pipe-delimited): {line[:50]}...")
    
    logging.info(f"Loaded {len(samples)} samples from {test_dir}")
    
    # Log sample distribution by duration (if we have duration info)
    if samples:
        logging.info(f"First sample: {samples[0][0]}")
        logging.info(f"Last sample: {samples[-1][0]}")
    
    return samples


class ASRInference:
    """ASR inference wrapper for benchmarking."""
    
    def __init__(
        self,
        exp_dir: str,
        bpe_model: str,
        epoch: int,
        avg: int,
        device: str = "cuda",
        beam_size: int = 10,
        method: str = "modified_beam_search",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size
        self.method = method
        self.sample_rate = 16000
        
        logging.info(f"Initializing ASR on {self.device}")
        
        # Load BPE
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        
        # Build model
        self._build_model(exp_dir, epoch, avg)
        
        # Init Fbank
        self._init_fbank()
        
        logging.info("ASR ready")
    
    def _build_model(self, exp_dir: str, epoch: int, avg: int):
        """Build and load model."""
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args, _ = parser.parse_known_args([])
        
        self.params = get_params()
        self.params.update(vars(args))
        self.params.blank_id = self.sp.piece_to_id("<blk>")
        self.params.unk_id = self.sp.piece_to_id("<unk>")
        self.params.vocab_size = self.sp.get_piece_size()
        self.params.num_classes = [504]
        self.params.use_transducer = True
        self.params.use_ctc = False
        self.params.context_size = 2
        self.params.decoder_dim = 512
        self.params.joiner_dim = 512
        self.params.final_downsample = 1
        self.params.use_layer_norm = 0
        self.params.sample_rate = self.sample_rate
        self.params.feature_dim = 80
        
        self.model = get_model(self.params)
        
        # Load checkpoint
        exp_dir = Path(exp_dir)
        if avg == 1:
            load_checkpoint(str(exp_dir / f"epoch-{epoch}.pt"), self.model)
        else:
            start_epoch = max(1, epoch - avg)
            self.model.to(self.device)
            self.model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=str(exp_dir / f"epoch-{start_epoch}.pt"),
                    filename_end=str(exp_dir / f"epoch-{epoch}.pt"),
                    device=self.device,
                )
            )
        
        self.model.to(self.device)
        self.model.eval()
    
    def _init_fbank(self):
        """Initialize Fbank extractor."""
        opts = kaldifeat.FbankOptions()
        opts.device = self.device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = 80
        self.fbank = kaldifeat.Fbank(opts)
    
    @torch.no_grad()
    def transcribe_with_timing(self, audio_path: str) -> Tuple[str, float, float]:
        """
        Transcribe audio and measure latency.
        
        Returns:
            (hypothesis, latency_seconds, duration_seconds)
        """
        # Load audio
        wave, sr = torchaudio.load(audio_path)
        duration = wave.shape[1] / sr
        
        if sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, sr, self.sample_rate)
        
        wave = wave[0].contiguous().to(self.device)
        
        # Start timing from feature extraction
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        # Feature extraction
        features = self.fbank([wave])
        feature_lengths = torch.tensor([features[0].size(0)], device=self.device)
        features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)
        padding_mask = make_pad_mask(feature_lengths)
        
        # Encode
        encoder_out, encoder_out_lens = self.model.forward_encoder(features, padding_mask)
        
        # Decode
        if self.method == "greedy_search":
            hyp_tokens = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
        else:
            hyp_tokens = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=self.beam_size,
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        hypothesis = self.sp.decode(hyp_tokens[0])
        
        # Full normalization matching decode.sh pipeline
        hypothesis = full_normalize(hypothesis)
        
        return hypothesis, latency, duration


def warmup(asr: ASRInference, samples: List[Tuple[str, str]], num_warmup: int = 5):
    """Run warmup inference to stabilize GPU."""
    logging.info(f"Running {num_warmup} warmup iterations...")
    for i, (audio_path, _) in enumerate(samples[:num_warmup]):
        _ = asr.transcribe_with_timing(audio_path)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    logging.info("Warmup complete")


def run_analysis(
    asr: ASRInference,
    samples: List[Tuple[str, str]],
    output_dir: Path,
    test_name: str,
) -> Dict[str, BinStats]:
    """Run inference and collect statistics by duration bin."""
    
    # Initialize bins
    bins: Dict[str, BinStats] = {}
    for _, _, name in DURATION_BINS:
        bins[name] = BinStats(bin_name=name)
    
    total = len(samples)
    
    for idx, (audio_path, reference) in enumerate(samples):
        try:
            hypothesis, latency, duration = asr.transcribe_with_timing(audio_path)
            
            bin_name = get_duration_bin(duration)
            rtf = latency / duration if duration > 0 else 0
            errors, words = calculate_wer(reference, hypothesis)
            
            result = SampleResult(
                audio_path=audio_path,
                duration=duration,
                duration_bin=bin_name,
                reference=reference,
                hypothesis=hypothesis,
                latency_seconds=latency,
                rtf=rtf,
                wer=(errors / words * 100) if words > 0 else 0,
            )
            
            # Update bin stats
            bins[bin_name].count += 1
            bins[bin_name].total_duration += duration
            bins[bin_name].total_latency += latency
            bins[bin_name].total_words += words
            bins[bin_name].total_errors += errors
            bins[bin_name].samples.append(result)
            
            if (idx + 1) % 50 == 0:
                logging.info(f"Processed {idx + 1}/{total} samples")
                
        except Exception as e:
            logging.warning(f"Error processing {audio_path}: {e}")
    
    return bins


def print_summary(bins: Dict[str, BinStats], test_name: str):
    """Print summary table."""
    print("\n" + "=" * 90)
    print(f"ANALYSIS RESULTS: {test_name}")
    print("=" * 90)
    print(f"{'Duration Bin':<12} | {'Count':>8} | {'Avg Dur':>8} | {'Avg Lat':>10} | {'RTF':>8} | {'WER (%)':>8}")
    print("-" * 90)
    
    total_count = 0
    total_duration = 0
    total_latency = 0
    total_words = 0
    total_errors = 0
    
    for _, _, name in DURATION_BINS:
        stats = bins[name]
        if stats.count > 0:
            print(f"{stats.bin_name:<12} | {stats.count:>8} | {stats.avg_duration:>7.1f}s | {stats.avg_latency:>9.3f}s | {stats.rtf:>8.4f} | {stats.wer:>8.2f}")
            total_count += stats.count
            total_duration += stats.total_duration
            total_latency += stats.total_latency
            total_words += stats.total_words
            total_errors += stats.total_errors
    
    print("-" * 90)
    overall_rtf = total_latency / total_duration if total_duration > 0 else 0
    overall_wer = (total_errors / total_words * 100) if total_words > 0 else 0
    print(f"{'TOTAL':<12} | {total_count:>8} | {total_duration/total_count if total_count else 0:>7.1f}s | {total_latency/total_count if total_count else 0:>9.3f}s | {overall_rtf:>8.4f} | {overall_wer:>8.2f}")
    print("=" * 90)


def save_results(bins: Dict[str, BinStats], output_dir: Path, test_name: str):
    """Save detailed results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary JSON
    summary = {
        "test_name": test_name,
        "bins": {}
    }
    
    for name, stats in bins.items():
        summary["bins"][name] = {
            "count": stats.count,
            "total_duration": stats.total_duration,
            "total_latency": stats.total_latency,
            "avg_duration": stats.avg_duration,
            "avg_latency": stats.avg_latency,
            "rtf": stats.rtf,
            "wer": stats.wer,
        }
    
    with open(output_dir / f"{test_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # ============================================
    # STANDARDIZED CSV FORMAT FOR CROSS-REPO COMPARISON
    # Mỗi model (bất kể repo nào) export ra file này
    # Sau đó dùng compare_results.py để so sánh
    # ============================================
    csv_file = output_dir / f"{test_name}_results.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        # Header
        f.write("audio_path,duration_sec,duration_bin,reference,hypothesis,latency_sec,rtf\n")
        
        # Data
        for name, stats in bins.items():
            for sample in stats.samples:
                # Escape commas and quotes in text
                ref = sample.reference.replace('"', '""')
                hyp = sample.hypothesis.replace('"', '""')
                
                f.write(f'"{sample.audio_path}",{sample.duration:.3f},"{sample.duration_bin}",')
                f.write(f'"{ref}","{hyp}",{sample.latency_seconds:.6f},{sample.rtf:.6f}\n')
    
    logging.info(f"CSV results saved to: {csv_file}")
    
    # Detailed results TXT
    with open(output_dir / f"{test_name}_detailed.txt", "w") as f:
        for name, stats in bins.items():
            f.write(f"\n=== {name} ===\n")
            for sample in stats.samples:
                f.write(f"File: {sample.audio_path}\n")
                f.write(f"Duration: {sample.duration:.2f}s, Latency: {sample.latency_seconds:.4f}s, RTF: {sample.rtf:.4f}\n")
                f.write(f"REF: {sample.reference}\n")
                f.write(f"HYP: {sample.hypothesis}\n")
                f.write(f"WER: {sample.wer:.2f}%\n\n")
    
    logging.info(f"Results saved to {output_dir}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Analyze WER and Latency by audio duration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-dir", type=str, required=True, help="Test data directory")
    parser.add_argument("--exp-dir", type=str, default="/vietasr/data4/exp_finetune_v3")
    parser.add_argument("--bpe-model", type=str, default="/vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model")
    parser.add_argument("--epoch", type=int, default=19)
    parser.add_argument("--avg", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--method", type=str, default="modified_beam_search", choices=["modified_beam_search", "greedy_search"])
    parser.add_argument("--output-dir", type=str, default="/vietasr/results/duration_analysis")
    parser.add_argument("--num-warmup", type=int, default=5)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    # Get test name from directory
    test_name = Path(args.test_dir).name
    output_dir = Path(args.output_dir)
    
    logging.info(f"Analyzing: {test_name}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Method: {args.method}, Beam: {args.beam_size}")
    
    # Load test data
    samples = load_test_data(args.test_dir)
    if not samples:
        logging.error(f"No samples found in {args.test_dir}")
        return
    
    # Initialize ASR
    asr = ASRInference(
        exp_dir=args.exp_dir,
        bpe_model=args.bpe_model,
        epoch=args.epoch,
        avg=args.avg,
        device=args.device,
        beam_size=args.beam_size,
        method=args.method,
    )
    
    # Warmup
    warmup(asr, samples, args.num_warmup)
    
    # Run analysis
    bins = run_analysis(asr, samples, output_dir, test_name)
    
    # Print and save results
    print_summary(bins, test_name)
    save_results(bins, output_dir, test_name)


if __name__ == "__main__":
    main()
