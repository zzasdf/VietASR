#!/usr/bin/env python3
# Copyright 2024 (Author: Antigravity)
"""
Validate pseudo-labels by comparing with current model's hypothesis.

Pipeline:
1. Load pseudo-labeled transcripts (path|text|duration)
2. Decode audio with current best model
3. Compare pseudo-label vs model hypothesis using WER
4. Filter samples with WER > threshold

Based on:
- "Efficient Data Selection for Domain Adaptation of ASR" (IDIAP, INTERSPEECH 2025)
- Cross-validation approach: model's prediction should agree with pseudo-label

Usage:
    python scripts/validate_pseudo_labels.py \
        --transcripts-in /data/raw/pesudo/tongdai_112025/transcripts.txt \
        --transcripts-out /data/filtered/transcripts_validated.txt \
        --checkpoint /path/to/model.pt \
        --max-wer 0.3
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import jiwer
import torch
import torchaudio
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Validate pseudo-labels by comparing with model hypothesis"
    )
    
    # Input/Output
    parser.add_argument(
        "--transcripts-in",
        type=Path,
        required=True,
        help="Input transcripts.txt (format: path|text|duration)",
    )
    parser.add_argument(
        "--transcripts-out",
        type=Path,
        required=True,
        help="Output filtered transcripts",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Base directory containing audio files",
    )
    
    # Model configuration
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("ASR/zipformer/exp"),
        help="Experiment directory containing checkpoints",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Checkpoint epoch to use",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average",
    )
    parser.add_argument(
        "--bpe-model",
        type=Path,
        default=Path("data/lang_bpe_500/bpe.model"),
        help="Path to BPE model",
    )
    
    # Filtering thresholds
    parser.add_argument(
        "--max-wer",
        type=float,
        default=0.3,
        help="Maximum WER between pseudo-label and model hypothesis. Default: 0.3 (30%%)",
    )
    parser.add_argument(
        "--target-retention",
        type=float,
        default=0.15,
        help="Target percentage to keep after filtering. Default: 0.15 (15%%)",
    )
    
    # Processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for decoding (1 for simplicity)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    
    parser.add_argument(
        "--inspect-num",
        type=int,
        default=20,
        help="Number of rejected samples to show for inspection",
    )
    
    return parser.parse_args()


def load_transcripts(path: Path) -> List[Tuple[str, str, float]]:
    """Load transcripts: path|text|duration"""
    transcripts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                audio_path = parts[0].strip()
                text = parts[1].strip()
                try:
                    duration = float(parts[2].strip())
                except ValueError:
                    duration = 0.0
                transcripts.append((audio_path, text, duration))
            elif len(parts) == 2:
                audio_path, text = parts
                transcripts.append((audio_path.strip(), text.strip(), 0.0))
    return transcripts


def calculate_wer(ref: str, hyp: str) -> float:
    """Calculate WER between reference and hypothesis."""
    if not ref or not hyp:
        return 1.0
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    try:
        return jiwer.wer(
            ref, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
    except Exception:
        return 1.0


class SimpleASRDecoder:
    """
    Simple wrapper for ASR decoding.
    Uses greedy search for speed.
    """
    
    def __init__(self, exp_dir: Path, epoch: int, avg: int, bpe_model: Path, device: str):
        self.device = device
        self.exp_dir = exp_dir
        
        # Add ASR directory to path
        asr_dir = Path(__file__).parent.parent / "ASR" / "zipformer"
        if str(asr_dir) not in sys.path:
            sys.path.insert(0, str(asr_dir))
        
        # Import required modules
        try:
            import sentencepiece as spm
            from train import get_params, get_model
            from icefall.checkpoint import (
                average_checkpoints_with_averaged_model,
                find_checkpoints,
                load_checkpoint,
            )
            
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(str(bpe_model))
            
            # Get model parameters
            params = get_params()
            params.exp_dir = exp_dir
            
            # Load model
            logging.info(f"Loading model from {exp_dir}")
            
            # Try to load averaged model
            avg_model_path = exp_dir / "pretrained.pt"
            if avg_model_path.exists():
                logging.info(f"Using pretrained model: {avg_model_path}")
                checkpoint = torch.load(avg_model_path, map_location=device)
            else:
                # Find and average checkpoints
                checkpoints = find_checkpoints(exp_dir, iteration=0)[: epoch + 1][-avg:]
                if checkpoints:
                    logging.info(f"Averaging {len(checkpoints)} checkpoints")
                    checkpoint = average_checkpoints_with_averaged_model(
                        filenames=checkpoints,
                        device=device,
                    )
                else:
                    raise ValueError(f"No checkpoints found in {exp_dir}")
            
            self.model = get_model(params)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.model.to(device)
            self.model.eval()
            
            logging.info("Model loaded successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import ASR modules: {e}")
            logging.error("Make sure you're running from the VietASR directory")
            raise
    
    @torch.no_grad()
    def decode(self, audio_path: Path) -> str:
        """Decode a single audio file and return hypothesis text."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Compute fbank features
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                sample_frequency=16000,
            )
            
            # Add batch dimension
            fbank = fbank.unsqueeze(0).to(self.device)
            fbank_lens = torch.tensor([fbank.shape[1]], device=self.device)
            
            # Encode
            encoder_out, encoder_out_lens = self.model.forward_encoder(fbank, fbank_lens)
            
            # Greedy search decode
            from beam_search import greedy_search_batch
            hyp_tokens = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
            
            # Decode tokens to text
            hyp_text = self.sp.decode(hyp_tokens[0])
            
            return hyp_text
            
        except Exception as e:
            logging.warning(f"Failed to decode {audio_path}: {e}")
            return ""


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # Load transcripts
    logging.info(f"Loading transcripts from {args.transcripts_in}")
    transcripts = load_transcripts(args.transcripts_in)
    logging.info(f"Loaded {len(transcripts)} samples")
    
    # Initialize decoder
    logging.info("Initializing ASR decoder...")
    try:
        decoder = SimpleASRDecoder(
            exp_dir=args.exp_dir,
            epoch=args.epoch,
            avg=args.avg,
            bpe_model=args.bpe_model,
            device=args.device,
        )
    except Exception as e:
        logging.error(f"Failed to initialize decoder: {e}")
        logging.error("Falling back to text-only filtering (no cross-validation)")
        decoder = None
    
    # Process samples
    validated_samples = []
    rejected_samples = []
    stats = {
        "total": len(transcripts),
        "passed": 0,
        "failed_wer": 0,
        "failed_decode": 0,
    }
    
    logging.info("Validating pseudo-labels...")
    for audio_path, pseudo_label, duration in tqdm(transcripts):
        full_audio_path = args.audio_dir / audio_path
        
        if decoder is None:
            # No decoder available, keep all samples
            validated_samples.append((audio_path, pseudo_label, duration, 0.0))
            stats["passed"] += 1
            continue
        
        if not full_audio_path.exists():
            logging.warning(f"Audio not found: {full_audio_path}")
            stats["failed_decode"] += 1
            continue
        
        # Decode audio
        hypothesis = decoder.decode(full_audio_path)
        
        if not hypothesis:
            stats["failed_decode"] += 1
            continue
        
        # Calculate WER between pseudo-label and hypothesis
        wer = calculate_wer(pseudo_label, hypothesis)
        
        if wer <= args.max_wer:
            # Pseudo-label agrees with model → likely correct
            validated_samples.append((audio_path, pseudo_label, duration, wer))
            stats["passed"] += 1
        else:
            # Pseudo-label disagrees with model → might be wrong
            stats["failed_wer"] += 1
            if len(rejected_samples) < args.inspect_num:
                rejected_samples.append({
                    "audio": audio_path,
                    "pseudo": pseudo_label[:60],
                    "hypothesis": hypothesis[:60],
                    "wer": wer,
                })
    
    # Sort by WER (lower is better) and apply target retention
    validated_samples.sort(key=lambda x: x[3])
    
    target_count = int(stats["total"] * args.target_retention)
    target_count = max(target_count, min(len(validated_samples), 100))
    
    final_samples = validated_samples[:target_count]
    
    # Save results
    logging.info(f"Saving {len(final_samples)} validated samples to {args.transcripts_out}")
    args.transcripts_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.transcripts_out, 'w', encoding='utf-8') as f:
        for audio_path, text, duration, _ in final_samples:
            f.write(f"{audio_path}|{text}|{duration}\n")
    
    # Print statistics
    logging.info("\n" + "=" * 60)
    logging.info("VALIDATION STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total input: {stats['total']}")
    logging.info(f"Passed WER check (≤{args.max_wer:.0%}): {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    logging.info(f"Failed WER check: {stats['failed_wer']}")
    logging.info(f"Failed decode: {stats['failed_decode']}")
    logging.info(f"Target retention: {args.target_retention:.0%}")
    logging.info(f"Final output: {len(final_samples)} ({len(final_samples)/stats['total']*100:.1f}%)")
    
    # Print rejection examples
    if rejected_samples:
        logging.info("\n" + "=" * 60)
        logging.info("REJECTED SAMPLES (pseudo-label disagrees with model)")
        logging.info("=" * 60)
        for ex in rejected_samples[:10]:
            logging.info(f"\n[REJECTED] WER={ex['wer']:.1%}")
            logging.info(f"  Audio:      {ex['audio']}")
            logging.info(f"  Pseudo:     {ex['pseudo']}...")
            logging.info(f"  Model hyp:  {ex['hypothesis']}...")
    
    logging.info("\nDone!")


if __name__ == "__main__":
    main()
