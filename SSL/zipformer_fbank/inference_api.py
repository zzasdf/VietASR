#!/usr/bin/env python3
"""
ASR Inference API for Zipformer Transducer Model

This module provides a clean, easy-to-use API for speech-to-text inference
using the trained Zipformer ASR model.

Example usage:
    from inference_api import ASRInference, transcribe_audio
    
    # Option 1: Using the class
    asr = ASRInference(
        exp_dir="data4/exp_finetune_v3",
        bpe_model="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model",
        device="cuda",
        epoch=30,
        avg=5,
    )
    text = asr.transcribe("audio.wav")
    texts = asr.transcribe(["audio1.wav", "audio2.wav"])
    
    # Option 2: Simple function
    text = transcribe_audio("audio.wav")
"""
import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
import torchaudio
import kaldifeat
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from icefall.utils import make_pad_mask
from icefall.checkpoint import average_checkpoints, average_checkpoints_with_averaged_model, load_checkpoint
from finetune import add_model_arguments, get_params, get_model
from beam_search import modified_beam_search, greedy_search_batch

LOG_EPS = math.log(1e-10)

# Vietnamese text normalization
DICT_MAP = {}
_mappings = [
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
]
for src, tgt in _mappings:
    DICT_MAP[src] = tgt
    DICT_MAP[src.capitalize()] = tgt.capitalize()
    DICT_MAP[src.upper()] = tgt.upper()


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text with proper diacritics and common fixes."""
    for k, v in DICT_MAP.items():
        text = text.replace(k, v)
    # Common typo fixes
    text = text.replace("a lô", "alo")
    text = text.replace("A lô", "Alo")
    text = text.replace("A Lô", "Alo")
    return text


class ASRInference:
    """
    High-level ASR inference class for the Zipformer Transducer model.
    
    Features:
    - Load model from checkpoint (single or averaged)
    - Process audio files (single or batch)
    - Support GPU/CPU inference
    - Vietnamese text normalization
    
    Args:
        exp_dir: Path to experiment directory containing checkpoints
        bpe_model: Path to BPE model file
        device: Device for inference ("cuda", "cuda:0", "cpu")
        epoch: Epoch number of checkpoint to load
        avg: Number of checkpoints to average (from epoch-avg+1 to epoch)
        checkpoint: Direct path to checkpoint file (overrides epoch/avg)
        beam_size: Beam size for modified_beam_search
        method: Decoding method ("modified_beam_search" or "greedy_search")
    """
    
    def __init__(
        self,
        exp_dir: str = "data4/exp_finetune_v3",
        bpe_model: str = "viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model",
        device: str = "cuda",
        epoch: int = 30,
        avg: int = 5,
        checkpoint: Optional[str] = None,
        beam_size: int = 10,
        method: str = "modified_beam_search",
    ):
        self.device = self._get_device(device)
        self.beam_size = beam_size
        self.method = method
        self.sample_rate = 16000
        self.feature_dim = 80
        
        logging.info(f"Initializing ASRInference on device: {self.device}")
        
        # Load BPE model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        
        # Create model params
        self._init_params()
        
        # Build model
        self.model = self._build_model()
        
        # Load checkpoint
        self._load_checkpoint(exp_dir, epoch, avg, checkpoint)
        
        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Fbank extractor
        self.fbank = self._init_fbank()
        
        logging.info(f"ASRInference ready. Model params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_device(self, device: str) -> torch.device:
        """
        Get and validate the device for inference.
        
        Args:
            device: Device string like "cuda", "cuda:0", "cuda:1", "cpu"
            
        Returns:
            torch.device object
            
        Raises:
            ValueError: If specified GPU index is not available
        """
        if device == "cpu":
            return torch.device("cpu")
        
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        
        # Parse device string
        if device == "cuda":
            return torch.device("cuda")
        
        # Handle cuda:N format
        if device.startswith("cuda:"):
            try:
                gpu_index = int(device.split(":")[1])
                num_gpus = torch.cuda.device_count()
                
                if gpu_index >= num_gpus:
                    raise ValueError(
                        f"GPU {gpu_index} not available. "
                        f"Only {num_gpus} GPU(s) found: cuda:0 to cuda:{num_gpus-1}"
                    )
                
                logging.info(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
                return torch.device(device)
                
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid device format: {device}. Use 'cuda', 'cuda:0', 'cuda:1', etc.")
                raise
        
        raise ValueError(f"Unknown device: {device}. Use 'cuda', 'cuda:N', or 'cpu'")
    
    def _init_params(self):
        """Initialize model parameters."""
        # Get default params and update with necessary values
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args, _ = parser.parse_known_args([])
        
        self.params = get_params()
        self.params.update(vars(args))
        
        # Set vocab params from BPE model
        self.params.blank_id = self.sp.piece_to_id("<blk>")
        self.params.unk_id = self.sp.piece_to_id("<unk>")
        self.params.vocab_size = self.sp.get_piece_size()
        self.params.sample_rate = self.sample_rate
        self.params.feature_dim = self.feature_dim
        
        # Ensure required params (from training config)
        self.params.num_classes = [504]
        self.params.use_transducer = True
        self.params.use_ctc = False
        self.params.context_size = 2
        self.params.decoder_dim = 512
        self.params.joiner_dim = 512
        self.params.final_downsample = 1
        self.params.use_layer_norm = 0
    
    def _build_model(self):
        """Build the ASR model."""
        return get_model(self.params)
    
    def _load_checkpoint(
        self,
        exp_dir: str,
        epoch: int,
        avg: int,
        checkpoint: Optional[str],
    ):
        """Load model checkpoint."""
        if checkpoint:
            logging.info(f"Loading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu")
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        else:
            exp_dir = Path(exp_dir)
            if avg == 1:
                ckpt_path = exp_dir / f"epoch-{epoch}.pt"
                logging.info(f"Loading single checkpoint: {ckpt_path}")
                load_checkpoint(str(ckpt_path), self.model)
            else:
                # FAST METHOD: Use average_checkpoints_with_averaged_model
                # This only loads 2 files (start and end) instead of all N checkpoints
                # because icefall saves incremental averaged models during training
                start_epoch = epoch - avg
                if start_epoch < 1:
                    start_epoch = 1
                    
                filename_start = str(exp_dir / f"epoch-{start_epoch}.pt")
                filename_end = str(exp_dir / f"epoch-{epoch}.pt")
                
                logging.info(f"Loading averaged model (FAST): epoch {start_epoch} to {epoch}")
                logging.info(f"  Using 2-file method (much faster than loading {avg} files)")
                
                self.model.to(self.device)
                self.model.load_state_dict(
                    average_checkpoints_with_averaged_model(
                        filename_start=filename_start,
                        filename_end=filename_end,
                        device=self.device,
                    )
                )
                logging.info("Checkpoint loading complete!")
    
    def _init_fbank(self) -> kaldifeat.Fbank:
        """Initialize Fbank feature extractor."""
        opts = kaldifeat.FbankOptions()
        opts.device = self.device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = self.feature_dim
        return kaldifeat.Fbank(opts)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and return waveform tensor."""
        wave, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, sr, self.sample_rate)
        return wave[0].contiguous()  # Take first channel, ensure contiguous
    
    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor],
        normalize: bool = True,
    ) -> Union[str, List[str]]:
        """
        Transcribe audio file(s) to text.
        
        Args:
            audio: Path to audio file, list of paths, or waveform tensor (N, samples)
            normalize: Whether to apply Vietnamese text normalization
            
        Returns:
            Transcribed text (single string if single input, list if batch)
        """
        single_input = isinstance(audio, str)
        
        # Load audio files
        if isinstance(audio, str):
            audio = [audio]
        
        if isinstance(audio, list):
            waves = [self._load_audio(f).to(self.device) for f in audio]
        else:
            # Assume tensor input (N, samples)
            if audio.ndim == 1:
                waves = [audio.to(self.device)]
            else:
                waves = [audio[i].to(self.device) for i in range(audio.shape[0])]
        
        # Extract Fbank features
        features = self.fbank(waves)
        feature_lengths = torch.tensor([f.size(0) for f in features], device=self.device)
        
        # Pad features
        features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)
        
        # Create padding mask
        padding_mask = make_pad_mask(feature_lengths)
        
        # Forward encoder
        encoder_out, encoder_out_lens = self.model.forward_encoder(features, padding_mask)
        
        # Decode
        if self.method == "greedy_search":
            hyp_tokens = greedy_search_batch(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
        else:  # modified_beam_search
            hyp_tokens = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=self.beam_size,
            )
        
        # Convert tokens to text
        results = []
        for tokens in hyp_tokens:
            text = self.sp.decode(tokens)
            if normalize:
                text = normalize_vietnamese_text(text)
            results.append(text)
        
        return results[0] if single_input else results


def transcribe_audio(
    audio_path: str,
    exp_dir: str = "data4/exp_finetune_v3",
    bpe_model: str = "viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model",
    device: str = "cuda",
    epoch: int = 30,
    avg: int = 5,
    beam_size: int = 10,
) -> str:
    """
    Simple function to transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file (WAV, 16kHz recommended)
        exp_dir: Path to experiment directory
        bpe_model: Path to BPE model
        device: Device for inference ("cuda" or "cpu")
        epoch: Epoch number for checkpoint
        avg: Number of checkpoints to average
        beam_size: Beam size for decoding
        
    Returns:
        Transcribed text
    """
    asr = ASRInference(
        exp_dir=exp_dir,
        bpe_model=bpe_model,
        device=device,
        epoch=epoch,
        avg=avg,
        beam_size=beam_size,
    )
    return asr.transcribe(audio_path)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ASR Inference API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_files",
        type=str,
        nargs="+",
        help="Audio file(s) to transcribe",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="data4/exp_finetune_v3",
        help="Experiment directory",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model",
        help="Path to BPE model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', or 'cpu'",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Epoch number",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam size for modified_beam_search",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="modified_beam_search",
        choices=["modified_beam_search", "greedy_search"],
        help="Decoding method",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    
    # Create ASR instance
    asr = ASRInference(
        exp_dir=args.exp_dir,
        bpe_model=args.bpe_model,
        device=args.device,
        epoch=args.epoch,
        avg=args.avg,
        beam_size=args.beam_size,
        method=args.method,
    )
    
    # Transcribe
    results = asr.transcribe(args.audio_files)
    
    # Print results
    if isinstance(results, str):
        results = [results]
    
    for audio_file, text in zip(args.audio_files, results):
        print(f"\n{audio_file}:")
        print(f"  {text}")
