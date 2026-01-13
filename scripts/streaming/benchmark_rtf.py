#!/usr/bin/env python3
import torch
import time
import logging
import argparse
import sys
import os

# Add SSL/zipformer_fbank to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(root_dir, "SSL/zipformer_fbank"))

import torchaudio
import kaldifeat
from icefall.utils import make_pad_mask
from finetune import get_params, get_model, add_model_arguments
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--bpe-model", type=str, required=True, help="Path to BPE model")
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file for benchmark")
    parser.add_argument("--chunk-frames", type=str, default="16,32,64", help="List of chunk sizes (frames) to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup chunks")
    
    add_model_arguments(parser)
    return parser

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Load Params & Model
    params = get_params()
    params.update(vars(args))
    
    # Load BPE to get vocab size
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    params.vocab_size = sp.get_piece_size()
    params.blank_id = 0 
    
    # Force streaming config
    params.causal = True 
    params.use_transducer = True
    
    device = torch.device(args.device)
    print(f"Loading model on {device}...")
    
    model = get_model(params)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Handle state dict mismatch if necessary
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()
    
    # Load Audio
    print(f"Loading audio: {args.wav}")
    wave, sample_rate = torchaudio.load(args.wav)
    wave = wave.to(device)
    duration = wave.shape[1] / sample_rate
    print(f"Audio Duration: {duration:.2f}s")
    
    # Feature Extractor
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    fbank = kaldifeat.Fbank(opts)
    
    features = fbank(wave) # (T, 80)
    total_frames = features.size(0)
    print(f"Total Frames: {total_frames}")

    # Benchmark loop
    chunk_sizes = [int(x) for x in args.chunk_frames.split(",")]
    
    print("\n" + "="*80)
    print(f"{'Chunk (Frames)':<15} | {'Chunk (ms)':<12} | {'Process Time (s)':<18} | {'RTF':<10} | {'Status'}")
    print("-" * 80)
    
    # Access the Zipformer2 encoder directly
    # Structure: model.encoder.encoder is Zipformer2
    zipformer_encoder = model.encoder.encoder
    
    for chunk_size in chunk_sizes:
        # Check if model supports streaming forward
        if not hasattr(zipformer_encoder, "streaming_forward"):
            print(f"{chunk_size:<15} | {'N/A':<12} | {'N/A':<18} | {'N/A':<10} | {'Not Supported'}")
            continue

        # Initialize states
        # get_init_states(batch_size, device)
        states = zipformer_encoder.get_init_states(batch_size=1, device=device)
        
        start_time = time.time()
        
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        # Streaming Loop
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_frames)
            
            chunk_feat = features[start:end, :].unsqueeze(0) # (1, T_chunk, 80)
            chunk_len = torch.tensor([chunk_feat.size(1)], device=device)
            mask = make_pad_mask(chunk_len) # All False (no padding)
            
            # Stateful Forward
            # x, x_lens, states = streaming_forward(x, x_lens, states, mask)
            
            # We need to simulate padding mask correctly.
            # src_key_padding_mask is False for valid tokens.
            padding_mask = torch.zeros((1, chunk_feat.size(1)), dtype=torch.bool, device=device)
            
            try:
                encoder_out, encoder_out_lens, states = zipformer_encoder.streaming_forward(
                    x=chunk_feat,
                    x_lens=chunk_len,
                    states=states,
                    src_key_padding_mask=padding_mask
                )
            except Exception as e:
                print(f"Error in streaming_forward: {e}")
                break
            
        total_time = time.time() - start_time
        rtf = total_time / duration
        chunk_ms_real = chunk_size * 10 
        
        print(f"{chunk_size:<15} | {chunk_ms_real:<12} | {total_time:<18.4f} | {rtf:<10.4f} | {'OK'}")
        
    print("="*80)
    print("RTF < 1.0 means FASTER than real-time.")

if __name__ == "__main__":
    main()
