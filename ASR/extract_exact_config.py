#!/usr/bin/env python3
"""
Extract exact architecture configuration from pretrained Zipformer model.

This script analyzes the pretrained model's state_dict to determine the
exact architecture parameters needed for fine-tuning.

Usage:
    python extract_exact_config.py models/zipformer-30m-rnnt/pretrained.pt
"""

import torch
import sys
from pathlib import Path


def extract_config(model_path):
    """Extract architecture configuration from pretrained model."""
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_path}")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']
    
    print(f"Total parameters in checkpoint: {len(state_dict)}")
    
    # Extract encoder dimensions for each stack
    encoder_dims = []
    feedforward_dims = []
    num_layers_per_stack = []
    
    print(f"\nDetecting encoder stacks...\n")
    
    for stack_idx in range(20):  # Try up to 20 stacks
        # encoder_dim from bypass_scale shape
        bypass_key = f'encoder.encoder.encoders.{stack_idx}.layers.0.bypass_scale'
        
        if bypass_key in state_dict:
            encoder_dim = state_dict[bypass_key].shape[0]
            encoder_dims.append(encoder_dim)
            
            print(f"Stack {stack_idx}:")
            print(f"  encoder_dim: {encoder_dim}")
            
            # Count layers in this stack
            layer_count = 0
            for layer_idx in range(50):  # Try up to 50 layers per stack
                layer_key = f'encoder.encoder.encoders.{stack_idx}.encoder.layers.{layer_idx}.feed_forward3.in_proj.weight'
                if layer_key in state_dict:
                    layer_count += 1
                    
                    # Get feedforward_dim from first layer
                    if layer_idx == 0:
                        ff_shape = state_dict[layer_key].shape
                        # For gated activation, the output dim is half of the weight rows
                        ff_dim = ff_shape[0] // 2
                        feedforward_dims.append(ff_dim)
                        print(f"  feedforward_dim: {ff_dim}")
                else:
                    break
            
            num_layers_per_stack.append(layer_count)
            print(f"  num_layers: {layer_count}")
            print()
        else:
            break
    
    num_stacks = len(encoder_dims)
    
    if num_stacks == 0:
        print("ERROR: No encoder stacks found!")
        print("\nSample keys in checkpoint:")
        for i, k in enumerate(list(state_dict.keys())[:20]):
            print(f"  {i+1}. {k}")
        return
    
    print(f"{'='*70}")
    print(f"DETECTED CONFIGURATION ({num_stacks} stacks)")
    print(f"{'='*70}\n")
    
    # Build command-line arguments
    print("Use these arguments for fine-tuning:\n")
    print(f'--encoder-dim "{",".join(map(str, encoder_dims))}" \\')
    print(f'--feedforward-dim "{",".join(map(str, feedforward_dims))}" \\')
    print(f'--num-encoder-layers "{",".join(map(str, num_layers_per_stack))}" \\')
    
    # Encoder unmasked dim (heuristic: 2/3 of encoder_dim, rounded to nearby value)
    unmasked_dims = []
    for enc_dim in encoder_dims:
        # Common patterns: 192->192, 256->192, 384->256, 512->256
        unmasked = int(enc_dim * 2 / 3)
        # Round to common values
        if unmasked > 200 and unmasked < 260:
            unmasked = 256
        elif unmasked > 180 and unmasked <= 200:
            unmasked = 192
        unmasked_dims.append(unmasked)
    
    print(f'--encoder-unmasked-dim "{",".join(map(str, unmasked_dims))}" \\')
    
    # Num heads heuristic
    num_heads_list = []
    for enc_dim in encoder_dims:
        if enc_dim <= 256:
            num_heads = 4
        elif enc_dim <= 384:
            num_heads = 4
        elif enc_dim <= 512:
            num_heads = 8
        else:
            num_heads = 8
        num_heads_list.append(num_heads)
    
    print(f'--num-heads "{",".join(map(str, num_heads_list))}" \\')
    
    # Suggest downsampling and cnn kernel based on num_stacks
    if num_stacks == 5:
        print(f'--downsampling-factor "1,2,4,4,2" \\')
        print(f'--cnn-module-kernel "31,31,15,15,31" \\')
    elif num_stacks == 6:
        print(f'--downsampling-factor "1,2,4,8,4,2" \\')
        print(f'--cnn-module-kernel "31,31,15,15,15,31" \\')
    else:
        # Generic pattern
        ds_factor = ["1"] + ["2"] * (num_stacks - 2) + ["1"]
        cnn_kernel = ["31"] * num_stacks
        print(f'--downsampling-factor "{",".join(ds_factor)}" \\')
        print(f'--cnn-module-kernel "{",".join(cnn_kernel)}" \\')
    
    # Decoder and Joiner dimensions
    decoder_key = 'decoder.embedding.weight'
    if decoder_key in state_dict:
        vocab_size, decoder_dim = state_dict[decoder_key].shape
        print(f'--decoder-dim {decoder_dim} \\')
        print(f'--vocab-size {vocab_size}  # (auto-detected from BPE) \\')
    
    joiner_key = 'joiner.decoder_proj.weight'
    if joiner_key in state_dict:
        joiner_dim = state_dict[joiner_key].shape[0]
        print(f'--joiner-dim {joiner_dim}')
    
    # Summary table
    print(f'\n{"="*70}')
    print("ARCHITECTURE SUMMARY")
    print(f'{"="*70}')
    print(f"{'Stack':<8} {'encoder_dim':<12} {'ff_dim':<12} {'layers':<8} {'heads':<8}")
    print(f"{'-'*70}")
    for i in range(num_stacks):
        enc = encoder_dims[i] if i < len(encoder_dims) else '?'
        ff = feedforward_dims[i] if i < len(feedforward_dims) else '?'
        layers = num_layers_per_stack[i] if i < len(num_layers_per_stack) else '?'
        heads = num_heads_list[i] if i < len(num_heads_list) else '?'
        print(f"{i:<8} {enc:<12} {ff:<12} {layers:<8} {heads:<8}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print(f'\n{"="*70}')
    print("VERIFICATION")
    print(f'{"="*70}')
    print("\nTo verify, check if encoder_embed.out.weight shape matches:")
    embed_key = 'encoder_embed.out.weight'
    if embed_key in state_dict:
        shape = state_dict[embed_key].shape
        expected_dim = encoder_dims[0] if encoder_dims else "?"
        print(f"  {embed_key}: {list(shape)}")
        print(f"  Expected output dim: {expected_dim} (should match shape[0])")
        if shape[0] == expected_dim:
            print(f"  ✅ MATCH!")
        else:
            print(f"  ❌ MISMATCH! Expected {expected_dim}, got {shape[0]}")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_exact_config.py <path_to_pretrained.pt>")
        print("\nExample:")
        print("  python extract_exact_config.py models/zipformer-30m-rnnt/pretrained.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: File not found: {model_path}")
        sys.exit(1)
    
    extract_config(model_path)
