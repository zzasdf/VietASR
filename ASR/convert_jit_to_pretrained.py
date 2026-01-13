#!/usr/bin/env python3
"""
Script to extract state_dict from TorchScript model for fine-tuning.

The HuggingFace model hynt/Zipformer-30M-RNNT-6000h provides jit_script.pt
(TorchScript format), which is meant for inference. To fine-tune, we need
to extract the weights and save them in the training checkpoint format.

Usage:
    python convert_jit_to_pretrained.py \
        --jit-path models/zipformer-30m-rnnt/jit_script.pt \
        --output-path models/zipformer-30m-rnnt/pretrained.pt \
        --analyze-only
"""

import argparse
import torch
from collections import OrderedDict


def analyze_jit_model(jit_path: str):
    """Analyze the structure of a JIT model."""
    print(f"\n{'='*60}")
    print(f"Analyzing TorchScript model: {jit_path}")
    print(f"{'='*60}")
    
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    state_dict = jit_model.state_dict()
    
    print(f"\nTotal number of keys: {len(state_dict)}")
    
    # Group keys by prefix
    prefixes = {}
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)
    
    print(f"\nKey prefixes found:")
    for prefix, keys in sorted(prefixes.items()):
        print(f"  {prefix}: {len(keys)} keys")
    
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        shape = list(state_dict[key].shape)
        print(f"  {i+1:3d}. {key} -> {shape}")
    
    print(f"\nLast 10 keys:")
    for i, key in enumerate(list(state_dict.keys())[-10:]):
        shape = list(state_dict[key].shape)
        print(f"  {len(state_dict)-9+i:3d}. {key} -> {shape}")
    
    return state_dict


def convert_jit_to_pretrained(
    jit_path: str, 
    output_path: str,
    key_mapping: dict = None
):
    """
    Extract state_dict from JIT model and save in training format.
    
    Args:
        jit_path: Path to the TorchScript model
        output_path: Path to save the converted checkpoint
        key_mapping: Optional dict to rename keys (old_prefix -> new_prefix)
    """
    print(f"\nLoading TorchScript model from {jit_path}")
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    jit_state_dict = jit_model.state_dict()
    
    print(f"Found {len(jit_state_dict)} keys")
    
    # Apply key mapping if provided
    fixed_state_dict = OrderedDict()
    
    for key, value in jit_state_dict.items():
        new_key = key
        
        if key_mapping:
            for old_prefix, new_prefix in key_mapping.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
        
        fixed_state_dict[new_key] = value
    
    # Save in format expected by icefall's load_checkpoint
    # Format: {"model": state_dict, ...}
    checkpoint = {
        "model": fixed_state_dict,
    }
    
    torch.save(checkpoint, output_path)
    print(f"Saved pretrained checkpoint to {output_path}")
    print(f"Checkpoint format: {{\"model\": state_dict}} with {len(fixed_state_dict)} keys")


def verify_compatibility(jit_path: str, model_keys_file: str = None):
    """
    Verify if JIT state_dict keys are compatible with training model.
    
    If model_keys_file is provided, compare against that list.
    Otherwise, just print analysis.
    """
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    jit_keys = set(jit_model.state_dict().keys())
    
    if model_keys_file:
        with open(model_keys_file, 'r') as f:
            model_keys = set(line.strip() for line in f)
        
        missing = model_keys - jit_keys
        extra = jit_keys - model_keys
        
        print(f"\nCompatibility check:")
        print(f"  JIT keys: {len(jit_keys)}")
        print(f"  Model keys: {len(model_keys)}")
        print(f"  Missing from JIT: {len(missing)}")
        print(f"  Extra in JIT: {len(extra)}")
        
        if missing:
            print(f"\n  Sample missing keys:")
            for k in list(missing)[:5]:
                print(f"    - {k}")
        
        if extra:
            print(f"\n  Sample extra keys:")
            for k in list(extra)[:5]:
                print(f"    - {k}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TorchScript model to training checkpoint format"
    )
    parser.add_argument(
        "--jit-path",
        type=str,
        required=True,
        help="Path to the TorchScript model (jit_script.pt)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the converted checkpoint"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the model, don't convert"
    )
    parser.add_argument(
        "--fix-encoder-prefix",
        action="store_true",
        help="Fix common issue where encoder keys have 'encoder.encoder.' prefix"
    )
    
    args = parser.parse_args()
    
    # Always analyze first
    state_dict = analyze_jit_model(args.jit_path)
    
    if args.analyze_only:
        print("\n[Analyze only mode - not converting]")
        return
    
    if args.output_path is None:
        args.output_path = args.jit_path.replace('.pt', '_pretrained.pt')
        print(f"\nNo output path specified, using: {args.output_path}")
    
    # Define key mappings if needed
    key_mapping = None
    if args.fix_encoder_prefix:
        # Check if this fix is needed
        sample_keys = list(state_dict.keys())[:5]
        if any(k.startswith("encoder.encoder.") for k in sample_keys):
            key_mapping = {
                "encoder.encoder.": "encoder.",
            }
            print("\nApplying encoder prefix fix")
    
    convert_jit_to_pretrained(
        args.jit_path,
        args.output_path,
        key_mapping=key_mapping
    )
    
    print("\n" + "="*60)
    print("DONE! Next steps:")
    print("="*60)
    print(f"""
1. Test loading the converted checkpoint:
   
   cd /home/hiennt/VietASR/ASR
   python -c "
   import torch
   ckpt = torch.load('{args.output_path}', map_location='cpu')
   print('Keys in checkpoint:', list(ckpt.keys()))
   print('Number of model params:', len(ckpt['model']))
   "

2. Use with train.py for fine-tuning:
   
   python ./zipformer/train.py \\
     --pretrain-path {args.output_path} \\
     --pretrain-type ASR \\
     ... other training args
""")


if __name__ == "__main__":
    main()
