#!/usr/bin/env python3
import torch
import sys

def extract_encoder(checkpoint_path, output_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Giữ chỉ encoder và encoder_embed
    encoder_state = {}
    for k, v in ckpt["model"].items():
        if k.startswith("encoder.") or k.startswith("encoder_embed."):
            encoder_state[k] = v
    
    # Tạo checkpoint mới
    new_ckpt = {
        "model": encoder_state,
        "epoch": ckpt.get("epoch", 0),
        "optimizer": None,
    }
    
    torch.save(new_ckpt, output_path)
    print(f"✅ Đã lưu encoder-only checkpoint: {output_path}")
    print(f"   - Số lượng keys: {len(encoder_state)}")
    print(f"   - Kích thước: {sum(v.numel() for v in encoder_state.values()):,} parameters")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_encoder.py <input_checkpoint.pt> <output_encoder.pt>")
        sys.exit(1)
    
    extract_encoder(sys.argv[1], sys.argv[2])