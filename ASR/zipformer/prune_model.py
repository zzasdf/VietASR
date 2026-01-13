import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import argparse
import os
import sentencepiece as spm # Cần import cái này
from train import get_params, get_model, add_model_arguments

def prune_module(module, amount):
    """Hàm tiện ích để cắt tỉa Linear layers"""
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')

def prune_decoder_safe(model, amount=0.5):
    print(f"✂️ Pruning Decoder Self-Attention: {amount}")
    if hasattr(model, 'attention_decoder') and model.attention_decoder is not None:
        for layer in model.attention_decoder.decoder_layers:
            # Chỉ cắt self_attn, KHÔNG cắt src_attn hay feed_forward
            prune_module(layer.self_attn, amount)
    else:
        print("⚠️ Warning: Không tìm thấy Attention Decoder")

def prune_encoder_last_layers(model, amount=0.5, num_last_stacks=2):
    print(f"✂️ Pruning Last {num_last_stacks} Encoder Stacks: {amount}")
    encoders = model.encoder.encoders
    total_stacks = len(encoders)
    start_idx = total_stacks - num_last_stacks
    
    for i in range(start_idx, total_stacks):
        stack = encoders[i]
        print(f"   - Processing Encoder Stack {i}...")
        for layer in stack.layers:
            if hasattr(layer, 'self_attn1'):
                prune_module(layer.self_attn1, amount)
            elif hasattr(layer, 'self_attn'):
                prune_module(layer.self_attn, amount)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn file .pt gốc")
    parser.add_argument("--output", required=True, help="Đường dẫn FILE lưu model")
    parser.add_argument("--prune_decoder", type=float, default=0.5, help="Tỉ lệ cắt Decoder Attn")
    parser.add_argument("--prune_encoder", type=float, default=0.0, help="Tỉ lệ cắt Encoder cuối")
    
    # === FIX 1: Thêm tham số bpe-model thủ công ===
    parser.add_argument("--bpe-model", type=str, required=True, help="Path to BPE model")
    # ==============================================

    # Nạp các tham số kiến trúc model (encoder-dim, layers...)
    add_model_arguments(parser)

    args = parser.parse_args()

    # 1. Load Params
    params = get_params()
    params.update(vars(args))
    
    # === FIX 2: Tính vocab_size từ BPE model ===
    # Model cần biết vocab_size để khởi tạo lớp Linear cuối cùng
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.vocab_size = sp.get_piece_size()
    # Các token đặc biệt (khớp với train.py)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    # ===========================================

    # Cấu hình cứng cho khớp với bài toán CTC/AED
    params.use_transducer = False
    params.use_ctc = True
    params.use_attention_decoder = True
    
    # Khởi tạo model
    print("🏗️ Đang khởi tạo model...")
    model = get_model(params)
    
    # 2. Load Checkpoint
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    # 3. Thực hiện Pruning
    if args.prune_decoder > 0:
        prune_decoder_safe(model, amount=args.prune_decoder)
        
    if args.prune_encoder > 0:
        prune_encoder_last_layers(model, amount=args.prune_encoder)

    # 4. Lưu Model mới
    if os.path.isdir(args.output):
        args.output = os.path.join(args.output, "pruned_model.pt")
        
    print(f"💾 Saving pruned model to: {args.output}")
    torch.save({'model': model.state_dict()}, args.output)
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    main()