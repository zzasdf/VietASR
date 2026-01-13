
#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import torch
import torchaudio
import kaldifeat
import sentencepiece as spm
import math

# Add SSL/zipformer_fbank to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(root_dir, "SSL/zipformer_fbank"))

from icefall.utils import make_pad_mask
from finetune import get_params, get_model, add_model_arguments
import k2

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--bpe-model", type=str, required=True, help="Path to BPE model")
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file")
    parser.add_argument("--chunk-frames", type=int, default=32, help="Chunk size in frames (e.g. 16, 32, 64)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--context-size", type=int, default=2, help="Decoder context size")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "beam_search"], help="Decoding method")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for beam search")
    
    add_model_arguments(parser)
    return parser

class Hypothesis:
    def __init__(self, decoder_input, score=0.0, text=[]):
        self.decoder_input = decoder_input # Tensor (1, context_size)
        self.score = score
        self.text = text # List of token IDs
        self.decoder_out = None # Cache decoder output
        self.lm = None # Cache projected decoder output

def greedy_search_step(model, params, encoder_out, decoder_input, debug=False):
    """
    encoder_out: (N, T, C) = (1, T_chunk, enc_dim)
    decoder_input: (N, context_size) = (1, 2)
    """
    joiner = model.joiner
    decoder = model.decoder
    blank_id = params.blank_id
    device = encoder_out.device
    
    # Project encoder output first
    # encoder_out: (1, T, enc_dim) -> am: (1, T, joiner_dim)
    am = joiner.encoder_proj(encoder_out)
    
    if debug:
        logging.info(f"  encoder_out shape: {encoder_out.shape}, am shape: {am.shape}")
    
    hyps = [] # New tokens found in this chunk
    
    for t in range(encoder_out.size(1)):
        # current_am: (1, 1, joiner_dim)
        current_am = am[:, t:t+1, :]
        
        # decoder_out: (1, 1, decoder_dim)
        decoder_out = decoder(decoder_input, need_pad=False)
        # lm: (1, 1, joiner_dim)
        lm = joiner.decoder_proj(decoder_out)
        
        if debug and t == 0:
            logging.info(f"  decoder_input: {decoder_input}, decoder_out shape: {decoder_out.shape}, lm shape: {lm.shape}")
        
        # Joiner expects 4D: (N, T, s_range, C)
        # So we need to unsqueeze: current_am (1, 1, C) -> (1, 1, 1, C)
        #                          lm (1, 1, C) -> (1, 1, 1, C)
        logits = joiner(
            current_am.unsqueeze(2),  # (1, 1, 1, joiner_dim)
            lm.unsqueeze(1),          # (1, 1, 1, joiner_dim)
            project_input=False
        )  # (1, 1, 1, vocab_size)
        
        if debug and t == 0:
            logging.info(f"  logits shape: {logits.shape}")
            probs = torch.softmax(logits.squeeze(), dim=-1)
            top5 = probs.topk(5)
            logging.info(f"  Top 5 probs: {list(zip(top5.indices.tolist(), [f'{p:.3f}' for p in top5.values.tolist()]))}")
        
        # Get best token
        y = logits.argmax().item()  # Use argmax() without dim for 4D tensor
        
        if y != blank_id:
            hyps.append(y)
            logging.info(f"Frame {t}: Found token {y}")
            # Update decoder input: shift left and append new token
            new_input = decoder_input.clone()
            new_input[0, :-1] = decoder_input[0, 1:]
            new_input[0, -1] = y
            decoder_input = new_input
            
    return decoder_input, hyps

def modified_beam_search_step(model, params, encoder_out, beams, beam_size=5):
    """
    beams: List[Hypothesis]
    """
    joiner = model.joiner
    decoder = model.decoder
    blank_id = params.blank_id
    device = encoder_out.device
    
    am = joiner.encoder_proj(encoder_out) # (1, T, dim)
    
    for t in range(encoder_out.size(1)):
        current_am = am[:, t:t+1, :] # (1, 1, dim)
        
        A = beams # Current set of hypotheses
        B = [] # Next set of hypotheses
        
        # 1. Process each hypothesis in A
        for hyp in A:
            # If we haven't cached decoder output for this hyp, compute it
            if hyp.decoder_out is None:
                hyp.decoder_out = decoder(hyp.decoder_input, need_pad=False)
                hyp.lm = joiner.decoder_proj(hyp.decoder_out)
            
            # Compute log_probs
            logits = joiner(current_am, hyp.lm, project_input=False) # (1, 1, 1, vocab)
            log_probs = torch.log_softmax(logits.squeeze(), dim=-1) # (vocab,)
            
            # 2. Top-k candidates
            # We need standard beam search logic:
            # - Always keep Blank (stay in same node)
            # - Expand to other tokens
            
            # Optimization: Only pick top-k tokens + blank
            topk_log_probs, topk_indices = log_probs.topk(beam_size)
            
            for k in range(beam_size):
                token = topk_indices[k].item()
                log_p = topk_log_probs[k].item()
                
                new_score = hyp.score + log_p
                
                if token == blank_id:
                    # Case 1: Blank -> Stay in parsing state, move to next frame
                    new_hyp = Hypothesis(
                        decoder_input=hyp.decoder_input,
                        score=new_score,
                        text=hyp.text[:]
                    )
                    # Reuse cache because decoder state didn't change
                    new_hyp.decoder_out = hyp.decoder_out
                    new_hyp.lm = hyp.lm
                    B.append(new_hyp)
                else:
                    # Case 2: Non-blank -> Update decoder state, stay in current frame? 
                    # Standard Transducer Beam Search usually allows multiple symbols per frame.
                    # For streaming simplicity (and matching greedy logic often used), 
                    # we can strictly follow "one symbol per frame" or standard "expand until blank".
                    # Here we implement "Modified Beam Search" (Graves): 
                    # restrict to at most one non-blank symbol per frame is a common approximation for speed.
                    # Or simpler: Just add to B and let the next frame handle subsequent tokens.
                    
                    # Update decoder input
                    new_input = hyp.decoder_input.clone()
                    new_input[0, :-1] = hyp.decoder_input[0, 1:]
                    new_input[0, -1] = token
                    
                    new_hyp = Hypothesis(
                        decoder_input=new_input,
                        score=new_score,
                        text=hyp.text + [token]
                    )
                    # Cache invalid, will recompute next time needed
                    new_hyp.decoder_out = None 
                    new_hyp.lm = None
                    B.append(new_hyp)

        # 3. Pruning B to beam_size
        # Sort by score descending
        B.sort(key=lambda x: x.score, reverse=True)
        beams = B[:beam_size]
        
    return beams

@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = get_parser()
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 1. Load BPE
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    # Try to find blank token id
    if sp.is_unknown(sp.piece_to_id("<blk>")):
         if sp.is_unknown(sp.piece_to_id("<blank>")):
             logging.warning("Warning: <blk> or <blank> token not found in BPE. Using ID 0 as blank.")
             blank_id = 0
         else:
             blank_id = sp.piece_to_id("<blank>")
    else:
        blank_id = sp.piece_to_id("<blk>")

    params = get_params()
    params.update(vars(args))
    params.vocab_size = sp.get_piece_size()
    params.blank_id = blank_id
    
    # Force streaming config
    params.causal = True
    params.use_transducer = True
    
    logging.info(f"Model Params: causal={params.causal}, chunk_size={args.chunk_frames}, final_downsample={params.final_downsample}")
    if not params.final_downsample:
        logging.warning("Warning: final_downsample is False (0). If your checkpoint has downsample=2, this will cause size mismatch.")
    
    # 2. Load Model
    logging.info(f"Loading model from {args.checkpoint}")
    model = get_model(params)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    # Get HubertModel (contains encoder_embed, layer_norm, and Zipformer2 encoder)
    hubert_model = model.encoder
    zipformer_encoder = hubert_model.encoder  # Zipformer2
    
    logging.info(f"use_layer_norm: {hubert_model.use_layer_norm}")
    
    # 3. Load Audio
    logging.info(f"Loading audio {args.wav}")
    wave, sample_rate = torchaudio.load(args.wav)
    # 4. Feature Extraction setup
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    fbank = kaldifeat.Fbank(opts)
    
    logging.info(f"Wave shape: {wave.shape}")
    # Force input to be list of 1D tensors to be safe with kaldifeat
    if wave.ndim == 2:
        wave = wave[0] # (T,)
    wave = wave.to(device) # Move to device after potential slicing
        
    features = fbank(wave) # (T, 80)
    # If fbank returns a list (because we might have failed batching?), handle it.
    # But usually fbank(tensor) returns tensor if single input? 
    # Actually kaldifeat.Fbank(tensor_1d) -> tensor_2d (frames, feat)
    
    logging.info(f"Features shape: {features.shape}")
    total_frames = features.size(0)
    
    # 5. Initialize States
    # Zipformer encoder states
    states = zipformer_encoder.get_init_states(batch_size=1, device=device)
    # Conv2dSubsampling (encoder_embed) cache for ConvNeXt
    embed_cache = hubert_model.encoder_embed.get_init_states(batch_size=1, device=device)
    logging.info(f"Initialized encoder states: {len(states)} tensors, embed_cache shape: {embed_cache.shape}")
    
    # Init Decoder Input (standard icefall format: [-1, ..., -1, blank_id])
    context_size = args.context_size
    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [params.blank_id],
        device=device,
        dtype=torch.int64
    ).reshape(1, context_size)
    logging.info(f"Initial decoder_input: {decoder_input}")
    
    # Decoding Loop
    chunk_size = args.chunk_frames
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    
    final_result_tokens = []
    
    # For Beam Search
    beams = [Hypothesis(decoder_input=decoder_input, score=0.0, text=[])]
    
    logging.info(f"Starting {args.method} decoding with chunk={chunk_size}, beam={args.beam_size if args.method == 'beam_search' else 1}")

    for i in range(num_chunks):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, total_frames)
        
        # Get chunk of fbank features: (chunk_len, 80)
        chunk_fbank = features[start_frame:end_frame, :]
        
        # Step 1: Add batch dimension -> (1, chunk_len, 80)
        chunk_fbank = chunk_fbank.unsqueeze(0)
        chunk_len_tensor = torch.tensor([chunk_fbank.size(1)], device=device)
        
        # Step 2: Pass through encoder_embed (Conv2dSubsampling) using STREAMING mode
        # streaming_forward expects cache and returns updated cache
        embedded_features, embedded_lens, embed_cache = hubert_model.encoder_embed.streaming_forward(
            chunk_fbank, chunk_len_tensor, embed_cache
        )
        # embedded_features: (B, T', D) = (1, T', 192) where D = encoder_dim[0] = 192
        # Note: T' = (T-7)//2 - 3 due to ConvNeXt padding requirements
        
        # Check if we got any frames after subsampling
        if embedded_features.size(1) == 0:
            logging.warning(f"Chunk {i}: No frames after subsampling, skipping")
            continue
        
        # Step 3: Apply layer_norm if exists (expects [..., D] where D=192)
        if hubert_model.layer_norm is not None:
            embedded_features = hubert_model.layer_norm(embedded_features)
        
        # Debug first chunk
        if i == 0:
            logging.info(f"Chunk fbank shape: {chunk_fbank.shape}")
            logging.info(f"After encoder_embed streaming: {embedded_features.shape}")
            logging.info(f"Features mean: {embedded_features.mean():.3f}, std: {embedded_features.std():.3f}")
        
        # Step 5: Prepare for Zipformer streaming_forward
        # Zipformer expects (T, N, C) so transpose from (B, T', C) to (T', B, C)
        x = embedded_features.transpose(0, 1)  # (T', 1, D)
        
        T_sub = x.size(0)  # subsampled chunk length
        x_lens = torch.tensor([T_sub], device=device)
        
        # Get left context for mask
        left_context = zipformer_encoder.left_context_frames[0]
        mask_len = left_context + T_sub
        padding_mask = torch.zeros((1, mask_len), dtype=torch.bool, device=device)
        
        # Step 6: Streaming Encoder Forward
        encoder_out, _, states = zipformer_encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=states,
            src_key_padding_mask=padding_mask
        )
        
        # encoder_out: (T_out, 1, D) -> transpose to (1, T_out, D)
        encoder_out = encoder_out.transpose(0, 1)
        
        if i == 0:
            logging.info(f"First chunk - encoder_out shape after transpose: {encoder_out.shape}")
        
        if args.method == "greedy":
            # Enable debug for first chunk
            decoder_input, new_tokens = greedy_search_step(model, params, encoder_out, decoder_input, debug=(i==0))
            if new_tokens:
                final_result_tokens.extend(new_tokens)
                print(f"Partial: {sp.decode(final_result_tokens)}")
        else:
            # Beam Search
            beams = modified_beam_search_step(model, params, encoder_out, beams, beam_size=args.beam_size)
            # Print best beam so far
            best_beam = beams[0]
            if i % 5 == 0:
                 print(f"Partial Best Beam: {sp.decode(best_beam.text)}")
                 
    # Final Result
    if args.method == "greedy":
        final_text = sp.decode(final_result_tokens)
    else:
        best_beam = beams[0]
        final_text = sp.decode(best_beam.text)
        
    print("\n" + "="*50)
    print(f"FINAL RESULT: {final_text}")
    print("="*50)

if __name__ == "__main__":
    main()
