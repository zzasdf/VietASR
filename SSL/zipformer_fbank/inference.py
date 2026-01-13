#!/usr/bin/env python3
import argparse
import logging
import math
import torch
import torchaudio
import k2
import kaldifeat
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
from icefall.utils import str2bool, num_tokens, make_pad_mask
from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from finetune import add_model_arguments, get_params, get_model
from beam_search import modified_beam_search, greedy_search_batch
from typing import List
import os
from pathlib import Path

# Normalization logic from decode.py
DICT_MAP = {}
mappings = [
    ("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"),
    ("òe", "oè"), ("óe", "oé"), ("ỏe", "oẻ"), ("õe", "oẽ"), ("ọe", "oẹ"),
    ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ"),
]
for src, tgt in mappings:
    DICT_MAP[src] = tgt
    DICT_MAP[src.capitalize()] = tgt.capitalize()
    DICT_MAP[src.upper()] = tgt.upper()

def normalize_tone_and_typos(text):
    for k, v in DICT_MAP.items():
        text = text.replace(k, v)
    # Fix specific case: a lô -> alo
    text = text.replace("a lô", "alo")
    text = text.replace("A lô", "Alo")
    text = text.replace("A Lô", "Alo")
    return text

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint. If provided, --epoch and --avg are ignored.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        help="The epoch to decode.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average.",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        help="Path to the experiment directory.",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt.",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="Path to bpe.model.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="modified_beam_search",
        choices=["greedy_search", "modified_beam_search"],
        help="Decoding method.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
    )
    
    # parser.add_argument(
    #     "--sample-rate",
    #     type=int,
    #     default=16000,
    # )
    
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
    )

    add_model_arguments(parser)

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to decode",
    )

    return parser

def read_sound_files(filenames: List[str], expected_sample_rate: float) -> List[torch.Tensor]:
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        ans.append(wave[0].contiguous())
    return ans

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))
    
    # Ensure params has necessary attributes for get_model
    if not hasattr(params, "num_classes"): params.num_classes = [504]
    
    if params.bpe_model:
        sp = spm.SentencePieceProcessor()
        sp.load(params.bpe_model)
        params.blank_id = sp.piece_to_id("<blk>")
        params.unk_id = sp.piece_to_id("<unk>")
        params.vocab_size = sp.get_piece_size()
    else:
        token_table = k2.SymbolTable.from_file(params.tokens)
        params.blank_id = token_table["<blk>"]
        params.unk_id = token_table["<unk>"]
        params.vocab_size = num_tokens(token_table) + 1 # Assuming this matches checkpoint 2000

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s")
    logging.info(f"{params}")

    device = torch.device("cpu")
    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_model(params)
    
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        if not args.epoch or not args.exp_dir:
            raise ValueError("Must provide either --checkpoint or (--epoch and --exp-dir)")
        
        args.exp_dir = Path(args.exp_dir)
        if args.avg == 1:
            load_checkpoint(f"{args.exp_dir}/epoch-{args.epoch}.pt", model)
        else:
            start = args.epoch - args.avg + 1
            filenames = []
            for i in range(start, args.epoch + 1):
                if i >= 1:
                    filenames.append(f"{args.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    
    model.to(device)
    model.eval()

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    # opts.mel_opts.high_freq = -400 # Commented out

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(params.sound_files, params.sample_rate)
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]
    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, device=device)

    logging.info(f"Features shape: {features.shape}, Mean: {features.mean()}, Std: {features.std()}")
    
    from icefall.utils import make_pad_mask
    padding_mask = make_pad_mask(feature_lengths)
    
    # Forward encoder to check output
    # SSL model expects padding_mask, not feature_lengths
    encoder_out, encoder_out_lens = model.forward_encoder(features, padding_mask)
    logging.info(f"Encoder out shape: {encoder_out.shape}, Mean: {encoder_out.mean()}, Std: {encoder_out.std()}")

    hyps = []
    if params.method == "greedy_search":
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
    elif params.method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
    
    for hyp in hyp_tokens:
        if params.bpe_model:
            text = sp.decode(hyp)
            text = normalize_tone_and_typos(text)
        else:
            text = ""
            for i in hyp:
                text += token_table[i]
            text = text.replace(" ", " ").strip()
        hyps.append(text)

    for filename, hyp in zip(params.sound_files, hyps):
        logging.info(f"\n{filename}:\n{hyp}\n")

    logging.info("Decoding Done")

if __name__ == "__main__":
    main()
