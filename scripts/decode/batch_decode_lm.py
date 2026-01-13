#!/usr/bin/env python3
"""
Batch Decode with Word-level KenLM (In-process Rescoring)

Features:
1. Loads KenLM model ONCE (saving 20+ mins of reloading per test set).
2. Uses Fast Beam Search to generate N-best hypotheses.
3. Rescores hypotheses using KenLM scores.
4. Supports processing multiple test sets in one run.

Usage:
    python scripts/decode/batch_decode_lm.py \
        --epoch 19 --avg 17 --gpu 0 \
        --test-sets call_center_private \
        --word-lm-path /path/to/lm.binary
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from lhotse import CutSet, load_manifest_lazy

# Add project paths
sys.path.append("SSL/zipformer_fbank")
sys.path.append("ASR/zipformer")
sys.path.append(".")

from decode import (
    decode_one_batch,
    get_params,
    add_model_arguments,
)
from beam_search import fast_beam_search_nbest, fast_beam_search
from icefall.decode import Nbest, get_texts
from asr_datamodule import FinetuneAsrDataModule
from finetune import get_model
from icefall.checkpoint import (
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from icefall.utils import (
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic args
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--avg", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=0)
    
    # Data args
    parser.add_argument(
        "--test-sets",
        type=str,
        default="tongdai_clean",
        help="Comma-separated list of test sets"
    )
    parser.add_argument("--manifest-dir", type=str, default="fbank/tongdai_hard")
    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_500/bpe.model")

    # KenLM args
    parser.add_argument(
        "--word-lm-path",
        type=str,
        required=True,
        help="Path to KenLM binary"
    )
    parser.add_argument(
        "--word-lm-scale",
        type=float,
        default=0.3,
        help="Scale for LM score (0.0 - 1.0)"
    )
    
    # Model config (context_size=2 default for bigram decoder)
    parser.add_argument("--context-size", type=int, default=2)

    # Model args
    parser.add_argument("--exp-dir", type=str, default="zipformer/exp")
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)
    
    # Decoding args
    parser.add_argument("--decoding-method", type=str, default="fast_beam_search_nbest")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--beam", type=float, default=20.0)
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--max-duration", type=int, default=600)
    parser.add_argument("--num-paths", type=int, default=100)
    parser.add_argument("--nbest-scale", type=float, default=0.5)
    
    # Other
    parser.add_argument("--norm", type=str2bool, default=False)
    
    add_model_arguments(parser)
    return parser


def load_kenlm(lm_path: str):
    logging.info(f"⏳ Loading KenLM from: {lm_path}...")
    try:
        import kenlm
        start = time.time()
        model = kenlm.Model(lm_path)
        end = time.time()
        logging.info(f"✅ KenLM loaded in {end - start:.2f}s")
        return model
    except ImportError:
        logging.error("❌ pip install https://github.com/kpu/kenlm/archive/master.zip")
        sys.exit(1)


def decode_dataset_nbest(
    dl: torch.utils.data.DataLoader,
    params,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_lm,
    word_lm_scale: float
) -> Tuple[List[str], List[str], List[str]]:
    """
    Decode dataset using N-best generation and KenLM rescoring.
    Returns: cut_ids, ref_texts, hyp_texts
    """
    results = []
    
    # Lists to store results for WER calculation
    cut_ids = []
    ref_texts = []
    hyp_texts = []
    
    try:
        num_batches = len(dl)
        logging.info(f"Decoding {num_batches} batches...")
    except TypeError:
        # DynamicBucketingSampler might not support len()
        num_batches = None
        logging.info("Decoding batches (count unknown)...")
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Create trivial graph for acoustic decoding
    # vocab_size - 1 because we exclude blank from the symbols in the graph (usually)
    decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    
    for batch_idx, batch in enumerate(dl):
        feature = batch["audio"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        supervisions = batch["supervisions"]
        texts = supervisions["text"]
        # Extract cut_ids. In lhotse finetune dataset, 'cut' or 'id' might be available.
        # usually supervisions is a dict with keys.
        # Fallback to simple iteration if 'cut' list is present
        batch_cut_ids = supervisions.get("cut", [str(i) for i in range(len(texts))]) 
        # If 'cut' is a list of Cut objects, we need to extract IDs.
        # BUT 'supervisions' from K2 AsrDataModule usually implicitly handles collation.
        # Let's check 'cut' key presence.
        if "cut" in supervisions:
             batch_cut_ids = [c.id for c in supervisions["cut"]]
        elif "id" in supervisions: # Sometimes it's directly ID
             batch_cut_ids = supervisions["id"]
        else:
             # Fallback: dummy IDs if not found, though rare
             batch_cut_ids = [f"cut_{batch_idx}_{i}" for i in range(len(texts))]

        cut_ids.extend(batch_cut_ids)
        
        # Forward Encoder and Get Lattice
        with torch.no_grad():
            encoder_out, encoder_out_lens = model.forward_encoder(feature, padding_mask)
            
            # 1. Get Lattice
            lattice = fast_beam_search(
                model=model,
                decoding_graph=decoding_graph,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=params.beam,
                max_states=params.max_states,
                max_contexts=params.max_contexts,
                allow_partial=True,
            )
            
            # Move lattice to CPU to avoid OOM
            lattice = lattice.to("cpu")
            
            # 2. Get N-best (tokens)
            nbest = Nbest.from_lattice(
                lattice=lattice,
                num_paths=params.num_paths,
                nbest_scale=params.nbest_scale,
            )
            
            # Intersect with lattice to retrieve acoustic scores
            nbest = nbest.intersect(lattice)
            
            # 3. Get Texts (tokens) and mapping
            hyp_tokens = get_texts(nbest.fsa)
            cut_indices = nbest.shape.row_ids(1).tolist()
            
            # Ensure we get flat scores matching hyp_tokens
            ac_scores = nbest.fsa.get_tot_scores(use_double_scores=True, log_semiring=True).tolist()
            
            assert len(ac_scores) == len(hyp_tokens), f"Scores mismatch: {len(ac_scores)} vs {len(hyp_tokens)}"

        # Process results
        batch_size = len(texts)
        all_hyps_text = sp.decode(hyp_tokens) # Convert all to text
        
        # Group by cut_index
        # cut_indices is like [0, 0, 0, 1, 1, ..., batch_size-1]
        hyps_per_cut = [[] for _ in range(batch_size)]
        scores_per_cut = [[] for _ in range(batch_size)]
        
        for i, hyp_text, ac_score in zip(cut_indices, all_hyps_text, ac_scores):
            hyps_per_cut[i].append(hyp_text)
            scores_per_cut[i].append(ac_score)
            
        for i in range(batch_size):
            ref_text = texts[i]
            # Normalize Ref
            ref_text = ref_text.lower()
            ref_texts.append(ref_text)
            
            candidates = hyps_per_cut[i]
            candidate_scores = scores_per_cut[i]
            
            best_text = ""
            best_score = -float('inf')
            
            # Rescore
            for rank, (text, ac_score) in enumerate(zip(candidates, candidate_scores)):
                # Normalize Hyp is important for WER, but for LM scoring we might want original?
                # Usually LM is case sensitive or not depending on model. 
                # Assuming KenLM model is trained on same casing (likely lower).
                # But let's lower() just in case to match our WER logic.
                text_norm = text.lower()
                
                if not text_norm.strip():
                    lm_score = -9999
                else:
                    try:
                         # KenLM score (log10)
                         lm_score = word_lm.score(text_norm, bos=True, eos=True)
                         # DO NOT normalize by length for Rescoring (should match AC Total)
                         # words = text_norm.split()
                         # n_words = len(words) if len(words) > 0 else 1
                         # lm_score = lm_score / n_words
                    except:
                         lm_score = -9999
                
                # Formula: AC_Score + Scale * LM_Score
                # Note: valid AC scores are negative log probs (e.g. -100)
                # valid LM scores are negative log10 probs (e.g. -20)
                # If AC scores are positive (Cost), we should verify. 
                # But assuming higher is better for now as k2 typically uses scores.
                # If AC=13 (positive), and LM=-20. Total = 13 + 0.5*-20 = 3.
                total_score = ac_score + (word_lm_scale * lm_score)
                
                if i == 0 and rank < 3 and batch_idx == 0:
                     logging.info(f"Sample: '{text_norm}' | AC: {ac_score:.2f} | LM: {lm_score:.2f} | Tot: {total_score:.2f}")
                
                if total_score > best_score:
                    best_score = total_score
                    best_text = text_norm
            
            hyp_texts.append(best_text)
        
        if batch_idx % 10 == 0:
            if num_batches:
                 logging.info(f"Processed batch {batch_idx}/{num_batches}")
            else:
                 logging.info(f"Processed batch {batch_idx}")

    return cut_ids, ref_texts, hyp_texts


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    setup_logger(f"{args.exp_dir}/log/decode_lm_ngram")
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda", 0)
    
    # 1. Load Model
    params = get_params()
    params.update(vars(args))
    
    logging.info(f"Decoding method: {params.decoding_method}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    
    # Add required params for get_model
    params.vocab_size = sp.get_piece_size()
    params.blank_id = 0
    # context_size is often not in model args, added manually
    if not hasattr(params, "context_size"):
        params.context_size = 2
    
    logging.info("⏳ Loading ASR Model...")
    model = get_model(params)
    
    if params.epoch > 0:
        if params.use_averaged_model:
            # Load averaged model
            start = params.epoch - params.avg
            if start < 1: start = 1
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            
            logging.info(f"Averaging epochs {start} to {params.epoch}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
             load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    
    model.to(device)
    model.eval()
    
    # 2. Load KenLM
    word_lm = load_kenlm(params.word_lm_path)
    
    # 3. Decode
    test_sets = args.test_sets.split(",")
    
    for test_set in test_sets:
        logging.info(f"Processing: {test_set}")
        
        # Resolve manifest path
        potential_paths = [
            f"{args.manifest_dir}/cuts_{test_set}.jsonl.gz",
            f"{args.manifest_dir}/vietASR_cuts_{test_set}.jsonl.gz",
            f"data/fbank/{test_set}/cuts_{test_set}.jsonl.gz",
            f"data/fbank/{test_set}/vietASR_cuts_{test_set}.jsonl.gz"
        ]
        
        cuts_path = None
        for p in potential_paths:
            if os.path.exists(p):
                cuts_path = p
                break
        
        if not cuts_path:
            logging.error(f"❌ Manifest not found. Checked: {potential_paths}")
            continue

        logging.info(f"Loading cuts: {cuts_path}")
        try:
            cuts = load_manifest_lazy(cuts_path)
        except:
             cuts = CutSet.from_file(cuts_path)

        datamodule = FinetuneAsrDataModule(args)
        dl = datamodule.valid_dataloaders(cuts)
        
        # Run Decoding
        # NOW returns 3 values
        cut_ids, ref_texts, hyp_texts = decode_dataset_nbest(
            dl, params, model, sp, word_lm, args.word_lm_scale
        )
        
        # Save results
        # store_transcripts expects iterable of (cut_id, ref, hyp)
        results = list(zip(cut_ids, ref_texts, hyp_texts))
        
        store_transcripts(
            filename=f"{params.exp_dir}/recogs-{test_set}-lm.txt",
            texts=results,
        )
        
        # Simple WER calculation
        if len(ref_texts) > 0:
            import jiwer
            wer = jiwer.wer(ref_texts, hyp_texts)
            logging.info(f"📊 WER for {test_set}: {wer*100:.2f}%")
            
            with open(f"{params.exp_dir}/wer-{test_set}-lm.txt", "w") as f:
                f.write(f"WER: {wer*100:.2f}%\n")
                for r, h in zip(ref_texts, hyp_texts):
                    f.write(f"REF: {r}\nHYP: {h}\n\n")
        
    logging.info("✅ Done.")

if __name__ == "__main__":
    main()
