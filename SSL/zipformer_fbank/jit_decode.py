#!/usr/bin/env python3
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang, Zengwei Yao)
#
# See the LICENSE file in the root directory for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script này thực hiện giải mã bằng mô hình TorchScript (cpu_jit.pt hoặc tương tự).
Nó tích hợp logic load data từ lhotse manifest và chuẩn hóa văn bản giống decode.py.
"""

import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import các module từ dự án
from text_normalization import normalize_text, remove_filler_words
from asr_datamodule import FinetuneAsrDataModule
from icefall.utils import AttributeDict, setup_logger, store_transcripts, str2bool, write_error_stats
from finetune import get_params
from beam_search import modified_beam_search

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        required=True,
        help="Path to the torchscript model (ví dụ: cpu_jit.pt)",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="Thư mục experiment",
    )

    parser.add_argument(
        "--cuts-name",
        type=str,
        default="test",
        help="Tên tập cuts để giải mã (test, dev, hoặc custome name)",
    )

    parser.add_argument(
        "--norm",
        type=str2bool,
        default=False,
        help="Bật chuẩn hóa văn bản tiếng Việt khi tính WER",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Phương pháp giải mã. Hiện hỗ trợ:
          - greedy_search
          - modified_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size cho modified_beam_search",
    )

    return parser

def greedy_search_batch_jit(
    model: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
) -> List[List[int]]:
    """Greedy search cho mô hình JIT trong chế độ batch."""
    assert encoder_out.ndim == 3
    N = encoder_out.size(0)

    device = encoder_out.device
    # Giả định blank_id và context_size có sẵn trong decoder của JIT model
    # Nếu không có, có thể cần truyền vào qua argument
    try:
        blank_id = int(model.decoder.blank_id)
        context_size = int(model.decoder.context_size)
    except AttributeError:
        # Fallback nếu model cấu trúc khác (ví dụ từ jit_pretrained.py)
        blank_id = 0 # Default common blank
        context_size = 2 # Default zipformer context size

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    
    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(
        decoder_input,
        need_pad=torch.tensor([False], device=device),
    ).squeeze(1)

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = packed_encoder_out.data[start:end]
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
        )

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True
        
        if emitted:
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input,
                need_pad=torch.tensor([False], device=device),
            ).squeeze(1)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans

@torch.no_grad()
def decode_one_batch(
    model: torch.jit.ScriptModule,
    batch: dict,
    sp: spm.SentencePieceProcessor,
    decoding_method: str = "greedy_search",
    beam_size: int = 4,
) -> Dict[str, List[List[str]]]:
    device = next(model.parameters()).device
    feature = batch["audio"].to(device)
    padding_mask = batch["padding_mask"].to(device)

    # Note: JIT model encoder might take features and padding_mask or feature_lens
    # based on export logic. In jit_pretrained.py it was features and feature_lengths.
    # In decode.py (finetune) it was feature and padding_mask.
    # We try to match what export.py usually does.
    
    # Calculate feature_lens from padding_mask if needed
    feature_lens = (~padding_mask).sum(dim=1)
    
    encoder_out, encoder_out_lens = model.encoder(
        features=feature,
        feature_lengths=feature_lens,
    )

    if decoding_method == "greedy_search":
        hyp_tokens = greedy_search_batch_jit(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
    elif decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=beam_size,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {decoding_method}")
    
    hyps = []
    for hyp in sp.decode(hyp_tokens):
        hyps.append(hyp.split())
        
    return {decoding_method: hyps}

def decode_dataset(
    dl: DataLoader,
    model: torch.jit.ScriptModule,
    sp: spm.SentencePieceProcessor,
    norm: bool,
    decoding_method: str,
    beam_size: int,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    results = defaultdict(list)
    num_cuts = 0
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["cuts"]]

        hyps_dict = decode_one_batch(
            model=model,
            batch=batch,
            sp=sp,
            decoding_method=decoding_method,
            beam_size=beam_size,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_text_norm = normalize_text(ref_text, use_extended=norm)
                ref_words = ref_text_norm.split()
                
                hyp_text = " ".join(hyp_words)
                hyp_text_norm = normalize_text(hyp_text, use_extended=norm)
                hyp_words_norm = hyp_text_norm.split()

                ref_words_filtered = remove_filler_words(ref_words)
                hyp_words_filtered = remove_filler_words(hyp_words_norm)

                this_batch.append((cut_id, ref_words_filtered, hyp_words_filtered))

            results[name].extend(this_batch)

        num_cuts += len(texts)
        if batch_idx % 20 == 0:
            logging.info(f"Batch {batch_idx}/{num_batches}, cuts processed: {num_cuts}")

    return results

def save_results(
    exp_dir: Path,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
    suffix: str,
):
    res_dir = exp_dir / "jit_decode"
    res_dir.mkdir(parents=True, exist_ok=True)

    for key, results in results_dict.items():
        recog_path = res_dir / f"recogs-{test_set_name}-{key}-{suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"Transcripts stored in {recog_path}")

        errs_filename = res_dir / f"errs-{test_set_name}-{key}-{suffix}.txt"
        with open(errs_filename, "w", encoding="utf-8") as f:
            wer = write_error_stats(f, f"{test_set_name}-{key}", results, enable_log=True)
        logging.info(f"WER for {test_set_name}-{key}: {wer}")

@torch.no_grad()
def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = get_parser()
    FinetuneAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    
    suffix = f"{args.decoding_method}-jit"
    if args.decoding_method == "modified_beam_search":
        suffix += f"-beam-{args.beam_size}"
    
    if args.norm:
        suffix += "-norm"
    
    args.exp_dir = Path(args.exp_dir)
    setup_logger(f"{args.exp_dir}/log-jit-decode-{args.cuts_name}")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading JIT model: {args.nn_model_filename}")
    model = torch.jit.load(args.nn_model_filename, map_location=device)
    model.eval()
    model.to(device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    data_module = FinetuneAsrDataModule(args)
    
    test_cuts_list = []
    test_sets = []

    if args.cuts_name == "all":
        test_sets = ["test", "dev"]
        test_cuts_list = [data_module.test_cuts(), data_module.dev_cuts()]
    elif args.cuts_name == "test":
        test_sets = ["test"]
        test_cuts_list = [data_module.test_cuts()]
    elif args.cuts_name == "dev":
        test_sets = ["dev"]
        test_cuts_list = [data_module.dev_cuts()]
    else:
        custom_cuts_path = Path(args.manifest_dir) / f"vietASR_cuts_{args.cuts_name}.jsonl.gz"
        if custom_cuts_path.exists():
            from lhotse import load_manifest_lazy
            test_sets = [args.cuts_name]
            test_cuts_list = [load_manifest_lazy(custom_cuts_path)]
        else:
            logging.error(f"Cuts file not found: {custom_cuts_path}")
            return

    for test_set, cuts in zip(test_sets, test_cuts_list):
        logging.info(f"Decoding dataset: {test_set} using {args.decoding_method}")
        dl = data_module.test_dataloaders(cuts)
        results_dict = decode_dataset(
            dl=dl, 
            model=model, 
            sp=sp, 
            norm=args.norm,
            decoding_method=args.decoding_method,
            beam_size=args.beam_size
        )
        
        save_results(args.exp_dir, test_set, results_dict, suffix)

    logging.info("Decoding Done!")

if __name__ == "__main__":
    main()
