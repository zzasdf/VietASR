#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
Usage:
(1) greedy search
./transducer/decode.py \
        --epoch 14 \
        --avg 7 \
        --exp-dir ./transducer/exp \
        --max-duration 100 \
        --decoding-method greedy_search

"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn

from fnt import FactorizeTransducer, FNTJoiner
from ifnt import ImprovedFactorizedTransducer, IFNTJoiner
from asr_datamodule import AsrDataModule

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2
from icefall.utils import (
    AttributeDict,
    setup_logger,
    add_sos,
    str2bool,
)
import k2, lhotse
import copy
import math
from torch.nn.parallel import DistributedDataParallel as DDP


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=False,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=34,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transducer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
        """,
    )

    parser.add_argument(
        "--vocab-decoder-source-type",
        type=str,
        default="transducer",
        help="""
        type of the vocab decoder source
        """,
    )

    parser.add_argument(
        "--pretrain-path",
        type=str,
        help="""
        Pretrained checkpoint for encoder
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Used only when --decoding-method is beam_search",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="FNT",
        help="FNT or IFNT",
    )

    parser.add_argument(
        "--train-stage",
        type=str,
        default="asr",
        help="""
        asr or adapt,
        """,
    )

    parser.add_argument(
        "--use-local-rnnt-loss",
        type=str2bool,
        default=False,
        help="If True, use cpp rnnt loss instead of torchaudio.functional.rnnt_loss",
    )

    parser.add_argument(
        "--test-cut",
        type=str,
        default="data/devtest/mgb2_cuts_dev.jsonl.gz",
        help="path to the test cut jsonl file"
    )

    parser.add_argument("--device", default=0, type=int)

    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            # decoder params
            "decoder_embedding_dim": 1024,
            "num_decoder_layers": 2,
            "decoder_dim": 512,
            "embedding_dropout": 0.1,
            "rnn_dropout": 0.1,
            "env_info": get_env_info(),
        }
    )
    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_transducer_model(params: AttributeDict):
    assert params.model_type == "FNT" or params.model_type == "IFNT"
    assert params.use_transducer is True
    if params.model_type == "FNT":
        model = FactorizeTransducer.build_model(params)
    elif params.model_type == "IFNT":
        model = ImprovedFactorizedTransducer.build_model(params)
    else:
        raise
    return model


def compute_ppl_batch(
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    params: AttributeDict
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # info = dict()
    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module

    with torch.no_grad():
        batch_loss, batch_tokens, batch_result = model.ppl_one_batch(y=y)
        # info['utt_ppl'] = batch_result['ppl']
        # info['ppl'] = batch_result['ppl']
        return batch_loss, batch_tokens


def compute_ppl_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    total_tokens = 0
    total_ppl = 0
    result = {"y": [], "label": [], "predict": []}
    for batch_idx, batch in enumerate(dl):
        # ppl, batch_tokens, batch_result_dict = compute_ppl_batch(
        batch_loss, batch_tokens = compute_ppl_batch(
            model=model,
            sp=sp,
            batch=batch,
            params=params,
        )
        total_tokens += batch_tokens
        total_ppl += batch_loss
        # for name, item in batch_result_dict.items():
            # result[name].extend(item)

    return math.exp(total_ppl / total_tokens), result


def save_results(
    params: AttributeDict,
    test_set_name: str,
    ppl,
    results_dict,
):
    recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    with open(recog_path, "w") as f:
        for i in range(len(results_dict["y"])):
            print(f"\ty: {results_dict['y'][i]}", file=f)
            print(f"\tlabel: {results_dict['label'][i]}", file=f)
            print(f"\tpredict: {results_dict['predict'][i]}", file=f)
    errs_info = params.res_dir / f"ppl-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print(f"ppl: \t{ppl}", file=f)


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    params.res_dir = params.exp_dir / "ppl" / params.suffix

    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"{params.res_dir}/log-dir-{date_time}"
    setup_logger(f"{log_dir}/log-decode-{params.suffix}")
    logging.info("Start computing ppl")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()
    model.device = device

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    asrDataModule = AsrDataModule(args)

    test_cuts = lhotse.load_manifest_lazy(args.test_cut)
    test_dl = asrDataModule.test_dataloaders(test_cuts)

    ppl, results_dict = compute_ppl_dataset(
        dl=test_dl,
        params=params,
        model=model,
        sp=sp,
    )
    logging.info(f"TOTAL PPL: {ppl}")

    save_results(
        params=params,
        test_set_name=Path(args.test_cut).stem,
        ppl=ppl,
        results_dict=results_dict,
    )

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
