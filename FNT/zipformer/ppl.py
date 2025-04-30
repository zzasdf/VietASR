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

(2) beam search
./transducer/decode.py \
        --epoch 14 \
        --avg 7 \
        --exp-dir ./transducer/exp \
        --max-duration 100 \
        --decoding-method beam_search \
        --beam-size 8
"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from joiner import Joiner
from model import FactorizeTransducer

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
import k2
import copy

class VocabDecoder(nn.Module):
    def __init__(self, net) -> None:
        super().__init__()
        self.blank_id =  net.vocab_decoder.blank_id
        self.decoder = copy.deepcopy(net.vocab_decoder)
        self.laynorm_proj = copy.deepcopy(net.joiner.laynorm_proj_vocab_decoder)
        self.fc = copy.deepcopy(net.joiner.fc_out_decoder_vocab)
    def forward(self, y):
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        lm_target = torch.clone(y.values)
        # since we don't use a pad_id, we just concat the labels, which can be done by using its values
        lm_target = lm_target.to(torch.int64)
        lm_target[lm_target > self.blank_id] -= 1
        # shift because the lm_predictor output does not include the blank_id

        sos_y = add_sos(y, sos_id=self.blank_id)

        sos_y_padded = sos_y.pad(mode="constant", padding_value=self.blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)
        decoder_out,_ = self.decoder(sos_y_padded)
        decoder_out = self.laynorm_proj(decoder_out)
        decoder_out = self.fc(decoder_out)
        lm_lprobs = torch.nn.functional.log_softmax(
            decoder_out,
            dim=-1,
        )
        lm_lprobs = [item[:y_len] for item, y_len in zip(lm_lprobs, y_lens)] 
        # note that the last output of each sentence is not used here
        lm_lprobs = torch.cat(lm_lprobs)
        loss = torch.nn.functional.nll_loss(
            lm_lprobs,
            lm_target,
            reduction="sum")
        return loss, y_lens



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
        default=11,
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
        "--beam-size",
        type=int,
        default=5,
        help="Used only when --decoding-method is beam_search",
    )

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
            "env_info": get_env_info(),
        }
    )
    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict):
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        joint_dim=params.joiner_dim,
        encoder_hidden_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_hidden_dim=params.decoder_dim,
        output_dim=params.vocab_size,
        blank_id=params.blank_id,
    )
    return joiner


def get_transducer_model(params: AttributeDict):
    encoder = get_encoder_model(params)
    encoder_embed = get_encoder_embed(params)
    blank_decoder = get_decoder_model(params)
    vocab_decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = FactorizeTransducer(
        encoder=encoder,
        encoder_embed=encoder_embed,
        blank_decoder=blank_decoder,
        vocab_decoder=vocab_decoder,
        joiner=joiner,
    )
    return model


def ppl_one_batch(
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    batch_idx: int,
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
    device = model.device

    feature = batch["inputs"]
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    # Now for the decoder, i.e., the prediction network
    result = dict()
    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    result["y"] = y
    result["batch_idx"] = [batch_idx] * len(y)
    result["label"] = []
    result["predict"] = []
    y = k2.RaggedTensor(y).to(device)
    row_splits = y.shape.row_splits(1)
    y_lens = row_splits[1:] - row_splits[:-1]

    blank_id = model.blank_id

    lm_target = torch.clone(y.values)
    # since we don't use a pad_id, we just concat the labels, which can be done by using its values
    lm_target = lm_target.to(torch.int64)
    lm_target[lm_target > blank_id] -= 1
    # shift because the lm_predictor output does not include the blank_id

    sos_y = add_sos(y, sos_id=blank_id)

    sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
    sos_y_padded = sos_y_padded.to(torch.int64)

    decoder_out = model.decoder(sos_y_padded, need_pad=True)
    lm_lprobs = torch.nn.functional.log_softmax(
        model.fc(model.laynorm_proj(decoder_out)),
        dim=-1,
    )
    ppl = 0
    acc_len = 0
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    # batch_size = lm_lprobs.shape[0]
    # concat_lm_lprobs = torch.cat(
    #     [item[:y_len] for item, y_len in zip(lm_lprobs, y_lens)]
    # )
    # ppl = torch.exp(loss_fn(concat_lm_lprobs, lm_target)) * batch_size
    for i in range(lm_lprobs.shape[0]):
        ppl+=torch.exp(loss_fn(lm_lprobs[i, :y_lens[i]], lm_target[acc_len:acc_len+y_lens[i]]))
        result["predict"].append(
            torch.argmax(lm_lprobs[i, : y_lens[i]], dim=-1)
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
        result["label"].append(
            lm_target[acc_len : acc_len + y_lens[i]].cpu().detach().numpy().tolist()
        )
        acc_len += y_lens[i]

    return ppl, lm_lprobs.shape[0], result

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
    total_size = 0
    total_ppl = 0
    result = {"batch_idx": [], "y": [], "label": [], "predict": []}
    for batch_idx, batch in enumerate(dl):
        ppl, batch_size, batch_result_dict = ppl_one_batch(
            model=model,
            sp=sp,
            batch=batch,
            batch_idx=batch_idx,
            params=params
        )
        total_size += batch_size
        total_ppl += ppl
        for name, item in batch_result_dict.items():
            result[name].extend(item)

    return total_ppl / total_size, result


def save_results(
    params: AttributeDict,
    test_set_name: str,
    ppl,
    results_dict,
):
    recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    with open(recog_path, "w") as f:
        for i in range(len(results_dict["y"])):
            print(f"batch_idx: {results_dict['batch_idx'][i]}", file=f)
            print(f"\ty: {results_dict['y'][i]}", file=f)
            print(f"\tlabel: {results_dict['label'][i]}", file=f)
            print(f"\tpredict: {results_dict['predict'][i]}", file=f)
    errs_info = params.res_dir / f"ppl-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print(f"ppl: \t{ppl}", file=f)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
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
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    from icefall.record_utils import backup
    backup(log_dir, __file__, params)

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    if params.vocab_decoder_source_type == "adaptation":
        model = VocabDecoder(model)

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

    if params.vocab_decoder_source_type == "transducer":
        model = VocabDecoder(model)

    model.to(device)
    model.eval()
    model.device = device

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        ppl, results_dict = compute_ppl_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            ppl=ppl,
            results_dict=results_dict,
        )

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
