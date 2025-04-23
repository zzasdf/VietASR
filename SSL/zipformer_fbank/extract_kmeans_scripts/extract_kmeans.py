# /usr/bin/bash
import argparse
import os, sys
from datetime import datetime
from pathlib import Path
import sentencepiece as spm
import json
import logging
import torch
import numpy as np
import joblib
# import finetune
import finetune_tri_stage as finetune
from torch import nn, einsum
from tqdm import tqdm
from asr_datamodule import FinetuneAsrDataModule
from einops import rearrange
from lhotse import CutSet, load_manifest_lazy
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
import time
from typing import Dict, Any,  Optional
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    setup_logger,
    str2bool,
)
from utils import get_avg_checkpoint


# formatter = logging.Formatter(
#     "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.DEBUG,
#     stream=sys.stdout,
# )
# logger = logging.getLogger("learn_kmeans")
# file_handler = logging.FileHandler(
#     filename=f'exp/extract_kmeans_iter1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
#     mode='w'
# )
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

def get_model(params, device):
    if params.checkpoint_type == "ASR":
        params.use_layernorm = False

    if params.checkpoint_type == "pretrain":
        model = finetune.get_model(params)
    else:
        params.final_downsample = True # to avoid parameter shape mismatch
        params.do_final_downsample = False # to not use down sample
        model = finetune.get_model(params)
        model.to(device)
        checkpoint = get_avg_checkpoint(
            params.pretrained_dir,
            params.epoch,
            params.avg,
            params.use_averaged_model,
            params.iter,
            device
        )
        if params.checkpoint_type == "ASR":
            for item in list(checkpoint):
                if not item.startswith("encoder.") and not item.startswith("encoder_embed."):
                    checkpoint.pop(item)
            checkpoint.pop("encoder.downsample_output.bias")
            missing_keys, unexpected_keys = model.encoder.load_state_dict(checkpoint, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
        logging.info(f"Init checkpoint, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

    model.eval()
    model.to(device)
    return model


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--km-path", type=str)
    parser.add_argument("--task-list", type=str)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--src-dir", type=str, help="for build list")
    parser.add_argument("--file", type=str)
    parser.add_argument("--device", type=int)

    # To decide which kind of checkpoint to use
    parser.add_argument("--checkpoint-type", type=str, default = "pretrain")
    parser.add_argument("--iteration", type=int, default=1)

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--pretrained-dir",
        type=str,
        help="""The pretrained model dir.
        It specifies the directory where the pretrained checkpoint is saved.""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    finetune.add_model_arguments(parser)

    return parser


class ApplyKmeans(object):
    def __init__(self, km_path, device):
        self.km_path = os.path.join(km_path, 'kmeans.pt')
        self.km_model = joblib.load(self.km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    @torch.no_grad()
    def __call__(self, x):
        # x: b, d
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1)
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def extract_feature(batch, model):
    if model is None:
        return batch["features"]
    device = next(model.parameters()).device
    audio = batch["audio"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    encoder_out, encoder_out_lens = model.forward_encoder(audio, padding_mask, do_final_down_sample=False)
    b, l, d = encoder_out.shape
    holder = []
    for i in range(b):
        holder.append(encoder_out[i, :encoder_out_lens[i], :])
    encoder_out = torch.cat(holder, dim=0)
    return encoder_out, encoder_out_lens


def sub_routine(batch, model, km_model, km_dict, device):
    feat, len_lis = extract_feature(batch, model)
    kmeans = km_model(feat).to(torch.device("cpu"))
    # logging.info(kmeans.shape)

    offset = 0
    cut_ids = [cut.id for cut in batch["cuts"]]
    # len_lis = batch["feature_lens"]
    for cut_id, feat_len in zip(cut_ids, len_lis):
        label = [str(int(item)) for item in kmeans[offset: offset+feat_len]]
        km_dict[cut_id] = " ".join(label)
        offset += feat_len


def remove_short_and_long_utt(c):

    if c.duration < 0.5 or c.duration > 30.0:
        return False

    # num_frames = c.num_frames if c.num_frames else c.duration * 100
    # T = ((num_frames - 7) // 2 + 1) // 2
    # tokens = sp.encode(c.supervisions[0].text, out_type=str)

    # if T < len(tokens):
    #     logging.warning(
    #         f"Exclude cut with ID {c.id} from training. "
    #         f"Number of frames (before subsampling): {num_frames}. "
    #         f"Number of frames (after subsampling): {T}. "
    #         f"Text: {c.supervisions[0].text}. "
    #         f"Tokens: {tokens}. "
    #         f"Number of tokens: {len(tokens)}"
    #     )
    #     return False

    return True


def main(args):
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    args.blank_id = sp.piece_to_id("<blk>")
    args.vocab_size = sp.get_piece_size()

    args.feature_dim = 80

    setup_logger(f"{args.km_path}/extract-kmeans-iter-{args.iteration}")

    logging.info(str(args))
    device = torch.device(f"cuda:{args.device}")
    model = ApplyKmeans(args.km_path, device)

    feature_model = get_model(args, device)

    # apply_kmeans = ApplyKmeans(km_path)
    # task_file = args.task_list
    # with open(task_file, 'r') as f:
        # task_list = f.readlines()
    # task_list = [item.split()  for item in task_list]

    # if args.start is not None:
        # task_list = task_list[args.start:args.end]

    if args.checkpoint_type == "ASR":
        args.iteration = 1

    task_list = []
    if args.src_dir is not None:
        cut_files = os.listdir(args.src_dir)
        cut_files = [os.path.join(args.src_dir, item) for item in cut_files if item.endswith(".jsonl.gz") and item.find("_raw")<=0]
        for src in cut_files:
            tgt = src.replace(".jsonl.gz", f"_km_iter{args.iteration}.jsonl.gz")
            task_list.append((src, tgt))
    elif args.file is not None:
        src = args.file
        tgt = src.replace(".jsonl.gz", f"_km_iter{args.iteration}.jsonl.gz")
        task_list.append((src, tgt))
    else:
        raise 


    for src, tgt in tqdm(task_list):
        # if os.path.isfile(tgt):
        #     continue
        cuts = CutSet.from_file(src)
        cuts = cuts.filter(remove_short_and_long_utt)

        km_dict = {}
        finetune_datamoddule = FinetuneAsrDataModule(args)
        test_dl = finetune_datamoddule.test_dataloaders(cuts)

        for i, batch in enumerate(test_dl):
            # logging.info(f'iter{i}:\t {len(batch)}')
            sub_routine(batch, feature_model, model, km_dict, device)
        
        def add_label(km_dict):
            def f(cut):
                cut.custom = dict()
                cut.custom["kmeans"] = km_dict[cut.id]
                return cut
            return f

        cuts = cuts.map(add_label(km_dict))

        cuts.to_file(tgt)
        logging.info("finished successfully")
    

if __name__ == "__main__":
    parser = get_parser()
    FinetuneAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    main(args)
