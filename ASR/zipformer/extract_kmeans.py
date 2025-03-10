# /usr/bin/bash
import argparse
import os
from pathlib import Path
import sentencepiece as spm
import json
import logging
import torch
import tqdm
import numpy as np
import joblib
import train as train_asr
from torch import nn, einsum
import tqdm
from asr_datamodule import AsrDataModule
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
    str2bool,
)

logger = logging.getLogger("dump_km_label")

def get_model(params, device):
    model = train_asr.get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()


    return model
# feature->kmeans from ASR model
# load ASR model
# load feature

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("task")
    parser.add_argument("--model-path", type = str)
    parser.add_argument("--task-list", type=str)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--src-dir", type=str, nargs="*", help="for build list")
    parser.add_argument(
        "--do-norm",
        action="store_true"
    )

    # To decide which kind of checkpoint to use
    parser.add_argument("--checkpoint-type", type=str, default = "pretrain")

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
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
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

    train_asr.add_model_arguments(parser)

    return parser

class ApplyKmeans(object):
    def __init__(self, km_path, device):
        self.km_model = joblib.load(km_path)
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

class GPUTaker:
    def __init__(self, device):
        self.start_time = time.time()
        self.device = device
    def run(self):
        now_time = time.time()
        while ((now_time-self.start_time)//60)%30==0:
            size = (10240, 10240)
            taker_a = torch.rand(size, device=self.device)
            taker_b = torch.rand(size, device=self.device)
            taker = torch.rand(size, device=self.device)
            taker = taker_a*taker_b
            now_time = time.time()

def extract_feature(batch, model, do_norm=False):
    if model is None:
        return batch["inputs"]
    device = next(model.parameters()).device
    audio = batch["inputs"].to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    encoder_out, encoder_out_lens = model.forward_encoder(audio, feature_lens, final_downsample = False)
    b, l, d = encoder_out.shape
    holder = []
    # import pdb; pdb.set_trace()
    for i in range(b):
        holder.append(encoder_out[i, :encoder_out_lens[i], :])

    if do_norm:
        encoder_out = torch.cat(holder, dim=0)
        encoder_norm = torch.sqrt(torch.square(encoder_out).sum(dim=-1))
        encoder_out = encoder_out/encoder_norm.unsqueeze(1)
    else:
        encoder_out = torch.cat(holder, dim=0)
    # encoder_out = torch.cat(holder, dim=0).to(torch.device("cpu")).detach().numpy()
    # import pdb; pdb.set_trace()
    return encoder_out, encoder_out_lens

def sub_routine(batch, model, km_model, km_dict, device, do_norm):
    # feat = torch.cat(feat_lis, dim=0)
    # print("----------------------")
    # print(feat.shape)
    feat, len_lis = extract_feature(batch, model, do_norm)
    kmeans = km_model(feat).to(torch.device("cpu"))
    # print(kmeans.shape)
    offset = 0
    cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
    # len_lis = batch["feature_lens"]
    for cut_id, feat_len in zip(cut_ids, len_lis):
        label = [str(int(item)) for item in kmeans[offset: offset+feat_len]]
        km_dict[cut_id] = " ".join(label)
        offset+=feat_len

def map_function(old_prefix, new_prefix):
    def f(cut):
        old_path = cut.features.storage_path
        assert old_path.startswith(old_prefix), f"{cut.id} has feature path {old_path}"
        cut.features.storage_path = new_prefix + old_path[len(old_prefix):]
        return cut
    return f

def main(args):
    args.exp_dir = Path(args.exp_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    args.blank_id = sp.piece_to_id("<blk>")
    args.unk_id = sp.piece_to_id("<unk>")
    args.vocab_size = sp.get_piece_size()

    args.feature_dim = 80

    logging.info(str(args))
    device = torch.device("cuda:0")
    model = ApplyKmeans(args.model_path, device)

    feature_model = get_model(args, device)

    # apply_kmeans = ApplyKmeans(km_path)
    task_file = args.task_list
    with open(task_file, 'r') as f:
        task_lis = f.readlines()
    task_lis = [item.split()  for item in task_lis]

    if args.start is not None:
        task_lis = task_lis[args.start:args.end]

    for src, tgt in tqdm.tqdm(task_lis):
        # if os.path.isfile(tgt):
        #     continue
        cuts = CutSet.from_file(src)
        cuts = cuts.map(map_function(
            old_prefix = "/old_workdir/data/icefall/gigaspeech2_asr/data/fbank/",
            new_prefix = "/workdir/data/vi/ssl_testset/"
            # old_prefix = "/old_workdir/data/icefall/gigaspeech2_asr/data/fbank/",
            # new_prefix = "/workdir/data/vi/ssl_finetune/fbank_2000h/"
        ))
        km_dict = {}
        asr_data = AsrDataModule(args)
        test_dl = asr_data.test_dataloaders(cuts)

        for i, batch in enumerate(test_dl):
            sub_routine(batch, feature_model, model, km_dict, device, args.do_norm)

        
        def add_label(km_dict):
            def f(cut):
                cut.custom = dict()
                cut.custom["kmeans"] = km_dict[cut.id]
                return cut
            return f

        cuts = cuts.map(add_label(km_dict))

        cuts.to_file(tgt)
        logger.info("finished successfully")
    

if __name__ == "__main__":
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()


    task = args.task
    if task == "parallel":
        pass
    elif task == "build_model":
        pass
        # model = RandomProjectionQuantizer(input_dim=80, codebook_dim=args.codebook_dim, codebook_size=args.codebook_size)
        # torch.save(model.state_dict(), args.model_path)
    elif task == "run":
        main(args)




