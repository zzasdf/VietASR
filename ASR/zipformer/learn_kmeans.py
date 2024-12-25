# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import train as asr_train
import logging
import os
import sys

import torch
import numpy as np
import random
import sentencepiece as spm
import lhotse
from asr_datamodule import TencentAsrDataModule
from typing import Dict, Any,  Optional
from sklearn.cluster import MiniBatchKMeans
from lhotse import CutSet, load_manifest_lazy
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from pathlib import Path
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    str2bool,
)

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)

def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def get_cuts(cut_files, src_dir):
    def map_function(old_prefix, new_prefix):
        def f(cut):
            old_path = cut.features.storage_path
            assert old_path.startswith(old_prefix), f"{cut.id} has feature path {old_path}"
            cut.features.storage_path = new_prefix + old_path[len(old_prefix):]
            return cut
        return f

    if cut_files is not None:
        cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in cut_files
        )

        # in case the path is not correct
        # cuts = cuts.map(map_function(
        #     old_prefix = "data/fbank_2000h/train_split/",
        #     new_prefix = "/workdir/data/vi/ssl_finetune/fbank_2000h/"
        # ))
    else:
        cut_files = os.listdir(src_dir)
        cut_files = [os.path.join(src_dir, item) for item in cut_files if item.endswith(".jsonl.gz") and item.find("_raw")<=0]
        sorted_filenames = sorted(cut_files)
        # print(sorted_filenames)
        cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in sorted_filenames
        )
    
        cuts = cuts.map(map_function(
            old_prefix = "data/fbank_2000h/train_split/",
            new_prefix = "/workdir/data/vi/ssl_finetune/fbank_2000h/"
        ))
    return cuts

def extract_feature(batch, model, do_norm=False):
    if model is None:
        return batch["inputs"]
    device = next(model.parameters()).device
    audio = batch["inputs"].to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    encoder_out, encoder_out_lens = model.forward_encoder(audio, feature_lens, final_downsample=False)
    b, l, d = encoder_out.shape
    holder = []
    # import pdb; pdb.set_trace()
    for i in range(b):
        holder.append(encoder_out[i, :encoder_out_lens[i], :])
    if do_norm:
        encoder_out = torch.cat(holder, dim=0)
        encoder_norm = torch.sqrt(torch.square(encoder_out).sum(dim=-1))
        encoder_out = encoder_out/encoder_norm.unsqueeze(1)
        encoder_out = encoder_out.to(torch.device("cpu")).detach().numpy()
    else:
        encoder_out = torch.cat(holder, dim=0).to(torch.device("cpu")).detach().numpy()
    # import pdb; pdb.set_trace()
    return encoder_out

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--km-path", type=str)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--files", type=str, nargs="*", default = None)
    parser.add_argument("--do-training", action="store_true")
    parser.add_argument("--src-dir", type=str, default = None)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max-iter", default=100, type=int)
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max-no-improvement", default=100, type=int)
    parser.add_argument("--n-init", default=20, type=int)
    parser.add_argument("--reassignment-ratio", default=0.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
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

    asr_train.add_model_arguments(parser)
    return parser


def get_model(params, device):
    if args.checkpoint_type == "pretrain":
        model = asr_train.get_model(params)
    else:
        model = asr_train.get_model(params)

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

def learn_kmeans(
    args,
    do_training,
    files,
    src_dir,
    km_path,
    n_clusters,
    seed,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    if do_training:
        km_model = get_km_model(
            n_clusters,
            init,
            max_iter,
            batch_size,
            tol,
            max_no_improvement,
            n_init,
            reassignment_ratio,
        )
        # km_model.fit(feat)
    else:
        km_model = joblib.load(km_path)
    cuts = get_cuts(files, src_dir)
    tencent_datamoddule = TencentAsrDataModule(args)
    train_dl = tencent_datamoddule.test_dataloaders(cuts)

    fix_random_seed(args.seed)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logging.info(f"Device: {device}")


    part_feats_holder = []
    model = get_model(args, device)


    for batch in train_dl:
        part_feats_holder.append(extract_feature(batch, model, do_norm=args.do_norm))

    part_feats = np.concatenate(part_feats_holder, axis=0)
    logging.info(f"data size: {part_feats.shape}")
    if do_training:
        km_model.fit(part_feats)
        joblib.dump(km_model, km_path)
    inertia = -km_model.score(part_feats) / len(part_feats)
    logging.info(f"Total inertia: {inertia:.5f}")


    # inertia = -km_model.score(feat) / len(feat)
    # logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    # fbank->feature's kmeans model
    parser = get_parser()
    TencentAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    args.blank_id = sp.piece_to_id("<blk>")
    args.vocab_size = sp.get_piece_size()

    args.feature_dim = 80
    logging.info(str(args))

    learn_kmeans(
        args,
        do_training = args.do_training,
        files = args.files,
        src_dir = args.src_dir,
        km_path = args.km_path,
        n_clusters = args.n_clusters,
        seed = args.seed,
        init = args.init,
        max_iter = args.max_iter,
        batch_size = args.batch_size,
        tol = args.tol,
        n_init = args.n_init,
        reassignment_ratio = args.reassignment_ratio,
        max_no_improvement = args.max_no_improvement,
    )
