# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import finetune_tri_stage as finetune
import logging
import os
import sys
from datetime import datetime

import torch
import numpy as np
import random
import sentencepiece as spm
import lhotse
from asr_datamodule import FinetuneAsrDataModule
from typing import Dict, Any,  Optional
from sklearn.cluster import MiniBatchKMeans
from lhotse import CutSet, load_manifest_lazy
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from finetune_tri_stage import add_model_arguments
from pathlib import Path
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

import joblib

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
#     filename=f'exp/learn_kmeans_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
#     mode='w'
# )
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


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
    if cut_files is not None:
        cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in cut_files
        )
    else:
        cut_files = os.listdir(src_dir)
        cut_files = [os.path.join(src_dir, item) for item in cut_files if item.endswith(".jsonl.gz") and item.find("_raw")<=0]
        sorted_filenames = sorted(cut_files)
        cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in sorted_filenames
        )
    return cuts


def extract_feature(batch, model):
    if model is None:
        return batch["features"]
    device = next(model.parameters()).device
    audio = batch["audio"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    encoder_out, encoder_out_lens = model.forward_encoder(audio, padding_mask, do_final_down_sample = False)
    b, l, d = encoder_out.shape
    holder = []
    for i in range(b):
        holder.append(encoder_out[i, :encoder_out_lens[i], :])      # (l, d), in a list
    encoder_out = torch.cat(holder, dim=0).to(torch.device("cpu")).detach().numpy()     # (L, d)
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
    parser.add_argument("--batch-size", default=1000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max-no-improvement", default=100, type=int)
    parser.add_argument("--n-init", default=20, type=int)
    parser.add_argument("--reassignment-ratio", default=0.0, type=float)
    parser.add_argument("--seed", type=int, default=42)

    # To decide which kind of checkpoint to use
    parser.add_argument("--checkpoint-type", type=str, default="pretrain")

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

    add_model_arguments(parser)
    return parser


def get_model(params, device):
    if params.checkpoint_type == "ASR":
        params.use_layernorm = False

    if params.checkpoint_type == "pretrain":
        model = finetune.get_model(params) 
    else:       # ASR & finetune
        params.final_downsample = True # to avoid parameter shape mismatch
        params.do_final_downsample = False # to not use down sample
        model = finetune.get_model(params)
        model.to(device)
        # logging.info(model)
        # for name in model.state_dict().keys():
        #     print(name)
        #     logging.info(name)
        # AsrModel, encoder is a HubertModel, with its encoder being a zipformer2
        checkpoint = get_avg_checkpoint(
            params.pretrained_dir,
            params.epoch,
            params.avg,
            params.use_averaged_model,
            params.iter,
            device
        )
        # logging.info([item for item in checkpoint])
        # AsrModel, with its encoder being a zipformer, model load this part as init for HubertModel's encoder
        
        if params.checkpoint_type == "ASR":
            for item in list(checkpoint):
                if not item.startswith("encoder.") and not item.startswith("encoder_embed."):
                    # print(item)
                    checkpoint.pop(item)        # leave only encoder part
            checkpoint.pop("encoder.downsample_output.bias")
            missing_keys, unexpected_keys = model.encoder.load_state_dict(checkpoint, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
        logging.info(f"Init checkpoint, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

    model.eval()
    model.to(device)
    return model

def learn_kmeans(
    args,
    do_training,
    files,
    src_dir,
    km_dir,
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
    km_path = os.path.join(km_dir, 'kmeans.pt')
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
    finetune_datamoddule = FinetuneAsrDataModule(args)
    train_dl = finetune_datamoddule.test_dataloaders(cuts)

    fix_random_seed(args.seed)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logging.info(f"Device: {device}")

    part_feats_holder = []
    model = get_model(args, device)

    for batch in train_dl:
        part_feats_holder.append(extract_feature(batch, model))     # model.forward_encoder
    # HubertModel.extract_feature

    # part_feats_holder = [arr.astype(np.float16) for arr in part_feats_holder]  
    # part_feats = np.empty((0, 512), dtype=np.float16)  
    # for chunk in np.array_split(part_feats_holder, 10000):  
        # part_feats = np.concatenate((part_feats, chunk), axis=0)
    part_feats = np.concatenate(part_feats_holder, axis=0)      # OOM, try to lower precision to fp16
    # part_feats = part_feats.astype(np.float16)
    logging.info(f"data size: {part_feats.shape}")

    if do_training:
        km_model.fit(part_feats)
        joblib.dump(km_model, km_path)

    inertia = -km_model.score(part_feats) / len(part_feats)
    logging.info(f"Total inertia: {inertia:.5f}")

    # inertia = -km_model.score(feat) / len(feat)
    # logging.info("total intertia: %.5f", inertia)
    logging.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    # parser.add_argument("--init", default="k-means++")
    # parser.add_argument("--batch_size", default=256, type=int)
    # parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    
    FinetuneAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    setup_logger(f"{args.km_path}/learn-kmeans-iter-{args.iteration}")
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
        km_dir = args.km_path,
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
