# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import torch
import numpy as np
import random
import lhotse
from typing import Dict, Any,  Optional
from sklearn.cluster import MiniBatchKMeans
from lhotse import CutSet, load_manifest_lazy
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

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

class KmeansDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        # self._validate(cuts)
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        features = [cut.load_features() for cut in cuts]
        feature_lens = [cut.num_frames for cut in cuts]

        # part_feats = np.concatenate(part_feats, axis=0)
        return {
            "cuts": cuts,
            "features": features,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

def train_dataloaders(
    cuts_train: CutSet,
    sampler_state_dict: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """
    Args:
        cuts_train:
        CutSet for training.
        sampler_state_dict:
        The state dict for the training sampler.
    """
    logging.info("About to create train dataset")
    train = KmeansDataset()

    bucketing_sampler = True
    max_duration = 1200
    shuffle = True
    num_buckets = 30
    drop_last = True
    if bucketing_sampler:
        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=max_duration,
            shuffle=shuffle,
            num_buckets=num_buckets,
            drop_last=drop_last,
        )
    else:
        logging.info("Using SimpleCutSampler.")
        train_sampler = SimpleCutSampler(
            cuts_train,
            max_duration=args.max_duration,
            shuffle=args.shuffle,
        )
    logging.info("About to create train dataloader")

    if sampler_state_dict is not None:
        logging.info("Loading sampler state dict")
        train_sampler.load_state_dict(sampler_state_dict)

    # 'seed' is derived from the current random state, which will have
    # previously been set in the main process.
    seed = torch.randint(0, 100000, ()).item()
    worker_init_fn = _SeedWorkers(seed)

    num_workers = 2
    train_dl = DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )

    return train_dl

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
        # print(sorted_filenames)
        cuts = lhotse.combine(
            lhotse.load_manifest_lazy(p) for p in sorted_filenames
        )
    return cuts


def learn_kmeans(
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
    train_dl = train_dataloaders(cuts)

    part_feats_holder = []
    for batch in train_dl:
        part_feats_holder.extend(batch["features"])

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("km_path", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--files", type=str, nargs="*", default = None)
    parser.add_argument("--do_training", action="store_true")
    parser.add_argument("--src_dir", type=str, default = None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
