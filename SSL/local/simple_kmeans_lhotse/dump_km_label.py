# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm
from lhotse import CutSet
from lhotse.workarounds import Hdf5MemoryIssueFix

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(cuts):
    hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
    update_step = 100

    def iterate():
        for i, cut in enumerate(cuts): 
            if i%update_step==0:
                hdf5_fix.update()
            yield cut.id, cut.load_features()
    return iterate, len(cuts)


def dump_label(cut_file, save_file, km_path):
    apply_kmeans = ApplyKmeans(km_path)
    cuts = CutSet.from_file(cut_file)
    cuts = cuts.subset(first = 1000)
    generator, num = get_feat_iterator(cuts)
    iterator = generator()
    km_dict = {}
    for cut_id, feat in tqdm.tqdm(iterator, total=num):
        # feat = torch.from_numpy(feat).cuda()
        lab = apply_kmeans(feat).tolist()
        km_dict[cut_id] = " ".join(map(str, lab))
    
    def add_label(km_dict):
        def f(cut):
            cut.custom = dict()
            cut.custom["kmeans"] = km_dict[cut.id]
            return cut
        return f

    cuts = cuts.map(add_label(km_dict))

    cuts.to_file(save_file)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cut_file")
    parser.add_argument("save_file")
    parser.add_argument("km_path")
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))
