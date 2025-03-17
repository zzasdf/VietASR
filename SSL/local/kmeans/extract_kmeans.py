# /usr/bin/bash
import os
import json
import logging
import torch
import numpy as np
import joblib
from torch import nn, einsum
import tqdm
from einops import rearrange
from argparse import ArgumentParser
from lhotse import CutSet, load_manifest_lazy
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
import time
from typing import Dict, Any,  Optional

logger = logging.getLogger("dump_km_label")

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

        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = [feature.shape[0] for feature in features]
        features = torch.cat(features, dim=0)

        # feature_lens = [cut.num_frames for cut in cuts]

        # part_feats = np.concatenate(part_feats, axis=0)
        return {
            "cuts": cuts,
            "features": features,
            "feature_lens": feature_lens
        }

def test_dataloaders(
    cuts: CutSet,
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
    test = KmeansDataset()

    bucketing_sampler = True
    max_duration = 1200
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=max_duration,
        shuffle=False,
    )
    logging.debug("About to create test dataloader")
    num_workers = 2
    test_dl = DataLoader(
        test,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
    )
    return test_dl

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



def main(args):
    logging.info(str(args))
    device = torch.device("cuda:0")
    model = ApplyKmeans(args.model_path, device)

    # apply_kmeans = ApplyKmeans(km_path)
    task_file = args.task_list
    with open(task_file, 'r') as f:
        task_lis = f.readlines()
    task_lis = [item.split()  for item in task_lis]

    if args.start is not None:
        task_lis = task_lis[args.start:args.end]

    for src, tgt in task_lis:
        cuts = CutSet.from_file(src)
        km_dict = {}
        test_dl = test_dataloaders(cuts)

        gpu_taker = GPUTaker(device)
        def sub_routine(feat_lis, len_lis, cut_ids):
            feat = torch.cat(feat_lis, dim=0)
            # print("----------------------")
            # print(feat.shape)
            kmeans = model(feat).to(torch.device("cpu"))
            # print(kmeans.shape)
            offset = 0
            for cut_id, feat_len in zip(cut_ids, len_lis):
                label = [str(int(item)) for item in kmeans[offset: offset+feat_len]]
                km_dict[cut_id] = " ".join(label)
                offset+=feat_len

        feat_lis = []
        len_lis = []
        cut_ids = []
        for i, batch in enumerate(test_dl):
            feat_lis.append(batch["features"].to(device))
            cut_ids.extend([cut.id for cut in batch["cuts"]])
            len_lis.extend(batch["feature_lens"])
            if i%10 == 0:
                sub_routine(feat_lis, len_lis, cut_ids)
                feat_lis = []
                len_lis = []
                cut_ids = []
            gpu_taker.run()
            # elif i%5==0:
            #     gpu_taker(device)

        if len(cut_ids)>0:
            sub_routine(feat_lis, len_lis, cut_ids)
        
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
    parser = ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--task-flag", type = str)
    parser.add_argument("--model-path", type = str)
    parser.add_argument("--task-list", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--src-dir", type=str, nargs="*", help="for build list")
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




