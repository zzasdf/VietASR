#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import argparse
import logging
from pathlib import Path

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig, LilcomChunkyWriter

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in-manifest",
        type=str,
        required=True,
        help="Path to input manifest (cuts.jsonl.gz)",
    )

    parser.add_argument(
        "--out-manifest",
        type=str,
        required=True,
        help="Path to output manifest (cuts_with_feats.jsonl.gz)",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to store fbank features",
    )
    
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs",
    )

    return parser.parse_args()


def compute_fbank(args):
    in_manifest = Path(args.in_manifest)
    out_manifest = Path(args.out_manifest)
    out_dir = Path(args.out_dir)
    num_jobs = args.num_jobs

    if not in_manifest.exists():
        raise FileNotFoundError(f"{in_manifest} does not exist")

    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading {in_manifest}")
    cut_set = CutSet.from_file(in_manifest)
    logging.info(f"Loaded {len(cut_set)} cuts")

    # Force CPU to avoid GPU OOM issues
    device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    
    # Use KaldifeatFbank as it is standard in this repo for SSL
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info("Computing fbank features...")
    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{out_dir}/feats",
        num_jobs=num_jobs,
        storage_type=LilcomChunkyWriter,
    )

    logging.info(f"Saving to {out_manifest}")
    cut_set.to_file(out_manifest)
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    compute_fbank(args)
