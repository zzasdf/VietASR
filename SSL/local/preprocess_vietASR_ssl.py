#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.             (Yifan Yang)
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
import re
import string
import unicodedata
from pathlib import Path

from icefall.utils import str2bool
from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tgt-dir",
        type=str,
        required=True,
    )
    return parser.parse_args()


def preprocess_vietASR_ssl(args):
    src_dir = Path(args.src_dir)
    output_dir = Path(args.tgt_dir)
    output_dir.mkdir(exist_ok=True)

    dataset_parts = args.dataset.strip().split(" ", -1)

    logging.info("Loading manifest (may take 4 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix="vietASR-ssl",
        suffix="jsonl.gz",
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )
    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"vietASR-ssl_cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    preprocess_vietASR_ssl(args)


if __name__ == "__main__":
    main()
