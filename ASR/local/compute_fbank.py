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


"""
This file computes fbank features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from filter_cuts import filter_cuts
from icefall.utils import get_executor, str2bool
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("../data6/bactrungbo/manifests"),
        help="Path to the manifest directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data6/fbank"),
        help="Path to the output directory for features",
    )

    return parser.parse_args()


def compute_fbank(
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = False,
    src_dir: Path = Path("../data6/bactrungbo/manifests"),
    output_dir: Path = Path("../data6/fbank"),
):
    # Reduce to avoid OOM during feature extraction
    num_jobs = min(8, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "dev",
            "test",
            "train",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "vietASR"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    
    # Fallback: Manual load if read_manifests_if_cached fails
    if not manifests:
        logging.info("read_manifests_if_cached returned empty. Trying manual load...")
        manifests = {}
        from lhotse import load_manifest, CutSet, RecordingSet, SupervisionSet
        
        for part in dataset_parts:
            # Try multiple naming patterns
            cut_patterns = [
                src_dir / f"{prefix}_cuts_{part}.{suffix}",
                src_dir / f"cuts_{part}.{suffix}",
                src_dir / f"{part}_cuts.{suffix}",
            ]
            
            rec_patterns = [
                (src_dir / f"{prefix}_recordings_{part}.{suffix}", src_dir / f"{prefix}_supervisions_{part}.{suffix}"),
                (src_dir / f"recordings_{part}.{suffix}", src_dir / f"supervisions_{part}.{suffix}"),
                (src_dir / f"{part}_recordings.{suffix}", src_dir / f"{part}_supervisions.{suffix}"),
            ]
            
            found = False
            
            # Try to find cuts file first
            for cut_path in cut_patterns:
                if cut_path.exists():
                    logging.info(f"Found cuts for {part}: {cut_path}")
                    manifests[part] = {"cuts": load_manifest(cut_path)}
                    found = True
                    break
            
            # If no cuts, try to find recordings + supervisions
            if not found:
                for rec_path, sup_path in rec_patterns:
                    if rec_path.exists() and sup_path.exists():
                        logging.info(f"Found recordings+supervisions for {part}: {rec_path}, {sup_path}")
                        recordings = load_manifest(rec_path)
                        supervisions = load_manifest(sup_path)
                        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
                        manifests[part] = {"cuts": cuts}
                        found = True
                        break
            
            if not found:
                logging.warning(f"Could not find manifests for {part}")

    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            if "cuts" in m:
                cut_set = m["cuts"]
            else:
                cut_set = CutSet.from_manifests(
                    recordings=m["recordings"],
                    supervisions=m["supervisions"],
                )


            if "train" in partition:
                if bpe_model:
                    cut_set = filter_cuts(cut_set, sp)
            
            # Apply text normalization (normalize Vietnamese tone placement)
            logging.info(f"Applying text normalization for {partition}")
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))
            from normalize_hybrid import normalize_hybrid_tone
            
            def normalize_text(cut):
                for sup in cut.supervisions:
                    if sup.text:
                        sup.text = normalize_hybrid_tone(sup.text.lower())
                return cut
            
            cut_set = cut_set.map(normalize_text)
            
            # Apply speed perturbation if requested (for any partition)
            if perturb_speed:
                logging.info(f"Doing speed perturb for {partition}")
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank(
        bpe_model=args.bpe_model,
        dataset=args.dataset,
        perturb_speed=args.perturb_speed,
        src_dir=args.src_dir,
        output_dir=args.output_dir,
    )