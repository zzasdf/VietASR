#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
import os
from datetime import datetime
from pathlib import Path
from lhotse.cut import data

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig
import multiprocessing
from multiprocessing import Pool, Lock

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device_lock = Lock()

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "task",
        type=str,
        default="run"
    )

    parser.add_argument(
        "--src-dir",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        type=str,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of dataloading workers used for reading the audio.",
    )

    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        help="The number of splits of the subset",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (inclusive).",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (exclusive).",
    )

    return parser



def compute_fbank_vietASR_ssl_splits(args):
    num_splits = args.num_splits
    output_dir = f"{args.src_dir}/{args.dataset}_split"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    num_digits = 8  # num_digits is fixed by lhotse split-lazy

    start = args.start
    stop = args.stop
    if stop < start:
        stop = num_splits

    stop = min(stop, num_splits)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info(f"device: {device}")

    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"vietASR-ssl_cuts_{args.dataset}.{idx}.jsonl.gz"
        # if cuts_path.is_file():
        #     logging.info(f"{cuts_path} exists - skipping")
        #     continue

        raw_cuts_path = (
            output_dir / f"vietASR-ssl_cuts_{args.dataset}_raw.{idx}.jsonl.gz"
        )

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/vietASR-ssl_feats_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    if task == "run":
        logging.info(vars(args))
        compute_fbank_vietASR_ssl_splits(args)
    elif task == "parallel":
        # device_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        def cleanup(processes):
            print("Cleaning up...")
            for process in processes:
                try:
                    # 终止子进程
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # device_list = [0,1,2,3,4,5,6,7]*2
        device_list = os.environ["CUDA_VISIBLE_DEVICES"]
        device_list = [int(item) for item in device_list.split(",")]
        src_dir = args.src_dir
        dataset = args.dataset
        lock_file_name = f"{dataset}_device_lock"

        with open(lock_file_name,'w') as f:
            print(",".join([str(item) for item in device_list]), file=f)

        cuts_lis = os.listdir(os.path.join(src_dir, f"{dataset}_split"))
        cuts_lis = [item for item in cuts_lis if item.endswith("jsonl.gz") and item.find("_raw")>=0]
        task_list = [int(item.rsplit('.', maxsplit=3)[-3]) for item in cuts_lis]
        os.makedirs("log_tem", exist_ok=True)
        #for sub_task, device in zip(task, device_list):

        print(f"process number {len(device_list)}")
        process_pool = Pool(len(device_list))
        re_lis = []
        for i, index in enumerate(task_list):
            re_lis.append(process_pool.apply_async(run, (src_dir, dataset, index, lock_file_name)))
        process_pool.close()
        process_pool.join()
        re_lis = [res.get() for res in re_lis]

def run(src_dir, dataset, index, lock_file_name):
    print(f"task {src_dir} {dataset} {index} start")
    with device_lock:
        with open(lock_file_name,'r') as f:
            line = f.read()
        line = [int(item) for item in line.split(",")]
        print(f"task {src_dir} {dataset} {index} see device {line}")
        assert len(line)>0
        device = line[0]
        with open(lock_file_name,'w') as f:
            print(",".join([str(item) for item in line[1:]]), file=f)

    print(f"task {src_dir} {dataset} {index} using device {device}")
    state = os.system(f"CUDA_VISIBLE_DEVICES={device} PYTHONUTF8=1 python3 ./local/compute_fbank_vietASR_ssl_splits.py run --src-dir {src_dir} --dataset {dataset} --num-workers 2 --start {index} --stop {index+1} --batch-duration 1000 --num-splits {index+1} 2>&1 | tee log_tem/{dataset}_{index}.log")
    # state = 0
    with device_lock:
        with open(lock_file_name,'r') as f:
            line = f.read().strip()
        if len(line)>0:
            line = [int(item) for item in line.split(",")]
        else:
            line = []
        line.append(device)
        with open(lock_file_name,'w') as f:
            print(",".join([str(item) for item in line]), file=f)
    print(f"task {src_dir} {dataset} {index} finish")
    return state


if __name__ == "__main__":
    main()
