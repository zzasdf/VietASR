import gzip
import json
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--subset-name", type=str, default="train")
    parser.add_argument("--num-splits", type=int, default=16)
    parser.add_argument(
        "--target-duration",
        type=float,
        default=100,
        help="Duration for target cut in hours",
    )
    parser.add_argument(
        "--target-path",
        type=float,
        default=100,
        help="Duration for target cut in hours",
    )

    args = parser.parse_args()
    subset_name = args.subset_name
    num_splits = args.num_splits
    target_duration = args.target_duration * 3600
    src_dir = f"data/ssl_{subset_name}/{subset_name}_split"
    target_path = args.target_path
    num_digits = 8  # num_digits is fixed by lhotse split-lazy

    target_lines = []
    total_duration = 0
    for i, index in range(num_splits):
        idx = f"{i}".zfill(num_digits)
        cuts_path = os.path.join(
            src_dir, f"vietASR-ssl_cuts_{subset_name}.{idx}.jsonl.gz"
        )
        with gzip.open(cuts_path, "rt") as f:
            lines = f.read().splitlines()
        for line in lines:
            data = json.loads(line)
            total_duration += data["duration"]
            if total_duration > target_duration:
                break
            target_lines.append(line)
        if total_duration > target_duration:
            break

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with gzip.open(target_path, "wt") as f:
        for line in target_lines:
            print(line, file=f)
