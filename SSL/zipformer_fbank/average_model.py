#!/usr/bin/env python3
import argparse
import torch
import logging
from pathlib import Path
from icefall.checkpoint import average_checkpoints

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="The epoch to decode.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path to save the averaged checkpoint.",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s")

    device = torch.device("cpu")
    
    if args.avg == 1:
        src = args.exp_dir / f"epoch-{args.epoch}.pt"
        logging.info(f"Copying {src} to {args.dst}")
        checkpoint = torch.load(src, map_location=device)
        torch.save(checkpoint, args.dst)
    else:
        start = args.epoch - args.avg + 1
        filenames = []
        for i in range(start, args.epoch + 1):
            if i >= 1:
                filenames.append(f"{args.exp_dir}/epoch-{i}.pt")
        logging.info(f"Averaging {filenames}")
        avg_checkpoint = average_checkpoints(filenames, device=device)
        torch.save(avg_checkpoint, args.dst)
        logging.info(f"Saved averaged checkpoint to {args.dst}")

if __name__ == "__main__":
    main()
