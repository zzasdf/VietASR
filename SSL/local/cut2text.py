import gzip
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--tgt", type=str)
args = parser.parse_args()

with gzip.open(args.src, "rt") as f:
    lines = f.read().splitlines()
lines = [json.loads(line) for line in lines]

with open(args.tgt, "w") as f:
    for line in lines:
        print(line["supervisions"][0]["text"], file=f)
