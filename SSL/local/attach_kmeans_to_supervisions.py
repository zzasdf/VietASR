import os

import gzip
import json
from tqdm import tqdm

# os.system(
#     "cp /userhome/user/yfy62/librispeech_data/data4ssl/manifests/librispeech_*_dev-clean* ."
# )
# os.system(
#     "cp /userhome/user/yfy62/librispeech_data/data4ssl/manifests/librispeech_*_train* ."
# )
# os.system("chmod -R 644 *.jsonl.gz")
# os.system("gunzip *.gz")

dataset_parts = (
    "dev-clean",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "train-all-shuf",
)

kmeans_dir = "/workdir/work/kmeans/k500"
idx_dir = "/workdir/work/kmeans/shu"

kmeans = []
idxs = []
for part in ["train", "valid"]:
    with open(kmeans_dir + "/" + part + ".km", "r") as f:
        kmeans += f.read().splitlines()

    with open(idx_dir + "/" + part + ".tsv", "r") as f:
        lines = f.read().splitlines()
        idxs += [
            line.split("\t", -1)[0].split("/", -1)[-1].replace(".flac", "")
            for line in lines
            if ".flac" in line
        ]

idx2kmeans = {}
for idx, km in zip(idxs, kmeans):
    idx2kmeans[idx] = km

for part in dataset_parts:
    with gzip.open(f"data/wav/librispeech_cuts_{part}.jsonl.gz", "rt") as reader:
        with gzip.open(
            f"data/wav/librispeech_cuts_{part}_ssl.jsonl.gz", mode="wt"
        ) as writer:
            lines = reader.read().splitlines()
            lines = [json.loads(line) for line in lines]
            new_lines = []
            for obj in tqdm(lines):
                obj["custom"] = {"kmeans": idx2kmeans[obj["supervisions"][0]["id"]]}
                new_lines.append(json.dumps(obj))
            writer.write("\n".join(new_lines))

# os.system('for file in *_new.jsonl; do mv "$file" "${file%_new.jsonl}.jsonl"; done')
