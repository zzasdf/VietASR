import json
import gzip
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir")
    args = parser.parse_args()

    src_dir = args.src_dir
    lis = os.listdir(src_dir)

    tgt_dict = dict()
    for item in lis:
        item_split = item.split("_", maxsplit=3)
        if len(item_split)>=4 and item_split[1]=='tgt':
            if item_split[3] not in tgt_dict:
                tgt_dict[item_split[3]] = []
            tgt_dict[item_split[3]].append(item)
    for item in tgt_dict:
        with gzip.open(os.path.join(src_dir, item), 'wt') as f:
            lines = []
            for sub_item in tgt_dict[item]:
                with gzip.open(os.path.join(src_dir, sub_item), 'rt') as g:
                    sub_lines = g.read().splitlines()
                lines.extend(sub_lines)
            for line in lines:
                print(line, file=f)




