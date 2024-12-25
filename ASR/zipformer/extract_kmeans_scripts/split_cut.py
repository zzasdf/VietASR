import json
import gzip
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-list")
    parser.add_argument("--split-n", type=int)
    args = parser.parse_args()
    task_list = os.path.abspath(args.task_list)
    with open(task_list, 'r') as f:
        lines = f.read().splitlines()
    tasks = [line.split() for line in lines]

    split_n = args.split_n

    rand_index = os.urandom(8).hex()
    task_record = [[] for _ in range(split_n)]
    for src, tgt in tasks:
        with gzip.open(src, 'rt') as f:
            lines = f.read().splitlines()
        tgt_dir = os.path.dirname(tgt)
        os.makedirs(tgt_dir, exist_ok=True)
        tgt_name = os.path.basename(tgt)
        for i in range(split_n):
            with gzip.open(os.path.join(tgt_dir, f"{rand_index}_src_{i}_{tgt_name}"), 'wt') as f:
                for line in lines[i::split_n]:
                    print(line, file=f)
            task_record[i].append("{} {}".format(
                os.path.join(tgt_dir, f"{rand_index}_src_{i}_{tgt_name}"), 
                os.path.join(tgt_dir, f"{rand_index}_tgt_{i}_{tgt_name}")))
    
    for i in range(split_n):
        with open(f"{task_list}_{i}", 'w') as f:
            for task in task_record[i]:
                print(task, file=f)

