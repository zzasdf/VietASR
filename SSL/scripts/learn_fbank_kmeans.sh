#!/bin/bash

python local/simple_kmeans_lhotse/learn_kmeans_lhotse.py \
    kmeans_100h.pt 500 \
    --max_iter 100 \
    --files data/kmeans_manifest.jsonl.gz \
    --do_training