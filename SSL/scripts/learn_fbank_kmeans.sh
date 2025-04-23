#!/bin/bash

python local/simple_kmeans_lhotse/learn_kmeans_lhotse.py \
    exp/simple_kmeans/kmeans_170h.pt 500 \
    --max_iter 100 \
    --src_dir data/kmeans_170h \
    --do_training
