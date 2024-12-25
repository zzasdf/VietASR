#!/bin/bash

export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH

python zipformer/learn_kmeans.py \
    --km-path tem_data/kmeans.pt \
    --n-clusters 500 \
    --max-iter 100 \
    --files data/kmeans_manifest_from_2000h.jsonl.gz \
    --do-training \
    --max-duration 1000 \
    --checkpoint-type ASR \
    --exp-dir ./zipformer/exp-zipformer-50h \
    --epoch 246 \
    --avg 5 \
    --use-averaged-model 0 \
    --bpe-model data/ssl_finetune/vie_phonetisaurus_bpe_500/bpe.model \
