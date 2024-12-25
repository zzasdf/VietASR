#!/bin/bash
export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH
export PYTHONPATH=/workdir/work/icefall/egs/tencent/SSL/zipformer_fbank:$PYTHONPATH
python -m zipformer_fbank.extract_kmeans_scripts.learn_kmeans \
    --km-path tem_data/kmeans_100h_kmeans_ASR_50h_epoch3.pt \
    --n-clusters 500 \
    --max-iter 100 \
    --files data/kmeans_manifest.jsonl.gz \
    --do-training \
    --pretrained-dir zipformer_fbank/exp-kmeans_ASR_50h-all/epoch-3.pt \
    --exp-dir zipformer_fbank/exp-kmeans_ASR_50h-all/exp-epoch-3-tri-stage-50h \
    --epoch 195 \
    --avg 1 \
    --max-duration 1000 \
    --mask-before-cnn 1 \
    --checkpoint-type finetune \
    --use-averaged-model 0 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model