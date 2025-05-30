#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
# export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH


python ./zipformer/decode.py \
    --epoch 100 \
    --avg 5 \
    --use-averaged-model 1 \
    --exp-dir exp/rnnt \
    --max-duration 300 \
    --bpe-model data/unigram_5000.model \
    --decoding-method greedy_search \
    --manifest-dir ../FNT/data/devtest \
    --device 0 \
    "${@}"
