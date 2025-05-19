#! /usr/bin/bash
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH

python zipformer/ppl.py \
    --epoch 60 \
    --avg 5 \
    --bpe-model data/unigram_5000.model \
    --exp-dir exp/fnt-100h \
    --max-duration 100 \
    --model-type FNT \
    --test-cut data/devtest \
    --device 7 \
    "${@}"  # pass remaining arguments
