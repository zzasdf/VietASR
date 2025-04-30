#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH

python zipformer/train_asr.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --bpe-model data/unigram_5000.model \
    --exp-dir exp/fnt-100h \
    --max-duration 100 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356 \
    "${@}"  # pass remaining arguments
