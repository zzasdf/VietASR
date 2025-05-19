#! /usr/bin/bash
# export CUDA_VISIBLE_DEVICES="1,2,3,5"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH

python zipformer/train_asr.py \
    --world-size 4 \
    --num-epochs 100 \
    --start-epoch 1 \
    --use-fp16 0 \
    --bpe-model data/unigram_5000.model \
    --exp-dir exp/fnt-100h/adapt-100 \
    --load-path exp/fnt-100h/epoch-100.pt \
    --train-stage adapt \
    --max-duration 300 \
    --accum-steps 5 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356 \
    --base-lr 0.05 \
    "${@}"  # pass remaining arguments
