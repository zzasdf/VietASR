#! /usr/bin/bash
# export CUDA_VISIBLE_DEVICES="1,2,3,5"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH

python zipformer/train_asr.py \
    --world-size 4 \
    --num-epochs 50 \
    --start-epoch 1 \
    --use-fp16 0 \
    --bpe-model data/unigram_5000.model \
    --model-type IFNT \
    --exp-dir exp/ifnt/adapt \
    --load-path exp/ifnt/epoch-250.pt \
    --train-stage adapt \
    --adapt-dir data/adapt \
    --max-duration 500 \
    --accum-steps 2 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356 \
    --base-lr 0.005 \
    "${@}"  # pass remaining arguments
