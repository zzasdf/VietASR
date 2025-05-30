#! /usr/bin/bash
# export CUDA_VISIBLE_DEVICES="1,2,3,5"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH

python zipformer/train_asr.py \
    --world-size 8 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --bpe-model data/unigram_5000.model \
    --train-cut data/asr/mgb2_cuts_train_0.jsonl.gz \
    --valid-cut data/asr/mgb2_cuts_dev.jsonl.gz \
    --exp-dir exp/ifnt-pretrain \
    --model-type IFNT \
    --max-duration 50 \
    --accum-grad 3 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --seed 1332 \
    --master-port 12356 \
    --base-lr 0.005 \
    --lr-epochs 100 \
    "${@}"  # pass remaining arguments

# ./scripts/pretrain_with_encoder.sh --encoder-path ../ASR/zipformer/asr-100h/epoch-60.pt --freeze-encoder True 
# manually stop after 30 epochs and then run:
# ./scripts/pretrain_with_encoder.sh --warmup-encoder-step 300 \
