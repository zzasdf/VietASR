#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# change --label to the train label you want to use
# change --label-rate to 100 when you use the k-means of Fbank as target, in other cases it should be 50
# change --exp-dir to the path you want to save checkpoints

python zipformer_fbank/pretrain.py \
    --world-size 4 \
    --num-epochs 300 \
    --start-epoch 1 \
    --use-fp16 1 \
    --manifest-prefix alg100k \
    --label-type iter1 \
    --label-rate 50 \
    --sample-rate 100 \
    --exp-dir zipformer_fbank/exp_iter2 \
    --max-duration 2400 \
    --accum-grad 1 \
    --min-keep-size 200 \
    --max-sample-size 1562 \
    --mask-before-cnn 1 \
    --mask-prob 0.80 \
    --dropout-input 0.1 \
    --dropout-features 0.1 \
    --base-lr 0.045 \
    --save-every-n 100000 \
    --master-port 12356

for i in {0..7}; do CUDA_VISIBLE_DEVICES=$i python ~/busygpu/run.py & done
