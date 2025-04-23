#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# change --label to the train label you want to use
# change --label-rate to 100 when you use the k-means of Fbank as target, in other cases it should be 50
# change --exp-dir to the path you want to save checkpoints

# modify --num-classes to your kmeans clusters

python zipformer_fbank/pretrain.py \
    --world-size 8 \
    --num-epochs 10 \
    --start-epoch 1 \
    --use-fp16 1 \
    --label-rate 50 \
    --sample-rate 100 \
    --iteration 1 \
    --exp-dir exp/pretrain-iter1 \
    --max-duration 500 \
    --accum-grad 1 \
    --min-keep-size 200 \
    --mask-before-cnn 1 \
    --max-sample-size 1562 \
    --mask-prob 0.80 \
    --dropout-input 0.1 \
    --dropout-features 0.1 \
    --base-lr 0.045 \
    --save-every-n 15000 \
    --master-port 12356 \
    --manifest-dir data/pretrain \
    "${@}"  # pass remaining arguments
    