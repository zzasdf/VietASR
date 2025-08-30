#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

for ((epoch=$2; epoch>$2-50; epoch-=10)); do
  for ((avg=5; avg<=$epoch-20; avg+=10)); do
    python ./zipformer_fbank/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir zipformer_fbank/exp_iter2_epoch143avg123_ft_ws4_md2400 \
    --max-duration 2000 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/fbank \
    --final-downsample 1 \
    --cuts-name "all"
  done
done
