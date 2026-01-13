export CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH=${PWD}/SSL/zipformer_fbank:$PYTHONPATH

python SSL/zipformer_fbank/finetune.py \
    --world-size 2 \
    --master-port 12357 \
    --num-epochs 30 \
    --start-epoch 7 \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --manifest-dir data/fbank_finetune \
    --base-lr 0.0045 \
    --scheduler-type tri_stage \
    --max-lr-update 80000 \
    --phase-ratio "0.1,0.4,0.5" \
    --max-duration 800 \
    --accum-grad 2 \
    --enable-spec-aug True \
    --enable-musan False \
    --warmup-encoder-step 0 \
    --freeze-encoder-step 0 \
    --use-fp16 1 \
    --exp-dir SSL/zipformer_fbank/exp_finetune \
    --save-every-n 5000 \
    --tensorboard True
