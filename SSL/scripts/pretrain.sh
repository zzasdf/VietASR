cd /vietasr

export CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH=${PWD}/SSL/zipformer_fbank:$PYTHONPATH

python SSL/zipformer_fbank/pretrain.py \
    --world-size 2 \
    --num-epochs 30 \
    --start-epoch 21 \
    --use-fp16 1 \
    --label-type kmeans \
    --label-rate 50 \
    --sample-rate 100 \
    --exp-dir SSL/zipformer_fbank/exp \
    --max-duration 500 \
    --train-cut large \
    --accum-grad 4 \
    --min-keep-size 200 \
    --mask-before-cnn 1 \
    --max-sample-size 1562 \
    --mask-prob 0.80 \
    --dropout-input 0.1 \
    --dropout-features 0.1 \
    --base-lr 0.045 \
    --save-every-n 5000 \
    --master-port 12356 \
    --train-manifest data/vad_lhotse/cuts_train.jsonl.gz \
    --dev-manifest data/vad_lhotse/cuts_dev.jsonl.gz