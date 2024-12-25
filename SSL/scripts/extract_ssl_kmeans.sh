#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH
export PYTHONPATH=/workdir/work/icefall/egs/tencent/SSL/zipformer_fbank:$PYTHONPATH

python -m zipformer_fbank.zipformer_layer_feature.extract_kmeans_scripts.extract_kmeans run \
    --task-list ${task_file} \
    --model-path ${model_path} \
    --pretrained-dir zipformer_fbank/exp-kmeans-all/epoch-3.pt \
    --max-duration 500 \
    --mask-before-cnn 1 \
    --final-downsample 0 \
    --encoder-feature-layer 3 \
    --bpe-model data/ssl_finetune/Vietnam_bpe_2000_new/bpe.model \
    --old-prefix /userhome/user/jhz00/data/icefall/gigaspeech2_asr/data/fbank \
    --new-prefix /workdir/data/vi/ssl_testset
