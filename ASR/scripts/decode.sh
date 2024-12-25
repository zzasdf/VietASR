#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3
export PYTHONPATH=/workdir/work/icefall:$PYTHONPATH

# export PYTHONPATH=/old_workdir/work/icefall-gigaspeech2:${PYTHONPATH}
bpe_model=lang_bpe_500
run_time=$(date +"%Y-%m-%d-%H-%M-%S")
log_path=zipformer/exp-zipformer-50h_phone2/log_decode
mkdir -p ${log_path}

python ./zipformer/decode.py \
    --epoch $1 \
    --avg $2 \
    --exp-dir zipformer/exp-zipformer-50h_phone2 \
    --max-duration 1000 \
    --bpe-model data/ssl_finetune/vie_phonetisaurus_bpe_500/bpe.model \
    --decoding-method greedy_search \
    --manifest-dir data/ssl_finetune \
    --use-averaged-model 0 \
    --cuts-name phone2

