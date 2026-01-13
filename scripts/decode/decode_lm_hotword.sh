#!/bin/bash

epoch=10
avg=9
exp_dir=data4/exp
lang_dir=data/lang_bpe_500
bpe_model=viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model
manifest_dir=data3/fbank

# Hotword file
hotword_file=data4/hotword.txt

# LM file
arpa_lm=data4/lm/lm_4gram.arpa
arpa_scale=0.5
context_score=2.0

echo "Decoding with LM + Hotword..."
echo "LM: $arpa_lm (scale $arpa_scale)"
echo "Hotword: $hotword_file (score $context_score)"

# Ensure hotword file exists
if [ ! -f "$hotword_file" ]; then
    echo "Creating sample hotword file at $hotword_file"
    echo "alo" > $hotword_file
    echo "ngân hàng" >> $hotword_file
fi

# Run decoding
# Note: We use modified_beam_search_lm_shallow_fusion method which we patched to support hotwords
# when context-graph is provided.

./SSL/zipformer_fbank/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir $exp_dir \
  --bpe-model $bpe_model \
  --max-duration 1000 \
  --decoding-method modified_beam_search_lm_shallow_fusion \
  --manifest-dir $manifest_dir \
  --context-file $hotword_file \
  --context-score $context_score \
  --arpa-lm-path $arpa_lm \
  --arpa-lm-scale $arpa_scale \
  --beam-size 10 \
  --use-averaged-model True \
  --use-layer-norm 0 \
  --final-downsample 1 \
  --cuts-name test

echo "Done."
