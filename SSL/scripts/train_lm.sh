#! /usr/bin/bash

mkdir -p data/lm

python SSL/shared/make_kn_lm.py \
    -ngram-order 4 \
    -text viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/transcript_words.txt \
    -lm data/lm/lm_4gram.arpa
