#! /usr/bin/bash

find data/fbank/*_split/*.jsonl.gz -type f | grep -v "_raw" | while read -r src; do
  suffix=$(echo "$src" | grep -oE '\.0000[0-9]+\.')
  tgt=$(echo "$src" | sed "s|$suffix|_iter1$suffix|")
  echo "$src $tgt"
done

