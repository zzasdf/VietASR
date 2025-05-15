#! /usr/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH=${PWD}/zipformer:$PYTHONPATH
export LD_LIBRARY_PATH=/hpc_stor03/sjtu_home/junzhe.liu/anaconda/envs/icefall/lib:$LD_LIBRARY_PATH

./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/mgb2_cuts_dev.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/commonvoice_ar_cuts_dev.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/commonvoice_ar_cuts_test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/dataocean_cuts_algeria-test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/dataocean_cuts_iraq-test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/sada_cuts_dev.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/sada_cuts_test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/masc_cuts_algeria.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/masc_cuts_iraq.jsonl.gz 
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/masc_cuts_morocco.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/masc_cuts_saudi.jsonl.gz
./scripts/ppl.sh --exp-dir exp/fnt-100h/old --model-type FNT --test-cut data/devtest/masc_cuts_MSA.jsonl.gz 

./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/mgb2_cuts_dev.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/commonvoice_ar_cuts_dev.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/commonvoice_ar_cuts_test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/dataocean_cuts_algeria-test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/dataocean_cuts_iraq-test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/sada_cuts_dev.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/sada_cuts_test.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/masc_cuts_algeria.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/masc_cuts_iraq.jsonl.gz 
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/masc_cuts_morocco.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/masc_cuts_saudi.jsonl.gz
./scripts/ppl.sh --exp-dir exp/ifnt-100h/old --model-type IFNT --test-cut data/devtest/masc_cuts_MSA.jsonl.gz 