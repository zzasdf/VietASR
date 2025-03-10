# VietASR

VietASR is a training pipeline designed for low resource ASR. It uses ASR-biased SSL to pretrain strong speech encoder with limited labeled data and large-scale unlabeled data. 

## Note
* Change /workdir/work/icefall to the path to you icefall path.

## data preparation
VietASR relies on [lhotse](https://github.com/lhotse-speech/lhotse) for data pretreatment.
This part we extract fbank feature from unsupervised audio segment. Check ./SSL/prepare_ssl.sh for details.
```bash
cd SSL
./prepare_ssl.sh
```

## extract kmeans
This part will train the kmeans model for hubert iter1, iter2, ASR base SSL, and extract kmeans label from source cuts.
### fbank kmeans
At first we will train the kmeans model on a small cuts (about 100h hours). Update the cuts path in SSL/scripts/learn_fbank_kmeans.sh, and then run
```bash
cd SSL
./scripts/learn_fbank_kmeans.sh
```
This part will read feature from source cuts and save to test cut, assume the source cuts paths and the target cuts paths are stored in a file(```--task-list``` in the script) in the following form:
```
src_cut_path1 target_cut_path1
src_cut_path2 target_cut_path2
```
Update the path in SSL/scripts/extract_fbank_kmeans.sh, and then run
```bash
cd SSL
./scripts/extract_fbank_kmeans.sh
```
### hubert iteration2 kmeans
See [fbank kmeans](#fbank-kmeans) for data preparation.

Train kmeans model.
```bash
cd SSL
./scripts/learn_ssl_kmeans.sh
```
Extract kmeans label.
```bash
cd SSL
./scripts/extract_ssl_kmeans.sh
```
### ASR model kmeans
See [fbank kmeans](#fbank-kmeans) for data preparation.

Train kmeans model.
```bash
cd ASR
./scripts/learn_ASR_kmeans.sh
```
Extract kmeans label.
```bash
cd ASR
./scripts/extract_ASR_kmeans.sh
```
### ASR model kmeans iter2
See [fbank kmeans](#fbank-kmeans) for data preparation.

This iteration will load the finetuned model from ASR label pretraining iteration 1. In the following scripts, ```--pretrained-dir``` means the path to the pretrained checkpoint, ```--exp-dir``` is the path to the finetune exp_dir, ```--model-path``` is the path to kmeans model.

Train kmeans model.
```bash
cd SSL
./scripts/learn_finetune_kmeans.sh
```
Extract kmeans label.
```bash
cd SSL
./scripts/extract_finetune_kmeans.sh
```

## pretrain
```bash
cd SSL
./scripts/run_ssl_fbank.sh
```
## finetune
```bash
cd SSL
./scripts/finetune_tri_stage.sh
```
## decode
```bash
cd SSL
./scripts/decode.sh $epoch $avg $gpu_id
```