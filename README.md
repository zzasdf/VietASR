# VietASR - Vietnamese Speech Recognition

> ASR system cho tiếng Việt sử dụng Zipformer + BPE, hỗ trợ SSL pretraining và LM decoding.

## 📁 Cấu trúc Project

```
VietASR/
├── ASR/                          # ASR Training Framework
│   ├── zipformer/                # Zipformer model
│   ├── local/                    # Data preparation
│   └── scripts/                  # Training scripts
│
├── SSL/                          # Self-Supervised Learning
│   ├── zipformer_fbank/          # Main SSL + Finetune code
│   │   ├── finetune.py          # Finetune script
│   │   ├── decode.py            # Decode script
│   │   └── beam_search.py       # Decoding methods
│   ├── shared/                   # Shared utilities (make_kn_lm.py)
│   └── scripts/                  # Training scripts
│
├── scripts/                      # Utility Scripts
│   ├── data/                     # Data preparation
│   │   ├── normalize_*.py       # Text normalization
│   │   └── combine_manifests.py # Manifest processing
│   ├── decode/                   # Decode scripts
│   │   ├── decode.sh            # Baseline decode
│   │   └── decode_with_lm.sh    # Decode với LM
│   ├── lm/                       # Language Model
│   │   ├── 01_tokenize_corpus.sh
│   │   ├── 02_train_lm.sh
│   │   └── 03_decode_with_lm.sh
│   └── *.py                      # Other utilities
│
├── utils/                        # Core Utilities
│   ├── tokenize_corpus_for_lm.py # LM tokenization
│   ├── extract_text_for_lm.py   # Extract text from manifest
│   └── compile_lg.py            # Compile LG graph
│
├── data4/                        # Data & Experiments
│   ├── exp*/                     # Experiment checkpoints
│   ├── fbank/                    # Fbank features
│   ├── lm/                       # Trained LMs
│   └── lm_corpus/                # LM training data
│
├── viet_iter3_pseudo_label/      # Pretrained checkpoint
│   └── data/Vietnam_bpe_2000_new/
│       ├── bpe.model            # BPE model
│       └── tokens.txt           # Token vocabulary
│
└── docker/                       # Docker configuration
```

## 🚀 Quick Start

### 1. Finetune Model

```bash
# Trong Docker container
cd /vietasr

python SSL/zipformer_fbank/finetune.py \
    --world-size 1 \
    --num-epochs 10 \
    --exp-dir data4/exp_finetune \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --manifest-dir data4/fbank \
    --base-lr 0.0003 \
    --use-layer-norm 0
```

### 2. Decode

```bash
# Baseline (không LM)
python SSL/zipformer_fbank/decode.py \
    --epoch 10 --avg 5 \
    --exp-dir data4/exp \
    --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --manifest-dir data4/fbank \
    --decoding-method modified_beam_search \
    --use-layer-norm 0 \
    --cuts-name dev
```

### 3. Train & Decode với LM

```bash
# Step 1: Tokenize corpus
bash scripts/lm/01_tokenize_corpus.sh data4/lm_corpus/all.txt data4/lm_corpus/all_bpe_ids.txt

# Step 2: Train LM
bash scripts/lm/02_train_lm.sh data4/lm_corpus/all_bpe_ids.txt data4/lm/lm_4gram_bpe.arpa 4

# Step 3: Decode với LM
bash scripts/lm/03_decode_with_lm.sh --epoch 10 --avg 5 \
    --lm-path data4/lm/lm_4gram_bpe.arpa --lm-scale 0.3
```

## 📋 Scripts Reference

| Script | Mô tả |
|--------|-------|
| `scripts/lm/01_tokenize_corpus.sh` | Tokenize text → BPE IDs |
| `scripts/lm/02_train_lm.sh` | Train n-gram LM |
| `scripts/lm/03_decode_with_lm.sh` | Decode với LM shallow fusion |
| `scripts/lm/decode_baseline.sh` | Decode baseline |
| `scripts/lm/compare_wer.sh` | So sánh WER |
| `scripts/data/normalize_*.py` | Normalize text |
| `scripts/filter_bad_labels.py` | Filter WER-based |

## ⚙️ Key Parameters

### Decode parameters (cho `viet_iter3_pseudo_label` checkpoint)

```bash
--use-layer-norm 0      # BẮT BUỘC
--final-downsample 1    # Có thể bỏ
--beam-size 10          # Beam size
```

### LM parameters

```bash
--arpa-lm-scale 0.3     # LM weight (tune: 0.1-0.7)
--decoding-method modified_beam_search_lm_shallow_fusion
```

## 📊 Experiments

| Exp Dir | Mô tả |
|---------|-------|
| `data4/exp` | Main experiments |
| `data4/exp_finetune` | Finetune experiments |
| `data4/exp_tongdai` | Tổng đài experiments |

## 🐳 Docker

```bash
# Attach vào container
docker exec -it hiennt_vietasr_gpu_20251121T2348 bash

# Working directory
cd /vietasr
```
