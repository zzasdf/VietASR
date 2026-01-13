# Hướng Dẫn Phân Tích ASR theo Duration

## Tổng Quan

Script `analyze_by_duration.py` giúp phân tích **WER** và **Latency (RTF)** theo 6 nhóm độ dài audio:

| Duration Bin | Mô Tả |
|--------------|-------|
| 0-4s | Câu ngắn, ít từ |
| 4-8s | Câu trung bình |
| 8-12s | Câu dài |
| 12-16s | Câu rất dài |
| 16-20s | Đoạn hội thoại |
| 20s+ | Đoạn dài, nhiều người nói |

---

## Cấu Trúc Dữ Liệu Test

### Format thư mục test:
```
/data/raw/test/<dataset_name>/
├── wavs/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── transcripts.txt
```

### Format file `transcripts.txt`:
```
/full/path/to/audio.wav|transcript text tiếng việt|duration_in_seconds
```

**Ví dụ thực tế:**
```
/data/raw/test/tongdai_clean/wavs/11_1_2022_tong_dai_segment_92_70.wav|ờ thì em gọi điện lại để tư vấn và hỗ trợ chị tốt hơn ạ|3.2
/data/raw/test/tongdai_clean/wavs/11_1_2022_tong_dai_segment_234_59.wav|dạ không biết là anh làm cái web này là mục đích là để chỉ để giới thiệu cái sản phẩm|4.899
```

**Delimiter:** `|` (pipe)
**Fields:** audio_path | text | duration

---

## 5 Test Sets

| # | Dataset | Docker Path |
|---|---------|-------------|
| 1 | tongdai_clean | `/data/raw/test/tongdai_clean` |
| 2 | call_center_private | `/data/raw/test/call_center_private` |
| 3 | tongdai_other_reviewed | `/data/raw/test/tongdai_other_reviewed` |
| 4 | regions.dong_nam_bo | `/data/raw/test/regions.dong_nam_bo` |
| 5 | regions.bac_trung_bo | `/data/raw/test/regions.bac_trung_bo` |

---

## Cách Chạy

### 1. Chạy với 1 dataset:

```bash
# Trong Docker container
cd /vietasr

export CUDA_VISIBLE_DEVICES=0  # Chọn GPU

python scripts/analyze_by_duration.py \
    --test-dir /data/raw/test/tongdai_clean \
    --exp-dir /vietasr/data4/exp_finetune_v3 \
    --epoch 19 \
    --avg 17 \
    --bpe-model /vietasr/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
    --beam-size 10 \
    --method modified_beam_search \
    --output-dir /vietasr/results/duration_analysis \
    --num-warmup 10
```

### 2. Chạy tất cả 5 datasets:

```bash
bash scripts/run_duration_analysis.sh
```

---

## Output

### 1. Console output (bảng tóm tắt):
```
==========================================================================================
ANALYSIS RESULTS: tongdai_clean
==========================================================================================
Duration Bin |    Count |  Avg Dur |    Avg Lat |      RTF |  WER (%)
------------------------------------------------------------------------------------------
0-4s         |      234 |     2.8s |    0.0542s |   0.0194 |     4.23
4-8s         |      456 |     5.9s |    0.0987s |   0.0167 |     3.87
8-12s        |      123 |     9.4s |    0.1523s |   0.0162 |     4.56
...
==========================================================================================
```

### 2. Files được tạo:

| File | Mô Tả |
|------|-------|
| `<dataset>_summary.json` | Thống kê tổng hợp theo bin |
| `<dataset>_results.csv` | Kết quả chi tiết từng sample (dùng cho cross-repo comparison) |
| `<dataset>_detailed.txt` | REF/HYP từng sample |

### 3. CSV Format (chuẩn để so sánh):
```csv
audio_path,duration_sec,duration_bin,reference,hypothesis,latency_sec,rtf
"/data/test/audio1.wav",3.45,"0-4s","xin chào","xin chào",0.0823,0.0239
```

---

## So Sánh 2 Models (Cross-Repo)

### Workflow:

```
VietASR Repo                        Team's Repo
     │                                  │
     ▼                                  ▼
analyze_by_duration.py        (script tương tự tạo CSV)
     │                                  │
     ▼                                  ▼
vietasr_results.csv           team_results.csv
     │                                  │
     └─────────────┬────────────────────┘
                   ▼
          compare_results.py
                   │
                   ▼
          Comparison Report
```

### Chạy so sánh:

```bash
python scripts/compare_results.py \
    --model-a-csv /vietasr/results/duration_analysis/tongdai_clean_results.csv \
    --model-a-name "Zipformer_VietASR" \
    --model-b-csv /path/to/team/tongdai_clean_results.csv \
    --model-b-name "Team_API" \
    --output-dir /vietasr/results/comparison
```

### CSV Format chuẩn (Team cần export giống):

```csv
audio_path,duration_sec,duration_bin,reference,hypothesis,latency_sec,rtf
```

---

## Điều Kiện So Sánh Công Bằng

| Setting | Phải Giống Nhau |
|---------|-----------------|
| GPU | Cùng loại (VD: RTX 3090) |
| Batch Size | 1 (single audio) |
| Beam Size | 10 |
| Decoding Method | modified_beam_search |
| Warmup | 10 samples trước khi đo |
| Audio Files | Cùng files test |

---

## Các Tham Số Quan Trọng

| Tham Số | Default | Mô Tả |
|---------|---------|-------|
| `--test-dir` | (required) | Thư mục chứa wavs/ và transcripts.txt |
| `--exp-dir` | data4/exp_finetune_v3 | Thư mục chứa checkpoints |
| `--epoch` | 19 | Epoch checkpoint |
| `--avg` | 17 | Số epochs để average |
| `--beam-size` | 10 | Beam search size |
| `--method` | modified_beam_search | Decoding method |
| `--num-warmup` | 5 | Số samples warmup trước khi đo |

---

## Troubleshooting

### Lỗi "No samples found":
- Kiểm tra file `transcripts.txt` có tồn tại không
- Đảm bảo format là `path|text|duration` với delimiter `|`

### Lỗi "Audio not found":
- Paths trong transcripts.txt phải là absolute path hợp lệ
- Hoặc file phải nằm trong thư mục `wavs/`

### Latency cao bất thường:
- Chạy warmup đủ số samples (--num-warmup 10)
- Kiểm tra GPU không bị share với process khác
- Dùng `CUDA_VISIBLE_DEVICES` để lock GPU
