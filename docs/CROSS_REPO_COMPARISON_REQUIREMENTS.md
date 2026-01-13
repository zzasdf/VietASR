# Yêu Cầu So Sánh ASR Models - Cross-Repo Comparison

## Mục Tiêu
So sánh **WER** và **RTF (Latency)** giữa 2 models trên cùng 5 tập test data.

---

## 1. Dữ Liệu Test (5 Tập)

| # | Dataset | Docker Path |
|---|---------|-------------|
| 1 | tongdai_clean | `/data/raw/test/tongdai_clean` |
| 2 | call_center_private | `/data/raw/test/call_center_private` |
| 3 | tongdai_other_reviewed | `/data/raw/test/tongdai_other_reviewed` |
| 4 | regions.dong_nam_bo | `/data/raw/test/regions.dong_nam_bo` |
| 5 | regions.bac_trung_bo | `/data/raw/test/regions.bac_trung_bo` |

### Format Input:
```
transcripts.txt:  /path/to/audio.wav|transcript text|duration_in_seconds
```

---

## 2. Preprocessing Chuẩn (BẮT BUỘC)

### 2.1. Filter Samples
```python
# Skip samples chứa '111'
if '111' in text:
    continue

# Xóa suffix ' ck' và ' spkt'
text = text.replace(' ck', '').replace(' spkt', '')
```

### 2.2. Text Normalization (giống nhau cho cả REF và HYP)
```python
# Lowercase
text = text.lower()

# Xóa filler words (á, à, ờ, ừ, ạ, ...)
# Optional: áp dụng vi2en.txt mappings nếu có
```

---

## 3. Output Yêu Cầu: CSV Format Chuẩn

**Tên file:** `<dataset_name>_results.csv`

**Header:**
```csv
audio_path,duration_sec,duration_bin,reference,hypothesis,latency_sec,rtf
```

**Duration Bins:**
- `0-4s`
- `4-8s`
- `8-12s`
- `12-16s`
- `16-20s`
- `20s+`

**Ví dụ:**
```csv
audio_path,duration_sec,duration_bin,reference,hypothesis,latency_sec,rtf
"/data/test/audio1.wav",3.45,"0-4s","xin chào anh","xin chào anh",0.0823,0.0239
"/data/test/audio2.wav",7.21,"4-8s","em cảm ơn anh ạ","em cảm ơn anh",0.1456,0.0202
```

---

## 4. Điều Kiện So Sánh Công Bằng

| Setting | Yêu Cầu |
|---------|---------|
| **Batch Size** | 1 (single audio) |
| **Warmup** | 10 samples trước khi đo latency |
| **GPU Sync** | `torch.cuda.synchronize()` trước/sau inference |
| **Latency Scope** | Từ feature extraction → decode xong |

---

## 5. Script Mẫu (Python)

```python
import time
import torch
import csv

# Warmup
for i in range(10):
    _ = model.transcribe(warmup_audio)
torch.cuda.synchronize()

# Run inference
results = []
for audio_path, reference in samples:
    duration = get_audio_duration(audio_path)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    hypothesis = model.transcribe(audio_path)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    latency = end - start
    rtf = latency / duration
    duration_bin = get_duration_bin(duration)
    
    results.append({
        'audio_path': audio_path,
        'duration_sec': duration,
        'duration_bin': duration_bin,
        'reference': reference.lower(),
        'hypothesis': hypothesis.lower(),
        'latency_sec': latency,
        'rtf': rtf,
    })

# Save CSV
with open('dataset_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'audio_path', 'duration_sec', 'duration_bin',
        'reference', 'hypothesis', 'latency_sec', 'rtf'
    ])
    writer.writeheader()
    writer.writerows(results)
```

---

## 6. Duration Bin Function

```python
DURATION_BINS = [
    (0, 4, "0-4s"),
    (4, 8, "4-8s"),
    (8, 12, "8-12s"),
    (12, 16, "12-16s"),
    (16, 20, "16-20s"),
    (20, float('inf'), "20s+"),
]

def get_duration_bin(duration):
    for min_d, max_d, name in DURATION_BINS:
        if min_d <= duration < max_d:
            return name
    return "20s+"
```

---

## 7. So Sánh Kết Quả

Sau khi nhận được 5 file CSV, chạy script so sánh:

```bash
python compare_results.py \
    --model-a-csv /path/to/vietasr_tongdai_clean_results.csv \
    --model-a-name "VietASR_Zipformer" \
    --model-b-csv /path/to/team_tongdai_clean_results.csv \
    --model-b-name "Team_Model" \
    --output-dir /path/to/comparison_results
```

---

## 8. Checklist Cho Dev

- [ ] Đọc data từ `transcripts.txt` với format `path|text|duration`
- [ ] Filter bỏ samples có '111'
- [ ] Xóa ' ck' và ' spkt' khỏi text
- [ ] Inference batch_size=1
- [ ] Đo latency với GPU sync
- [ ] Export CSV đúng format
- [ ] Chạy cả 5 datasets
- [ ] Gửi lại 5 file CSV

---

## Liên Hệ

Nếu có thắc mắc về data/format, liên hệ: [Your Contact]
