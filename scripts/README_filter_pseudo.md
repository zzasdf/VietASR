# Hướng dẫn Chạy Pipeline Lọc Pseudo-label

## Tổng quan
Pipeline này lọc dữ liệu pseudo-label qua nhiều giai đoạn để lấy ra các sample chất lượng cao nhất.

## Yêu cầu
- Docker container đang chạy (ID: `02ff3f6f9ff9`)
- Dữ liệu pseudo-label tại `/data/raw/pseudo/call_center/transcripts.txt`

## Cách chạy

### Bước 1: Vào Docker container
```bash
docker exec -it 02ff3f6f9ff9 bash
cd /vietasr
```

### Bước 2: Chạy Stage 1-2 (Heuristics & Repetition Filter)
```bash
# Tạo thư mục output
mkdir -p /data/raw/pseudo/call_center/filtered

# Chạy lọc
python scripts/filter_stage1_2_heuristics.py \
    --input /data/raw/pseudo/call_center/transcripts.txt \
    --output /data/raw/pseudo/call_center/filtered/stage1_2.txt \
    --stats-output /data/raw/pseudo/call_center/filtered/stage1_2_stats.json
```

**Thời gian ước tính**: ~2-5 phút cho 500k samples

**Kết quả**:
- `stage1_2.txt`: Các sample đã qua lọc
- `stage1_2_stats.json`: Thống kê chi tiết

### Bước 3: Chạy Stage 3 (Language Model Scoring)
```bash
python scripts/filter_stage3_lm_score.py \
    --input /data/raw/pseudo/call_center/filtered/stage1_2.txt \
    --lm-path /vietasr/data4/lm/lm_4gram.arpa \
    --output /data/raw/pseudo/call_center/filtered/stage3.txt
```

**Thời gian ước tính**: ~2-3 phút

### Hoặc: Chạy tất cả bằng 1 lệnh
```bash
bash scripts/filter_pseudo_sota.sh
```

## Kiểm tra kết quả
```bash
# Xem số lượng sample sau mỗi stage
wc -l /data/raw/pseudo/call_center/transcripts.txt
wc -l /data/raw/pseudo/call_center/filtered/stage1_2.txt
wc -l /data/raw/pseudo/call_center/filtered/stage3.txt

# Xem thống kê lọc
cat /data/raw/pseudo/call_center/filtered/stage1_2_stats.json
```

## Tùy chỉnh ngưỡng lọc
```bash
python scripts/filter_stage1_2_heuristics.py \
    --input /data/raw/pseudo/call_center/transcripts.txt \
    --output /data/raw/pseudo/call_center/filtered/stage1_2.txt \
    --min-duration 1.5 \      # Thay đổi độ dài tối thiểu
    --max-duration 30.0 \     # Thay đổi độ dài tối đa
    --min-words 2 \           # Thay đổi số từ tối thiểu
    --max-words 60            # Thay đổi số từ tối đa
```

## Lưu ý quan trọng
⚠️ **Dữ liệu gốc KHÔNG bị xóa**. Tất cả output được lưu vào thư mục `filtered/` riêng biệt.
