# ASR Model Downloader

Script này dùng để liệt kê và tải về các mô hình Nhận dạng giọng nói tự động (ASR) từ một file `asr_models.json` định nghĩa các mô hình và đường dẫn tải tương ứng.

## 📁 Cấu trúc

- `api/asr_models.json`: JSON chứa thông tin các mô hình ASR và link download các file.
- `api/download_model.py`: Script Python để liệt kê và tải mô hình.
- Các model trong `asr_models.json` được dẫn link từ folder: [Model](https://sync.admicro.vn/library/78595b0e-a020-4119-b0da-6fa8d12217f7/ASR/MODEL_FILES/AM)

## 🚀 Cách sử dụng

### 1. Chạy liệt kê mô hình (chỉ hiển thị, không tải):

```bash
cd api/
python3 download_model.py --no_download True
```

### 2. Tải mô hình theo tên:

```bash
cd api/
python3 download_model.py --model_name "model_name_here"
```

### 3. Tải mô hình theo chỉ số:

```bash
cd api/
python3 download_model.py --model_name 1
```

> Trong đó `1` là chỉ số mô hình hiển thị khi chạy `--no_download True`.

### 4. Chỉ định thư mục lưu:

```bash
cd api/
python3 download_model.py --model_name "model_name_here" --save_dir "./saved_models"
```

## 📥 Ví dụ `asr_models.json`

```json
{
  "conformer_small": {
    "model": {
      "config.yaml": "https://yourdomain.com/config.yaml",
      "model.pt": "https://yourdomain.com/model.pt"
    },
    "description": "Small Conformer model for ASR tasks"
  },
  "hubert_large": {
    "model": {
      "config.yaml": "https://yourdomain.com/hubert/config.yaml",
      "model.pt": "https://yourdomain.com/hubert/model.pt"
    },
    "description": "Large HuBERT model pretrained on 960h"
  }
}
```

## 🔧 Tham số dòng lệnh

| Tham số          | Kiểu    | Mặc định      | Mô tả                                                            |
| ----------------- | -------- | ---------------- | ------------------------------------------------------------------ |
| `--save_dir`    | `str`  | `"model_file"` | Thư mục để lưu mô hình tải về                             |
| `--model_name`  | `str`  | `""`           | Tên hoặc chỉ số của mô hình cần tải                       |
| `--no_download` | `bool` | `False`        | Nếu `True`, chỉ hiển thị danh sách mô hình mà không tả |

## 📦 Yêu cầu

- Python 3.x
- `wget` đã cài sẵn trong hệ thống

## 📜 Ghi chú

- Script sử dụng `os.system("wget ...")` để tải file, bạn cần đảm bảo `wget` khả dụng trong môi trường shell.
- Phần kenlm (trong file `asr_models.json` là `lm`) được dùng chung cho tất cả api nên đã lược bỏ để đỡ tốn disk

  - hãy tải riêng với lệnh sau

  ```
  cd model_file
  wget https://sync.admicro.vn/f/cfe7c9b423c441bb92da/?dl=1 -O lm
  ```
