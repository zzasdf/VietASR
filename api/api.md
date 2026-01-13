# 🧠 ASR FastAPI Server

Dịch vụ **ASR (Automatic Speech Recognition)** sử dụng FastAPI, hỗ trợ nhiều kiến trúc mô hình và định dạng khác nhau như **PyTorch, ONNX, OpenVINO, Dolphin, Transducer**, phục vụ nhận dạng giọng nói từ audio đầu vào (từ URL, file hoặc raw bytes).

---

## 🚀 Tính năng

- ✅ Nhận dạng giọng nói với các kiến trúc:
  - `ASR tiêu chuẩn (PyTorch)`
  - `Dolphin (Torch/ONNX)`
  - `Transducer`
  - `ONNX`
  - `OpenVINO`
- ✅ Tự động phân đoạn âm thanh với `auditok`
- ✅ Chuẩn hóa văn bản đầu ra (`text_norm`)
- ✅ Hỗ trợ beam search và LM decoding
- ✅ Tùy chọn chia nhỏ âm thanh dài > 10s
- ✅ Hỗ trợ nhiều kiểu request: JSON, form-data, raw bytes

---

## 🧩 Cài đặt

```bash
pip install -r requirements.txt
```

> Lưu ý: Yêu cầu thêm `ffmpeg`, `sox`, `auditok`, `uvicorn`, `espnet`, `onnxruntime`, `openvino`, `loguru`, v.v.

---

## 🛠️ Chạy Server

```bash
python api.py \
    --model_dir /path/to/model_dir \
    --device cuda \
    --port 5000
```

### Một số tùy chọn:

- `--model_dir` (**bắt buộc**): Đường dẫn chứa mô hình
- `--ext_model_dir`: Đường dẫn mô hình phụ (ensemble)
- `--device`: `cpu` hoặc `cuda` (mặc định `cpu`)
- `--port`: Cổng API, mặc định `5000`
- `--kenlm_alpha`, `--kenlm_beta`: Tham số cho LM decoding
- `--word_vocab_size`: Kích thước từ điển word-based (mặc định -1)

---

## 📤 Phản hồi API

```json
{
  "status": 1,
  "code": 200,
  "message": "process file success",
  "data": {
    "model_version": "asr_model_name",
    "result": [
      {
        "start": "0",
        "end": "4.8",
        "text": "xin chào bạn đang nghe dịch vụ chuyển giọng nói thành văn bản",
        "segments": [
          {"start": 0.2, "end": 1.5, "text": "xin chào bạn"},
          ...
        ]
      }
    ],
    "duration": "9.60",
    "infer_time": "1.45 s",
    "beam_size": 5
  }
}
```

---

## 📁 Cấu trúc thư mục model

```
model_dir/
├── config
├── model
├── feat_normalize
├── bpe_model
├── word_vocab
├── (tùy chọn) encoder / decoder / ctc / text_normalize / lm
```

---

## ✅ Ghi chú

- Token mặc định có thể chỉnh sửa trong mã nguồn (`PRIVATE_TOKEN`)
- Server sẽ xóa file sau khi xử lý
- Audio có thể là `.wav`, `.mp3`, `.m4a`, nhờ `pydub` chuyển đổi
