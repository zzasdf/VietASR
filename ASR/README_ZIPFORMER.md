# Zipformer-30M-RNNT Demo và Fine-tuning

Scripts cho việc demo và fine-tune model Zipformer-30M-RNNT-6000h từ HuggingFace.

## 📋 Tổng quan

Model: [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)
- Architecture: Zipformer (improved Conformer)
- Parameters: ~30M
- Training data: ~6000 giờ tiếng Việt
- Vocabulary: BPE ~2000 tokens

## 🚀 Quick Start

### 1. Download Model

```bash
cd /home/hiennt/VietASR/ASR
bash scripts/download_zipformer_model.sh
```

Model sẽ được download vào: `/home/hiennt/VietASR/models/zipformer-30m-rnnt/`

### 2. Test Demo

```bash
# Cài đặt dependencies
pip install sherpa-onnx soundfile

# Test với audio tự tạo
python test_demo.py

# Hoặc test với audio file có sẵn
python demo_sherpa_onnx.py --audio /path/to/test.wav
```

### 3. Fine-tune Model

#### Bước 1: Chuẩn bị data

```bash
# Tạo Lhotse manifests
python local/prepare_custom_manifest.py \
    --corpus-dir /path/to/your/data \
    --output-dir ../data4/manifests/my_dataset \
    --dataset-name my_dataset

# Compute fbank features
python local/compute_fbank.py \
    --src-dir ../data4/manifests/my_dataset \
    --output-dir ../data4/fbank
```

#### Bước 2: Convert JIT model to pretrained format

```bash
python convert_jit_to_pretrained.py \
    --jit-path ../models/zipformer-30m-rnnt/jit_script.pt \
    --output-path ../models/zipformer-30m-rnnt/pretrained.pt
```

#### Bước 3: Fine-tune

```bash
bash finetune.sh \
    --pretrained-path ../models/zipformer-30m-rnnt/pretrained.pt \
    --exp-dir ../data4/exp_zipformer_finetune \
    --num-epochs 10
```

### 4. Decode với Fine-tuned Model

```bash
bash decode.sh \
    --exp-dir ../data4/exp_zipformer_finetune \
    --epoch 10 \
    --avg 5 \
    --cuts-name test
```

### 5. Export Model

```bash
bash export.sh \
    --exp-dir ../data4/exp_zipformer_finetune \
    --output-dir ../models/zipformer-finetuned \
    --epoch 10 \
    --avg 3 \
    --export-jit 1
```

## 📁 Scripts

| Script | Mô tả |
|--------|-------|
| `scripts/download_zipformer_model.sh` | Download model từ HuggingFace |
| `demo_sherpa_onnx.py` | Demo inference với sherpa-onnx |
| `test_demo.py` | Quick test script |
| `convert_jit_to_pretrained.py` | Convert TorchScript → PyTorch checkpoint |
| `finetune.sh` | Fine-tune model với pretrained weights |
| `decode.sh` | Decode/transcribe audio |
| `export.sh` | Export model to TorchScript/ONNX |
| `local/prepare_custom_manifest.py` | Tạo Lhotse manifests |
| `local/compute_fbank.py` | Compute fbank features |

## 🔧 Requirements

```bash
# Core dependencies (already in Docker)
pytorch>=2.1.0
k2>=1.24.4
lhotse
kaldifeat
icefall
sentencepiece

# For demo
pip install sherpa-onnx soundfile

# For model download
pip install huggingface_hub
# or
apt-get install git-lfs
```

## 📝 Notes

### Về TorchScript vs PyTorch Checkpoint

Model từ HuggingFace chỉ có `jit_script.pt` (TorchScript) - tốt cho inference nhưng không trực tiếp dùng cho fine-tuning.

Giải pháp:
1. **Option 1**: Dùng `convert_jit_to_pretrained.py` để extract weights
2. **Option 2**: Request `pretrained.pt` từ tác giả model

### Model Architecture Parameters

⚠️ **QUAN TRỌNG**: Khi fine-tune, phải giữ nguyên architecture parameters:

```bash
ENCODER_DIM="384,384,384,384,384"
ENCODER_UNMASKED_DIM="256,256,256,256,256"
FEEDFORWARD_DIM="1024,1024,1024,1024,1024"
NUM_ENCODER_LAYERS="2,2,2,2,2"
NUM_HEADS="4,4,4,4,4"
DECODER_DIM=512
JOINER_DIM=512
```

### Data Format

- Audio: 16kHz, mono WAV
- Transcription: đã normalize (lowercase, bỏ dấu câu không cần)
- Duration: 0.5s - 30s (recommend)

### Fine-tuning Tips

1. **Learning rate**: Dùng 10x nhỏ hơn from-scratch training (~0.0003)
2. **Vocabulary**: Recommend dùng BPE model có sẵn từ pretrained
3. **Data**: Lọc data kém chất lượng với `local/prepare_finetune_data.py`
4. **Checkpointing**: Save mỗi 1000 steps, giữ 5 checkpoint cuối

## 🐳 Docker Usage

```bash
# Attach vào container
docker exec -it hiennt_vietasr_gpu bash

# Working directory
cd /vietasr/ASR

# Run scripts như bình thường
bash scripts/download_zipformer_model.sh
```

## ❓ Troubleshooting

### Demo không chạy

```bash
# Check dependencies
pip list | grep -E "sherpa|soundfile"

# Check model files
ls -lh ../models/zipformer-30m-rnnt/
```

### Fine-tuning OOM (Out of Memory)

Giảm `--max-duration` trong `finetune.sh`:

```bash
MAX_DURATION=150  # Thay vì 300
```

### Decode ra kết quả sai

1. Check `--use-layer-norm 0` (BẮT BUỘC cho model này)
2. Thử tăng `--beam-size` (default: 10)
3. Check audio format (16kHz, mono)

## 📚 References

- Model: https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h
- Icefall: https://github.com/k2-fsa/icefall
- Sherpa-ONNX: https://github.com/k2-fsa/sherpa-onnx
