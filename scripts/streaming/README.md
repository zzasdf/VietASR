# Streaming ASR Scripts

This directory contains scripts for training, benchmarking, and running inference with Streaming ASR models (Zipformer).

## 1. Training
Use `train.sh` to start fine-tuning.
```bash
./scripts/streaming/train.sh
```
Key parameters in `train.sh`:
- `--causal True`: Enable streaming mode.
- `--chunk-size "16,32,64,-1"`: Train with dynamic chunk sizes.
- `--final-downsample 1`: **Important** Use downsampling at output (factor 2).

## 2. Benchmark RTF (Real-Time Factor)
Measure the speed and latency of the model.

```bash
python scripts/streaming/benchmark_rtf.py \
  --checkpoint data4/exp_streaming/epoch-30.pt \
  --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
  --wav /path/to/test.wav \
  --chunk-frames "16,32,64" \
  --final-downsample 1  # MUST MATCH TRAINING CONFIG
```

## 3. Streaming Inference (Decode Wav to Text)
Run simulated streaming decoding on a wav file.

**Greedy Search:**
```bash
python scripts/streaming/inference.py \
  --checkpoint data4/exp_streaming/epoch-30.pt \
  --bpe-model viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/bpe.model \
  --wav /path/to/test.wav \
  --chunk-frames 32 \
  --final-downsample 1 \
  --method greedy
```

**Beam Search:**
```bash
python scripts/streaming/inference.py \
  ... \
  --method beam_search \
  --beam-size 5
```

### Common Errors
- `size mismatch for encoder.encoder.downsample_output.bias`: 
  You forgot to add `--final-downsample 1`. The default is 0 (False), but training usually uses 1 (True).
