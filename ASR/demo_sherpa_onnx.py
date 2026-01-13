#!/usr/bin/env python3
"""
Demo inference using sherpa-onnx with Zipformer-30M-RNNT model.

This script demonstrates how to use the pretrained Zipformer model
for Vietnamese ASR inference.

Requirements:
    pip install sherpa-onnx soundfile

Usage:
    # Single file
    python demo_sherpa_onnx.py --audio test.wav
    
    # Multiple files or directory
    python demo_sherpa_onnx.py --audio audio_dir/ --output results.txt
    
    # Use int8 quantized model for faster inference
    python demo_sherpa_onnx.py --audio test.wav --use-int8
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

try:
    import sherpa_onnx
except ImportError:
    print("Error: sherpa-onnx not installed.")
    print("Install with: pip install sherpa-onnx")
    exit(1)

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile not installed.")
    print("Install with: pip install soundfile")
    exit(1)


def create_recognizer(
    model_dir: str,
    use_int8: bool = False,
    num_threads: int = 4
) -> sherpa_onnx.OfflineRecognizer:
    """Create offline recognizer for Zipformer model.
    
    Args:
        model_dir: Path to model directory
        use_int8: Use int8 quantized encoder for faster inference
        num_threads: Number of threads for inference
        
    Returns:
        Configured OfflineRecognizer instance
    """
    model_dir = Path(model_dir)
    
    # Select encoder based on quantization
    if use_int8:
        encoder_path = model_dir / "encoder-epoch-20-avg-10.int8.onnx"
        if not encoder_path.exists():
            print(f"Warning: int8 encoder not found at {encoder_path}")
            print("Falling back to fp32 encoder...")
            encoder_path = model_dir / "encoder-epoch-20-avg-10.onnx"
    else:
        encoder_path = model_dir / "encoder-epoch-20-avg-10.onnx"
    
    decoder_path = model_dir / "decoder-epoch-20-avg-10.onnx"
    joiner_path = model_dir / "joiner-epoch-20-avg-10.onnx"
    tokens_path = model_dir / "config.json"
    
    # Verify all files exist
    for path in [encoder_path, decoder_path, joiner_path, tokens_path]:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
    
    print(f"Loading model from {model_dir}")
    print(f"  Encoder: {encoder_path.name}")
    print(f"  Decoder: {decoder_path.name}")
    print(f"  Joiner: {joiner_path.name}")
    print(f"  Tokens: {tokens_path.name}")
    print(f"  Threads: {num_threads}")
    
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(encoder_path),
        decoder=str(decoder_path),
        joiner=str(joiner_path),
        tokens=str(tokens_path),
        num_threads=num_threads,
    )
    
    return recognizer


def transcribe_audio(
    recognizer: sherpa_onnx.OfflineRecognizer,
    audio_path: str
) -> Tuple[str, float]:
    """Transcribe a single audio file.
    
    Args:
        recognizer: OfflineRecognizer instance
        audio_path: Path to audio file
        
    Returns:
        Tuple of (transcription, processing_time_seconds)
    """
    # Load audio
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return "", 0.0
    
    # Convert stereo to mono if needed
    if len(samples.shape) > 1:
        samples = samples[:, 0]
    
    # Warn if sample rate is not 16kHz
    if sample_rate != 16000:
        print(f"Warning: Audio should be 16kHz, got {sample_rate}Hz")
        print("  Transcription may be inaccurate. Consider resampling.")
    
    # Measure processing time
    start_time = time.time()
    
    # Create stream and decode
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    
    elapsed = time.time() - start_time
    transcription = stream.result.text
    
    return transcription, elapsed


def find_audio_files(path: str) -> List[str]:
    """Find all audio files in a directory or return single file.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of audio file paths
    """
    path = Path(path)
    
    if path.is_file():
        return [str(path)]
    
    if path.is_dir():
        audio_extensions = {'.wav', '.flac', '.mp3', '.ogg', '.opus'}
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(path.glob(f'**/*{ext}'))
        return [str(f) for f in sorted(audio_files)]
    
    raise ValueError(f"Path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo ASR inference with Zipformer-30M-RNNT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python demo_sherpa_onnx.py --audio test.wav
  
  # Directory of files
  python demo_sherpa_onnx.py --audio /path/to/audio/ --output results.txt
  
  # Use int8 quantized model
  python demo_sherpa_onnx.py --audio test.wav --use-int8
        """
    )
    
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file or directory"
    )
    parser.add_argument(
        "--model-dir",
        default="/home/hiennt/VietASR/models/zipformer-30m-rnnt",
        help="Path to model directory (default: %(default)s)"
    )
    parser.add_argument(
        "--use-int8",
        action="store_true",
        help="Use int8 quantized encoder for faster inference"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for inference (default: %(default)s)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file to save transcriptions (optional)"
    )
    
    args = parser.parse_args()
    
    # Create recognizer
    recognizer = create_recognizer(
        args.model_dir,
        use_int8=args.use_int8,
        num_threads=args.num_threads
    )
    
    # Find audio files
    audio_files = find_audio_files(args.audio)
    print(f"\nFound {len(audio_files)} audio file(s)")
    print("=" * 80)
    
    # Process each file
    results = []
    total_audio_duration = 0.0
    total_processing_time = 0.0
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_path}")
        
        # Get audio duration
        try:
            info = sf.info(audio_path)
            duration = info.duration
            total_audio_duration += duration
        except:
            duration = 0.0
        
        # Transcribe
        transcription, elapsed = transcribe_audio(recognizer, audio_path)
        total_processing_time += elapsed
        
        # Calculate RTF (Real-Time Factor)
        rtf = elapsed / duration if duration > 0 else 0.0
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Processing time: {elapsed:.2f}s (RTF: {rtf:.3f})")
        print(f"  Transcription: {transcription}")
        
        results.append({
            'file': audio_path,
            'duration': duration,
            'processing_time': elapsed,
            'rtf': rtf,
            'transcription': transcription
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(audio_files)}")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Total processing time: {total_processing_time:.2f}s")
    if total_audio_duration > 0:
        avg_rtf = total_processing_time / total_audio_duration
        print(f"Average RTF: {avg_rtf:.3f}")
        print(f"Speed: {1/avg_rtf:.1f}x realtime" if avg_rtf > 0 else "")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result['file']}\t{result['transcription']}\n")
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
