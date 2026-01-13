#!/usr/bin/env python3
"""
Quick test script for Zipformer model demo.

Creates a test audio file and runs inference to verify the model works.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def create_test_audio(output_path: str = "test_vi.wav", text: str = "Xin chào Việt Nam"):
    """Create a test audio file using espeak or gtts.
    
    Args:
        output_path: Path to save the audio file
        text: Text to synthesize
    """
    print(f"Creating test audio: '{text}'")
    
    # Try espeak first
    try:
        cmd = [
            "espeak",
            "-v", "vi",
            "-w", output_path,
            text
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Created test audio with espeak: {output_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("espeak not available, trying gtts...")
    
    # Try gtts
    try:
        from gtts import gTTS
        tts = gTTS(text, lang='vi')
        tts.save(output_path)
        
        # Convert to 16kHz WAV if needed
        try:
            cmd = [
                "ffmpeg", "-y", "-i", output_path,
                "-ar", "16000", "-ac", "1",
                f"{output_path}.tmp.wav"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            Path(output_path).unlink()
            Path(f"{output_path}.tmp.wav").rename(output_path)
        except:
            pass
        
        print(f"✓ Created test audio with gtts: {output_path}")
        return True
    except ImportError:
        print("gtts not available (install with: pip install gtts)")
    
    # Try TTS API if available
    print("Failed to create synthetic audio.")
    print("Please provide a real audio file for testing.")
    return False


def test_demo(
    model_dir: str,
    audio_path: str,
    use_int8: bool = False
):
    """Test the demo script.
    
    Args:
        model_dir: Path to model directory
        audio_path: Path to test audio
        use_int8: Use int8 quantized model
    """
    demo_script = Path(__file__).parent / "demo_sherpa_onnx.py"
    
    if not demo_script.exists():
        print(f"Error: Demo script not found: {demo_script}")
        return False
    
    print(f"\nTesting demo with:")
    print(f"  Model: {model_dir}")
    print(f"  Audio: {audio_path}")
    print(f"  Int8: {use_int8}")
    print("\n" + "="*80)
    
    cmd = [
        sys.executable,
        str(demo_script),
        "--model-dir", model_dir,
        "--audio", audio_path
    ]
    
    if use_int8:
        cmd.append("--use-int8")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("="*80)
        print("\n✓ Demo test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("="*80)
        print(f"\n✗ Demo test failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick test for Zipformer demo"
    )
    
    parser.add_argument(
        "--model-dir",
        default="/home/hiennt/VietASR/models/zipformer-30m-rnnt",
        help="Path to model directory"
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to test audio (if not provided, will create synthetic audio)"
    )
    parser.add_argument(
        "--use-int8",
        action="store_true",
        help="Use int8 quantized model"
    )
    parser.add_argument(
        "--create-audio-only",
        action="store_true",
        help="Only create test audio, don't run demo"
    )
    
    args = parser.parse_args()
    
    # Handle audio file
    audio_path = args.audio
    
    if audio_path is None:
        audio_path = "test_vi.wav"
        if not create_test_audio(audio_path):
            return 1
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    if args.create_audio_only:
        print(f"\nTest audio created: {audio_path}")
        print("You can now run:")
        print(f"  python ASR/demo_sherpa_onnx.py --audio {audio_path}")
        return 0
    
    # Test demo
    success = test_demo(
        model_dir=args.model_dir,
        audio_path=audio_path,
        use_int8=args.use_int8
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
