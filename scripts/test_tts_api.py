#!/usr/bin/env python3
"""Test TTS API với output WAV 16kHz cho training."""

import os
import requests
from pathlib import Path
from pydub import AudioSegment

PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"
TTS_API_URL = "https://speech.aiservice.vn/tts/api/v1/synthesis"


def synthesize_to_wav16k(text: str, output_path: str) -> dict:
    """Synthesize text to WAV 16kHz mono."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": PRIVATE_TOKEN
    }
    
    payload = {"text": text}
    
    print(f"📤 Synthesizing: '{text}'")
    
    # Call TTS API
    response = requests.post(TTS_API_URL, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        return {"success": False}
    
    result = response.json()
    data = result.get("data", {})
    audio_url = data.get("file_path")
    info = data.get("info", {})
    
    print(f"   TTS Duration: {info.get('duration'):.2f}s @ {info.get('sample_rate')}Hz")
    
    # Download m4a audio
    print(f"📥 Downloading...")
    audio_response = requests.get(audio_url, timeout=30)
    
    if audio_response.status_code != 200:
        print(f"❌ Download failed")
        return {"success": False}
    
    # Save temp m4a
    temp_m4a = "/tmp/temp_tts.m4a"
    with open(temp_m4a, "wb") as f:
        f.write(audio_response.content)
    
    # Convert to WAV 16kHz mono using pydub
    print(f"🔄 Converting to WAV 16kHz mono...")
    audio = AudioSegment.from_file(temp_m4a, format="m4a")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    
    # Cleanup temp
    os.remove(temp_m4a)
    
    # Get final info
    file_size = os.path.getsize(output_path)
    duration_ms = len(audio)
    
    print(f"✅ Saved: {output_path}")
    print(f"   Format: WAV 16kHz mono")
    print(f"   Duration: {duration_ms/1000:.2f}s")
    print(f"   Size: {file_size:,} bytes")
    
    return {
        "success": True,
        "output_path": output_path,
        "duration_sec": duration_ms / 1000,
        "sample_rate": 16000,
    }


def main():
    print("=" * 60)
    print("  TTS → WAV 16kHz Test (Training Compatible)")
    print("=" * 60)
    
    # Test với câu CS
    test_sentences = [
        "Xin chào",
        "Bạn cần upload file lên cloud",
        "Hãy check lại database server",
    ]
    
    output_dir = Path("/tmp/tts_wav16k")
    output_dir.mkdir(exist_ok=True)
    
    for i, text in enumerate(test_sentences):
        print(f"\n--- Test {i+1}/{len(test_sentences)} ---")
        output_path = str(output_dir / f"cs_{i:03d}.wav")
        result = synthesize_to_wav16k(text, output_path)
        
        if not result["success"]:
            print(f"❌ Failed for: {text}")
            break
    else:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print(f"📁 WAV files: {output_dir}/*.wav")
        
        # List files
        print("\n📋 Output files:")
        for f in sorted(output_dir.glob("*.wav")):
            size = f.stat().st_size
            print(f"   {f.name}: {size:,} bytes")


if __name__ == "__main__":
    main()
