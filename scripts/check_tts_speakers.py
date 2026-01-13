#!/usr/bin/env python3
"""Test TTS với các giọng VCCorp."""

import os
import requests
from pathlib import Path
from pydub import AudioSegment

PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"
TTS_API_URL = "https://speech.aiservice.vn/tts/api/v1/synthesis"

# VCCorp Speakers
VCCORP_SPEAKERS = [
    # Nam - Hà Nội
    ("hn_quockhanh", "Nam - HN"),
    ("hn_hoanganhquan", "Nam - HN"),
    ("hn_vanhoang", "Nam - HN"),
    # Nữ - Hà Nội
    ("hn_minhphuong", "Nữ - HN"),
    # Nam - HCM
    ("hcm_minhhoa", "Nam - HCM"),
    # Nữ - HCM
    ("hcm_thanhthao", "Nữ - HCM"),
]

TEST_TEXT = "Bạn cần upload file lên cloud"

def test_speaker(speaker_id: str, output_path: str) -> dict:
    """Test a speaker and save WAV 16kHz."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": PRIVATE_TOKEN
    }
    
    payload = {
        "text": TEST_TEXT,
        "speaker_id": speaker_id
    }
    
    try:
        r = requests.post(TTS_API_URL, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            return {"success": False, "error": f"HTTP {r.status_code}"}
        
        data = r.json().get("data", {})
        audio_url = data.get("file_path")
        info = data.get("info", {})
        
        # Download and convert
        audio_r = requests.get(audio_url, timeout=30)
        temp_m4a = "/tmp/temp.m4a"
        with open(temp_m4a, "wb") as f:
            f.write(audio_r.content)
        
        audio = AudioSegment.from_file(temp_m4a, format="m4a")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        os.remove(temp_m4a)
        
        return {
            "success": True,
            "speaker": info.get("speaker_id"),
            "duration": info.get("duration"),
            "file": output_path,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("  VCCorp TTS Speakers Test")
    print("=" * 60)
    print(f"  Text: '{TEST_TEXT}'")
    
    output_dir = Path("/tmp/vccorp_tts")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    for speaker_id, desc in VCCORP_SPEAKERS:
        print(f"\n📢 {speaker_id} ({desc})...")
        output_path = str(output_dir / f"{speaker_id}.wav")
        result = test_speaker(speaker_id, output_path)
        
        if result["success"]:
            print(f"   ✅ OK - {result['duration']:.2f}s")
            results.append((speaker_id, desc, result["file"]))
        else:
            print(f"   ❌ {result.get('error', 'Failed')}")
    
    print("\n" + "=" * 60)
    print(f"  ✅ {len(results)}/{len(VCCORP_SPEAKERS)} speakers working")
    print("=" * 60)
    
    if results:
        print(f"\n📁 WAV files (16kHz mono): {output_dir}/")
        for speaker, desc, path in results:
            size = os.path.getsize(path)
            print(f"   {speaker}.wav - {desc} ({size:,} bytes)")


if __name__ == "__main__":
    main()
