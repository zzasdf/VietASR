#!/usr/bin/env python3
"""
TTS Audio Synthesizer using VCC Internal API.
Generates WAV 16kHz mono audio from CS sentences.
"""

import os
import sys
import json
import base64
import requests
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydub import AudioSegment

# TTS API Configuration
TTS_API_URL = "https://speech.aiservice.vn/tts/api/v1/synthesis"
PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"

# VCCorp Speakers (tested and working)
VCCORP_SPEAKERS = [
    "hn_quockhanh",    # Nam - Hà Nội
    "hn_hoanganhquan", # Nam - Hà Nội
    "hn_vanhoang",     # Nam - Hà Nội
    "hn_minhphuong",   # Nữ - Hà Nội
    "hcm_minhhoa",     # Nam - HCM
    "hcm_thanhthao",   # Nữ - HCM
]


class VCCTTSBackend:
    """VCC Internal TTS API Backend."""
    
    def __init__(self, speaker_id: str = None):
        self.speaker_id = speaker_id
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": PRIVATE_TOKEN
        }
    
    def synthesize(self, text: str, output_path: Path) -> Dict:
        """
        Synthesize text to WAV 16kHz mono.
        Returns metadata dict with duration, sample_rate, etc.
        """
        payload = {"text": text}
        if self.speaker_id:
            payload["speaker_id"] = self.speaker_id
        
        try:
            # Call TTS API
            response = requests.post(
                TTS_API_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                return {"success": False, "error": f"HTTP {response.status_code}"}
            
            result = response.json()
            data = result.get("data", {})
            audio_url = data.get("file_path")
            info = data.get("info", {})
            
            if not audio_url:
                return {"success": False, "error": "No audio URL in response"}
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=60)
            if audio_response.status_code != 200:
                return {"success": False, "error": "Failed to download audio"}
            
            # Save temp m4a
            temp_m4a = str(output_path).replace(".wav", ".m4a")
            with open(temp_m4a, "wb") as f:
                f.write(audio_response.content)
            
            # Convert to WAV 16kHz mono
            audio = AudioSegment.from_file(temp_m4a, format="m4a")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            
            # Cleanup temp
            os.remove(temp_m4a)
            
            # Get duration
            duration_sec = len(audio) / 1000.0
            
            return {
                "success": True,
                "duration": duration_sec,
                "sample_rate": 16000,
                "speaker": info.get("speaker_id", self.speaker_id),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def synthesize_sentences(
    sentences: List[Dict],
    output_dir: Path,
    speakers: List[str],
    max_workers: int = 2,
    max_samples: int = None,
) -> List[Dict]:
    """
    Batch synthesize sentences with multiple speakers.
    Uses random speaker selection for diversity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if max_samples:
        sentences = sentences[:max_samples]
    
    results = []
    failed = 0
    
    def process_sentence(item):
        idx, sent = item
        # Random speaker for diversity
        speaker = random.choice(speakers)
        backend = VCCTTSBackend(speaker_id=speaker)
        
        output_path = output_dir / f"{sent['id']}.wav"
        result = backend.synthesize(sent["text"], output_path)
        
        if result["success"]:
            return {
                "id": sent["id"],
                "text": sent["text"],
                "audio_path": str(output_path),
                "duration": result["duration"],
                "speaker": result["speaker"],
                "category": sent.get("category", ""),
            }
        else:
            return None
    
    # Process with progress bar
    print(f"Synthesizing {len(sentences)} sentences with {len(speakers)} speakers...")
    print(f"Using {max_workers} workers (be patient, TTS API can be slow)")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_sentence, (i, s)): i 
            for i, s in enumerate(sentences)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="TTS"):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1
    
    print(f"\n✅ Success: {len(results)}/{len(sentences)}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="TTS synthesis for CS sentences")
    parser.add_argument(
        "--sentences", type=Path, 
        default=Path("data4/cs_augmentation/sentences/cs_sentences.jsonl"),
        help="Path to sentences JSONL file"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data4/cs_augmentation/audio"),
        help="Output directory for WAV files"
    )
    parser.add_argument(
        "--speakers", type=str, 
        default="hn_minhphuong,hn_quockhanh,hcm_minhhoa",
        help="Comma-separated list of speaker IDs"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to synthesize (for testing)"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel workers"
    )
    args = parser.parse_args()
    
    # Load sentences
    print(f"Loading sentences from {args.sentences}")
    sentences = []
    with open(args.sentences, encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line.strip()))
    print(f"Loaded {len(sentences)} sentences")
    
    # Parse speakers
    speakers = [s.strip() for s in args.speakers.split(",")]
    print(f"Using speakers: {speakers}")
    
    # Synthesize
    results = synthesize_sentences(
        sentences,
        args.output_dir,
        speakers,
        max_workers=args.workers,
        max_samples=args.max_samples,
    )
    
    # Save metadata
    metadata_file = args.output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "speakers": speakers,
            "samples": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metadata to {metadata_file}")
    
    # Stats
    if results:
        total_duration = sum(r["duration"] for r in results)
        print(f"\n📊 Stats:")
        print(f"   Total audio: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"   Avg duration: {total_duration/len(results):.2f}s per sentence")


if __name__ == "__main__":
    main()
