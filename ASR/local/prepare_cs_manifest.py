#!/usr/bin/env python3
"""
Create Lhotse manifests from synthesized TTS audio.
Generates recordings and supervisions compatible with compute_fbank.py
"""

import json
import argparse
import soundfile as sf
from pathlib import Path
from typing import List, Dict

from lhotse import Recording, SupervisionSegment, RecordingSet, SupervisionSet


def create_manifests(
    audio_dir: Path,
    metadata_file: Path,
    output_dir: Path,
    prefix: str = "vietASR"
) -> None:
    """
    Create Lhotse manifests from TTS audio metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"Loading metadata from {metadata_file}")
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)
    
    samples = metadata["samples"]
    print(f"Found {len(samples)} samples")
    
    recordings = []
    supervisions = []
    
    for sample in samples:
        audio_path = Path(sample["audio_path"])
        
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        # Get actual audio info
        info = sf.info(audio_path)
        duration = info.duration
        sample_rate = info.samplerate
        num_channels = info.channels
        
        rec_id = sample["id"]
        
        # Create Recording
        recording = Recording(
            id=rec_id,
            sources=[{
                "type": "file",
                "channels": list(range(num_channels)),
                "source": str(audio_path.absolute()),
            }],
            sampling_rate=sample_rate,
            num_samples=int(duration * sample_rate),
            duration=duration,
        )
        recordings.append(recording)
        
        # Create Supervision
        supervision = SupervisionSegment(
            id=rec_id,
            recording_id=rec_id,
            start=0.0,
            duration=duration,
            channel=0,
            text=sample["text"],
            speaker=sample.get("speaker", "tts"),
            language="vi",
        )
        supervisions.append(supervision)
    
    print(f"Created {len(recordings)} recordings and {len(supervisions)} supervisions")
    
    # Create Lhotse sets
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    
    # Save manifests
    rec_file = output_dir / f"{prefix}_recordings_cs.jsonl.gz"
    sup_file = output_dir / f"{prefix}_supervisions_cs.jsonl.gz"
    
    recording_set.to_file(rec_file)
    supervision_set.to_file(sup_file)
    
    print(f"Saved recordings to {rec_file}")
    print(f"Saved supervisions to {sup_file}")
    
    # Stats
    total_duration = sum(r.duration for r in recordings)
    print(f"\n📊 Stats:")
    print(f"   Total recordings: {len(recordings)}")
    print(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Create Lhotse manifests from TTS audio")
    parser.add_argument(
        "--audio-dir", type=Path,
        default=Path("data4/cs_augmentation/audio"),
        help="Directory containing WAV files"
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("data4/cs_augmentation/audio/metadata.json"),
        help="Path to metadata.json from TTS synthesis"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data4/cs_augmentation/manifests"),
        help="Output directory for manifests"
    )
    parser.add_argument(
        "--prefix", type=str, default="vietASR",
        help="Prefix for manifest filenames"
    )
    args = parser.parse_args()
    
    create_manifests(
        audio_dir=args.audio_dir,
        metadata_file=args.metadata,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
