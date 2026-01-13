#!/usr/bin/env python3
"""
Convert MUSAN recordings to cuts format.
This creates cuts directly from recordings without needing lhotse installed.
"""
import gzip
import json
from pathlib import Path

def create_cut_from_recording(recording):
    """Create a MonoCut from a Recording."""
    return {
        "id": f"{recording['id']}-0",
        "start": 0,
        "duration": recording["duration"],
        "channel": 0,
        "supervisions": [{
            "id": recording["id"],
            "recording_id": recording["id"],
            "start": 0,
            "duration": recording["duration"],
            "channel": 0,
        }],
        "recording": recording,
        "type": "MonoCut"
    }

def convert_recordings_to_cuts(recordings_file, cuts_file):
    """Convert recordings JSONL to cuts JSONL."""
    print(f"Converting {recordings_file.name} -> {cuts_file.name}")
    count = 0
    
    with gzip.open(recordings_file, 'rt', encoding='utf-8') as f_in, \
         gzip.open(cuts_file, 'wt', encoding='utf-8') as f_out:
        
        for line in f_in:
            recording = json.loads(line)
            cut = create_cut_from_recording(recording)
            json.dump(cut, f_out, ensure_ascii=False)
            f_out.write('\n')
            count += 1
    
    print(f"  Created {count} cuts")
    return count

def main():
    input_dir = Path("data/wav")
    
    # Convert each recordings file to cuts
    parts = ["noise", "speech"]
    
    for part in parts:
        recordings_file = input_dir / f"musan_recordings_{part}.jsonl.gz"
        cuts_file = input_dir / f"musan_cuts_{part}.jsonl.gz"
        
        if not recordings_file.exists():
            print(f"Warning: {recordings_file} not found")
            continue
        
        if cuts_file.exists():
            print(f"  {cuts_file.name} already exists, skipping")
            continue
            
        convert_recordings_to_cuts(recordings_file, cuts_file)
    
    print("\nNow combining all cuts...")
    
    # Combine all cuts
    output_file = input_dir / "musan_cuts.jsonl.gz"
    cut_files = [
        input_dir / "musan_cuts_music.jsonl.gz",
        input_dir / "musan_cuts_noise.jsonl.gz",
        input_dir / "musan_cuts_speech.jsonl.gz",
    ]
    
    total = 0
    with gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
        for cut_file in cut_files:
            if not cut_file.exists():
                print(f"  Warning: {cut_file.name} not found, skipping")
                continue
            
            print(f"  Adding {cut_file.name}...")
            count = 0
            with gzip.open(cut_file, 'rt', encoding='utf-8') as f_in:
                for line in f_in:
                    f_out.write(line)
                    count += 1
            print(f"    Added {count} cuts")
            total += count
    
    print(f"\nTotal cuts in final file: {total}")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
