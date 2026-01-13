#!/usr/bin/env python3
# Copyright 2024 (Author: Antigravity)

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import lhotse
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, CutSet
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare custom data for VietASR. Converts raw transcripts and wavs to Lhotse manifests."
    )
    # Mode 1: Single dataset
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Path to the transcript file (for single dataset mode).",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        help="Directory containing audio files (for single dataset mode).",
    )
    
    # Mode 2: Multiple datasets (Corpus dir)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="Root directory containing multiple datasets (subdirectories). Each subdir should have 'transcripts.txt' and 'wavs'.",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        help="Only include these dataset names (folder names). Example: --include tongdai regions.bac_trung_bo",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        help="Exclude these dataset names. Example: --exclude vlsp2020_train_set_01 cv-corpus-7.0-20210721",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Just list available datasets in --corpus-dir and exit (no processing).",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the output manifests.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="custom",
        help="Name prefix for the output files (e.g., 'train', 'hue_dialect').",
    )
    parser.add_argument(
        "--compute-fbank",
        action="store_true",
        help="If True, compute Fbank features immediately (requires GPU/CPU).",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip scanning wav-dir (use when transcript contains full paths).",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs for Fbank computation. Default: 8 (safe for shared servers).",
    )
    return parser.parse_args()

def scan_audio_files(wav_dir: Path, skip_scan: bool = False) -> Dict[str, Path]:
    """Recursively find audio files and map stem (id) to full path.
    
    Args:
        wav_dir: Directory containing audio files
        skip_scan: If True, skip scanning (use when transcript has full paths)
    """
    audio_map = {}
    if skip_scan:
        logging.info("Skipping audio directory scan (using full paths from transcript)")
        return audio_map
        
    if not wav_dir.exists():
        logging.warning(f"Wav directory not found: {wav_dir}")
        return audio_map
        
    logging.info(f"Scanning audio files in {wav_dir}...")
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    
    count = 0
    for root, _, files in os.walk(wav_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() in extensions:
                audio_map[path.stem] = path.absolute()
                count += 1
                if count % 100000 == 0:
                    logging.info(f"Scanned {count} audio files...")
    
    logging.info(f"Found {len(audio_map)} audio files.")
    return audio_map

def process_single_dataset(transcript_path: Path, wav_dir: Path, skip_scan: bool = False) -> (List[Recording], List[SupervisionSegment]):
    audio_map = scan_audio_files(wav_dir, skip_scan=skip_scan)
    recordings = []
    supervisions = []
    
    if not transcript_path.exists():
        logging.warning(f"Transcript file not found: {transcript_path}")
        return [], []

    logging.info(f"Processing transcript: {transcript_path}")
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            
            # Try pipe separator first (path|text|duration), then space separator
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    audio_id_or_path = parts[0].strip()
                    text = parts[1].strip()
                    try:
                        duration = float(parts[2].strip())
                    except ValueError:
                        continue
                else:
                    continue
            else:
                # Fallback to space separator (id text duration)
                parts = line.split()
                try:
                    duration = float(parts[-1])
                    text = " ".join(parts[1:-1])
                    audio_id_or_path = parts[0]
                except ValueError:
                    continue

            cut_id = Path(audio_id_or_path).stem
            
            if cut_id not in audio_map:
                if Path(audio_id_or_path).exists():
                     audio_path = Path(audio_id_or_path).absolute()
                     cut_id = audio_path.stem
                else:
                    continue
            else:
                audio_path = audio_map[cut_id]

            try:
                # Create Recording using provided duration to be fast
                recording = Recording(
                    id=cut_id,
                    sources=[{"type": "file", "channels": [0], "source": str(audio_path)}],
                    sampling_rate=16000, 
                    num_samples=int(duration * 16000),
                    duration=duration
                )
            except Exception as e:
                logging.warning(f"Error creating recording {audio_path}: {e}")
                continue

            recordings.append(recording)
            
            segment = SupervisionSegment(
                id=cut_id,
                recording_id=recording.id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="vietnamese",
                text=text,
            )
            supervisions.append(segment)
            
    return recordings, supervisions

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    all_recordings = []
    all_supervisions = []

    if args.corpus_dir:
        logging.info(f"Scanning corpus directory: {args.corpus_dir}")
        
        # List mode: just show available datasets
        if args.list_datasets:
            print("\nAvailable datasets:")
            for child in sorted(args.corpus_dir.iterdir()):
                if child.is_dir():
                    has_transcript = any((child / t).exists() for t in ["transcripts.txt", "transcripts", "transcript.txt"])
                    has_wav = any((child / w).exists() for w in ["wavs", "wav", "audio"])
                    status = "✓" if (has_transcript and has_wav) else "✗"
                    print(f"  {status} {child.name}")
            print("\nUse --include <name1> <name2> ... to select specific datasets.")
            return
        
        # Iterate over subdirectories
        for child in sorted(args.corpus_dir.iterdir()):
            if child.is_dir():
                # Filter by include/exclude
                if args.include and child.name not in args.include:
                    continue
                if args.exclude and child.name in args.exclude:
                    logging.info(f"Excluding: {child.name}")
                    continue
                
                # Look for 'transcripts.txt' or 'transcripts' (file) and 'wavs' (dir)
                transcript_candidates = [child / "transcripts.txt", child / "transcripts", child / "transcript.txt"]
                wav_dir_candidates = [child / "wavs", child / "wav", child / "audio"]
                
                found_transcript = None
                for t in transcript_candidates:
                    if t.exists():
                        found_transcript = t
                        break
                
                found_wav = None
                for w in wav_dir_candidates:
                    if w.exists():
                        found_wav = w
                        break
                
                if found_transcript and found_wav:
                    logging.info(f"Found dataset: {child.name}")
                    recs, sups = process_single_dataset(found_transcript, found_wav, skip_scan=getattr(args, 'skip_scan', False))
                    all_recordings.extend(recs)
                    all_supervisions.extend(sups)
                else:
                    logging.info(f"Skipping {child.name}: missing 'transcripts' or 'wavs'")
    
    elif args.transcript and args.wav_dir:
        recs, sups = process_single_dataset(args.transcript, args.wav_dir, skip_scan=getattr(args, 'skip_scan', False))
        all_recordings.extend(recs)
        all_supervisions.extend(sups)
    else:
        logging.error("Please provide either --corpus-dir OR (--transcript and --wav-dir)")
        return

    if not all_recordings:
        logging.error("No data found!")
        return

    logging.info(f"Total processed: {len(all_recordings)} segments.")

    # Create Manifests
    recording_set = RecordingSet.from_recordings(all_recordings)
    supervision_set = SupervisionSet.from_segments(all_supervisions)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    rec_path = args.output_dir / f"{args.dataset_name}_recordings.jsonl.gz"
    sup_path = args.output_dir / f"{args.dataset_name}_supervisions.jsonl.gz"
    
    logging.info(f"Saving recordings to {rec_path}")
    recording_set.to_file(rec_path)
    
    logging.info(f"Saving supervisions to {sup_path}")
    supervision_set.to_file(sup_path)

    # Create CutSet
    logging.info("Creating CutSet...")
    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )
    
    if args.compute_fbank:
        logging.info("Computing Fbank features...")
        cut_set = cut_set.compute_and_store_features(
            extractor=lhotse.Fbank(lhotse.FbankConfig(num_mel_bins=80)),
            storage_path=args.output_dir / f"{args.dataset_name}_feats",
            num_jobs=args.num_jobs,
        )
    
    cuts_path = args.output_dir / f"{args.dataset_name}_cuts.jsonl.gz"
    logging.info(f"Saving CutSet to {cuts_path}")
    cut_set.to_file(cuts_path)
    
    logging.info("Done!")
