#!/usr/bin/env python3
"""
Convert Espnet format to Icefall format with Train/Dev split capability.
Espnet format: folder/<dataset_name>/transcript.txt + wavs/
Icefall format: folder/<dataset_name>/{train,dev,test}/{*.wav, *.trans.txt}
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to Espnet data folder (contains transcript.txt and wavs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Icefall format",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="vietASR",
        help="Dataset name prefix (default: vietASR)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],  # REMOVED "dev"
        default="train",
        help="Which split is this input data? (train/test). If train, it will be split into train/dev.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0,
        help="Ratio of data to use for validation/dev set (only applies when split='train'). Default: 0.1",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Max duration in seconds to filter out long utterances",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Min duration in seconds to filter out short utterances",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling train/dev split",
    )
    return parser.parse_args()


def load_transcript(transcript_path: Path) -> pd.DataFrame:
    """Load transcript.txt with format: path_wav|text|duration"""
    logging.info(f"Loading transcript from {transcript_path}")
    
    data = []
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript file: {transcript_path}")

    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # TÁCH BẰNG DẤU | 
            parts = line.split('|')
            if len(parts) != 3:
                logging.warning(f"Line {line_num}: Expected 3 parts separated by '|', got {len(parts)}: {line}")
                continue
            
            path_wav = parts[0].strip()
            text = parts[1].strip()
            duration_str = parts[2].strip()
            
            # Convert duration sang float
            try:
                duration = float(duration_str)
            except ValueError:
                logging.warning(f"Line {line_num}: Invalid duration '{duration_str}': {line}")
                continue
            
            data.append({
                'path_wav': path_wav,
                'text': text,
                'duration': duration
            })
    
    df = pd.DataFrame(data)
    logging.info(f"Loaded {len(df)} utterances")
    return df


def process_subset(
    df: pd.DataFrame,
    input_wavs_dir: Path,
    output_dir: Path,
    dataset_name: str,
    split_name: str
):
    """
    Process a specific subset (dataframe) and write it to the output directory.
    """
    if df.empty:
        logging.warning(f"Subset {split_name} is empty. Skipping.")
        return

    # Create specific output folder (e.g., output/train/vietASR)
    target_dir = output_dir / split_name / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    transcriptions = []
    skipped_files = []
    
    logging.info(f"Processing split '{split_name}' with {len(df)} utterances...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {split_name}"):
        wav_name = Path(row["path_wav"]).name
        
        # Find actual wav file
        src_wav_path = input_wavs_dir / wav_name
        
        if not src_wav_path.exists():
            # Try alternative extensions
            for ext in [".wav", ".WAV", ".flac"]:
                alt_path = src_wav_path.with_suffix(ext)
                if alt_path.exists():
                    src_wav_path = alt_path
                    break
            else:
                skipped_files.append(str(src_wav_path))
                continue
        
        # Create new ID: remove extension, replace special chars
        utt_id = wav_name.replace(".wav", "").replace(".", "_").replace("-", "_")
        
        # Copy audio file
        dst_wav_path = target_dir / f"{utt_id}.wav"
        if not dst_wav_path.exists():
            shutil.copy2(src_wav_path, dst_wav_path)
        
        # Store transcription
        transcriptions.append(f"{utt_id} {row['text']}")
    
    if skipped_files:
        logging.warning(f"Skipped {len(skipped_files)} missing audio files in {split_name}")
    
    # Write transcript file
    trans_file = target_dir / f"{split_name}.trans.txt"
    with open(trans_file, "w", encoding="utf-8") as f:
        f.write("\n".join(transcriptions) + "\n")
    
    logging.info(f"Finished {split_name}: Saved to {trans_file}")


def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    logging.info("Starting conversion from Espnet to Icefall format")
    logging.info(f"Input: {args.input_dir}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Mode: {args.split}")

    # 1. Check input folders
    transcript_path = args.input_dir / "transcripts.txt"
    wavs_dir = args.input_dir / "wavs"
    
    if not wavs_dir.exists():
        raise FileNotFoundError(f"Missing wavs folder: {wavs_dir}")

    # 2. Load and Filter Data
    df = load_transcript(transcript_path)
    
    original_len = len(df)
    df = df[(df["duration"] >= args.min_duration) & (df["duration"] <= args.max_duration)]
    filtered_len = len(df)
    
    logging.info(
        f"Filtered {original_len - filtered_len}/{original_len} utterances "
        f"(duration range: {args.min_duration}s - {args.max_duration}s)"
    )

    # 3. Process Splits
    if args.split == "train":
        # Shuffle data
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        
        # Calculate split index
        dev_size = int(len(df) * args.dev_ratio)
        
        if dev_size == 0 and len(df) > 1:
            logging.warning("Dev ratio is too small given the dataset size. Forcing 1 sample to dev.")
            dev_size = 1
            
        dev_df = df.iloc[:dev_size]
        train_df = df.iloc[dev_size:]
        
        logging.info(f"Splitting input 'train' into: Train ({len(train_df)}) and Dev ({len(dev_df)}) with ratio {args.dev_ratio}")
        
        # Process Train
        process_subset(train_df, wavs_dir, args.output_dir, args.dataset_name, "train")
        # Process Dev
        process_subset(dev_df, wavs_dir, args.output_dir, args.dataset_name, "dev")
        
    else:
        # Test set - no splitting
        logging.info(f"Processing 'test' set ({len(df)} items)")
        process_subset(df, wavs_dir, args.output_dir, args.dataset_name, "test")
    
    logging.info("Conversion completed!")


if __name__ == "__main__":
    main()