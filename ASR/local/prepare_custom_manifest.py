import argparse
import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm


def prepare_custom_manifest(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Assume the structure is flat or contains subdirectories, 
    # but we look for transcript.txt files.
    # We will group everything into a single "train" set for now, 
    # or infer splits if the user organizes them into 'train', 'test' folders.
    
    # Let's try to detect splits based on top-level directories
    # If not found, assume everything is 'train'
    parts = ["train", "test", "dev"]
    detected_parts = []
    for p in parts:
        if (corpus_dir / p).is_dir():
            detected_parts.append(p)
    
    if not detected_parts:
        # Treat the root as one partition, default to 'train'
        # But wait, the user said /storage/asr/data/raw/train
        # So likely they point corpus_dir to /storage/asr/data/raw
        # and it has 'train'.
        # Or they point directly to 'train'.
        # Let's be flexible. If we find transcript.txt in root, it's 'train'.
        if (corpus_dir / "transcript.txt").exists():
             # This is a single split dataset
             # We will name it based on the directory name or just 'train'
             detected_parts = ["train"]
        else:
             # Maybe it's recursive
             detected_parts = ["train"] # Fallback

    
    # Actually, let's just walk the directory and find all transcript.txt
    # and aggregate them. 
    # But lhotse expects splits.
    # Let's assume the user provides the root which contains 'train', 'test' etc.
    # If the user provided path IS 'train', then we just process it as 'train'.
    
    dataset_parts = {} # name -> path
    
    if (corpus_dir / "transcript.txt").exists():
        dataset_parts["train"] = corpus_dir
    else:
        # Check subdirs
        for child in corpus_dir.iterdir():
            if child.is_dir():
                # Check if this child has transcript.txt or subdirs with it
                # For simplicity, let's assume standard structure: corpus/train, corpus/test
                if child.name in ["train", "test", "dev", "valid"]:
                    dataset_parts[child.name] = child
                else:
                    # Maybe the user has arbitrary names, treat as chunks of train?
                    # Let's stick to standard names or just process what we find.
                    pass
    
    # If still empty, maybe the user pointed to a dir that HAS subdirs with transcripts
    if not dataset_parts:
         # Just scan recursively?
         # Let's assume the user input IS the 'train' dir if no structure found.
         dataset_parts["train"] = corpus_dir

    with ThreadPoolExecutor(num_jobs) as ex:
        for part_name, part_path in dataset_parts.items():
            logging.info(f"Processing subset: {part_name} in {part_path}")
            
            recordings = []
            supervisions = []
            futures = []
            
            # Find all transcript.txt files
            trans_files = list(part_path.rglob("transcript.txt"))
            
            for trans_path in tqdm(trans_files, desc=f"Distributing tasks for {part_name}"):
                with open(trans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        futures.append(
                            ex.submit(parse_line, trans_path, line)
                        )

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            if not recordings:
                logging.warning(f"No recordings found for {part_name}")
                continue

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"vietASR_supervisions_{part_name}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"vietASR_recordings_{part_name}.jsonl.gz"
                )

            manifests[part_name] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests


def parse_line(
    transcript_path: Path,
    line: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    # Format: path txt duration
    # We assume space separation.
    # Text can contain spaces.
    # Duration is likely the last element.
    # Path is the first.
    
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    
    audio_path_str = parts[0]
    duration_str = parts[-1]
    text = " ".join(parts[1:-1])
    
    # Handle absolute vs relative path
    # If audio_path_str is not absolute, assume relative to transcript file directory
    audio_path = Path(audio_path_str)
    if not audio_path.is_absolute():
        audio_path = transcript_path.parent / audio_path
        
    if not audio_path.is_file():
        # Try checking if it's relative to the root of the dataset?
        # For now, strict check.
        logging.warning(f"Audio file not found: {audio_path}")
        return None
        
    try:
        duration = float(duration_str)
    except ValueError:
        logging.warning(f"Invalid duration: {duration_str}")
        return None

    recording_id = audio_path.stem
    
    # Create Recording
    # We trust the duration provided in the file to speed up loading
    # instead of reading the audio file header.
    recording = Recording(
        id=recording_id,
        sources=[
            {"type": "file", "channels": [0], "source": str(audio_path)}
        ],
        sampling_rate=16000, # Assumption, or we can read it if needed but slower
        num_samples=int(duration * 16000),
        duration=duration
    )
    
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=duration,
        channel=0,
        language="vietnamese",
        text=text,
    )
    
    return recording, segment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, help="Path to the data dir.")
    parser.add_argument(
        "--output-dir", type=Path, help="Path where to write the manifests."
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="How many threads to use.",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)

    prepare_custom_manifest(
        args.corpus_dir,
        output_dir=args.output_dir,
        num_jobs=args.num_jobs,
    )
