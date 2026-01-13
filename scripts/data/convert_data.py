#!/usr/bin/env python3
"""
Convert data to Lhotse manifests with deduplication and filtering logic.
Compatible with ESPnet preprocessing (load_data_test.py)
"""
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)


def create_utterance_id(filename: str) -> str:
    """
    Create utterance ID from filename using the same logic as old PrepareData:
    wav = '.'.join(dt[0].split('/')[-1].split('.')[:-1])
    """
    basename = Path(filename).name
    parts = basename.split('.')
    if len(parts) > 1:
        return '.'.join(parts[:-1])
    return basename


def clean_text_espnet(text: str) -> str:
    """
    Clean text using ESPnet logic:
    - Remove ' ck' and ' spkt' suffixes
    - Lowercase
    """
    text = text.lower()
    text = text.replace(' ck', '').replace(' spkt', '')
    return text.strip()


def should_skip_sample(
    wav_path: str,
    text: str,
    duration: float,
    min_duration: float = 1.0,
    max_duration: float =31.0,
    min_file_size: int = 10000,
    skip_patterns: list = None,
) -> Tuple[bool, str]:
    """
    Check if a sample should be skipped based on ESPnet filtering logic.
    
    Returns:
        Tuple of (should_skip, reason)
    """
    if skip_patterns is None:
        skip_patterns = ['111']
    
    # Filter 1: File doesn't exist
    if not os.path.exists(wav_path):
        return True, "file_not_found"
    
    # Filter 2: File size too small (< 10KB)
    try:
        if os.path.getsize(wav_path) <= min_file_size:
            return True, f"file_size_too_small (<= {min_file_size} bytes)"
    except OSError:
        return True, "file_size_check_failed"
    
    # Filter 3: Duration out of range
    if duration < min_duration:
        return True, f"duration_too_short (< {min_duration}s)"
    if duration >= max_duration:
        return True, f"duration_too_long (>= {max_duration}s)"
    
    # Filter 4: Text contains skip patterns (e.g., "111")
    for pattern in skip_patterns:
        if pattern in text:
            return True, f"text_contains_pattern: '{pattern}'"
    
    return False, ""


def prepare_dataset(
    dataset_path: Pathlike,
    output_dir: Pathlike,
    language: str = "vi",
    normalize_text: str = "none",
    num_jobs: int = 1,
    resample_rate: Optional[int] = None,
    deduplicate: bool = True,
    apply_espnet_filter: bool = True,  # NEW: Enable ESPnet-style filtering
    min_duration: float = 1.0,
    max_duration: float = 31.0,
):
    """
    Chuẩn bị manifest cho dataset với cấu trúc:
    - dataset_folder/
        - transcripts.txt (format: wav_path | text | duration)
        - wavs/ (chứa audio)
    
    Args:
        deduplicate: If True, remove duplicate entries based on utterance ID
        apply_espnet_filter: If True, apply ESPnet-style filtering:
            - Skip files < 10KB
            - Skip duration < 1s or >= 15s
            - Skip text containing "111"
            - Clean " ck" and " spkt" from text
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    dataset_name = dataset_path.name

    assert dataset_path.is_dir(), f"Không tìm thấy thư mục: {dataset_path}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Đang xử lý dataset: {dataset_name} tại {dataset_path}")
    if apply_espnet_filter:
        logging.info(f"ESPnet filtering ENABLED: duration=[{min_duration}, {max_duration}), file_size>=10KB, skip '111'")

    recordings_output_path = output_dir / f"recordings_{dataset_name}.jsonl.gz"
    supervisions_output_path = output_dir / f"supervisions_{dataset_name}.jsonl.gz"

    recordings = []
    supervisions = []
    futures = []
    
    # Track filter statistics
    filter_stats = {
        "file_not_found": 0,
        "file_size_too_small": 0,
        "duration_too_short": 0,
        "duration_too_long": 0,
        "text_contains_pattern": 0,
        "parse_error": 0,
    }

    # Tìm file transcripts.txt
    trans_file = dataset_path / "transcripts.txt"
    if not trans_file.exists():
        txt_files = list(dataset_path.glob("*.txt"))
        if txt_files:
            trans_file = txt_files[0]
            logging.warning(f"Không thấy transcripts.txt, đang dùng: {trans_file.name}")
        else:
            logging.error(f"Không tìm thấy file text nào trong {dataset_path}")
            return

    logging.info(f"Đọc transcript từ: {trans_file}")

    with ThreadPoolExecutor(num_jobs) as ex:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Đọc dòng"):
                if not line.strip():
                    continue
                futures.append(
                    ex.submit(
                        parse_utterance, 
                        dataset_path, 
                        line, 
                        language,
                        apply_espnet_filter,
                        min_duration,
                        max_duration,
                    )
                )

        for future in tqdm(futures, desc="Xử lý audio", leave=False):
            result = future.result()
            if result is None:
                filter_stats["parse_error"] += 1
                continue
            if isinstance(result, str):
                # It's a skip reason
                for key in filter_stats:
                    if key in result:
                        filter_stats[key] += 1
                        break
                continue
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

    if not recordings:
        logging.error("Không tạo được dữ liệu nào.")
        return

    original_count = len(recordings)
    logging.info(f"Tìm thấy {original_count} mẫu dữ liệu hợp lệ.")
    
    # Log filter statistics
    if apply_espnet_filter:
        total_skipped = sum(filter_stats.values())
        logging.info(f"ESPnet Filter Results (total skipped: {total_skipped}):")
        for reason, count in filter_stats.items():
            if count > 0:
                logging.info(f"  - {reason}: {count}")

    # ========== DEDUPLICATION LOGIC ==========
    if deduplicate:
        logging.info("Đang loại bỏ duplicate entries (giống sort -k1,1 -u)...")
        
        seen_ids = set()
        unique_recordings = []
        unique_supervisions = []
        
        paired = list(zip(recordings, supervisions))
        paired.sort(key=lambda x: x[0].id)
        
        for rec, sup in paired:
            if rec.id not in seen_ids:
                seen_ids.add(rec.id)
                unique_recordings.append(rec)
                unique_supervisions.append(sup)
        
        recordings = unique_recordings
        supervisions = unique_supervisions
        
        removed_count = original_count - len(recordings)
        if removed_count > 0:
            logging.warning(f"Đã loại bỏ {removed_count} duplicate entries.")
        logging.info(f"Còn lại {len(recordings)} mẫu dữ liệu sau deduplication.")

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    # --- RESAMPLE ---
    if resample_rate is not None:
        logging.info(f"Đang cấu hình Resample tự động về {resample_rate}Hz...")
        recording_set = recording_set.resample(resample_rate)

    # --- NORMALIZE TEXT ---
    if normalize_text == "lower":
        to_lower = lambda text: text.lower()
        supervision_set = SupervisionSet.from_segments(
            [s.transform_text(to_lower) for s in supervision_set]
        )
    elif normalize_text == "lower_tone":
        import sys
        sys.path.append(str(Path(__file__).parent / "utils"))
        from normalize_hybrid import normalize_hybrid_tone
        
        def normalize_fn(text):
            return normalize_hybrid_tone(text.lower())
            
        supervision_set = SupervisionSet.from_segments(
            [s.transform_text(normalize_fn) for s in supervision_set]
        )

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    logging.info(f"Đang lưu manifests vào: {output_dir}")
    recording_set.to_file(recordings_output_path)
    supervision_set.to_file(supervisions_output_path)

    logging.info(f"Hoàn tất!\n - {recordings_output_path}\n - {supervisions_output_path}")
    logging.info(f"Final sample count: {len(recording_set)}")


def parse_utterance(
    dataset_path: Path,
    line: str,
    language: str,
    apply_espnet_filter: bool = True,
    min_duration: float = 1.0,
    max_duration: float = 31.0,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    parts = line.strip().split('|')
    if len(parts) < 2:
        return None
    
    rel_wav_path = parts[0].strip()
    text = parts[1].strip()
    
    # Parse duration if available
    duration = None
    if len(parts) >= 3:
        try:
            duration = float(parts[2].strip())
        except ValueError:
            pass

    filename = Path(rel_wav_path).name
    audio_path = dataset_path / "wavs" / filename
    
    if not audio_path.exists():
        # Try using the path as-is
        if rel_wav_path.startswith('/'):
            audio_path = Path(rel_wav_path)
        else:
            audio_path = dataset_path / rel_wav_path
        
    if not audio_path.is_file():
        return "file_not_found"
    
    audio_path = audio_path.resolve()
    
    # Apply ESPnet-style filtering
    if apply_espnet_filter:
        # Clean text first (remove ' ck' and ' spkt')
        cleaned_text = clean_text_espnet(text)
        
        # Get duration from file if not provided
        if duration is None:
            try:
                import soundfile as sf
                info = sf.info(str(audio_path))
                duration = info.duration
            except Exception:
                try:
                    from lhotse.audio import AudioSource
                    audio_source = AudioSource.from_file(str(audio_path))
                    duration = audio_source.duration
                except Exception:
                    return "parse_error"
        
        # Apply filters
        should_skip, reason = should_skip_sample(
            str(audio_path),
            cleaned_text,  # Use cleaned text for pattern matching
            duration,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        
        if should_skip:
            return reason
        
        # Use cleaned text
        text = cleaned_text
    
    recording_id = create_utterance_id(str(audio_path))

    try:
        recording = Recording.from_file(audio_path, recording_id=recording_id)
        
        segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language=language,
            speaker=recording_id,
            text=text,
        )
        return recording, segment
    except Exception as e:
        logging.warning(f"Error reading {filename}: {e}")
        return "parse_error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo Lhotse Manifest với Deduplication và ESPnet Filtering")
    parser.add_argument("corpus_dir", type=Path, help="Thư mục chứa các dataset")
    parser.add_argument("output_dir", type=Path, help="Thư mục output")
    parser.add_argument("--language", type=str, default="vi", help="Ngôn ngữ")
    parser.add_argument("--normalize-text", type=str, default="none", choices=["none", "lower", "lower_tone"])
    parser.add_argument("--num-jobs", type=int, default=4, help="Số threads")
    parser.add_argument("--resample-rate", type=int, default=16000, help="Sample rate. Đặt 0 để giữ nguyên.")
    
    # Deduplication option
    parser.add_argument("--no-deduplicate", action="store_true", help="Tắt deduplication")
    
    # ESPnet filtering options
    parser.add_argument("--no-espnet-filter", action="store_true", help="Tắt ESPnet-style filtering")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Min duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=31.0, help="Max duration (seconds)")
    
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs="+",
        default=None,
        help="Danh sách tên dataset cần xử lý"
    )
    
    args = parser.parse_args()

    target_sr = args.resample_rate if args.resample_rate > 0 else None
    deduplicate = not args.no_deduplicate
    apply_espnet_filter = not args.no_espnet_filter

    corpus_dir = Path(args.corpus_dir)
    
    if args.datasets:
        dataset_paths = []
        for dataset_name in args.datasets:
            dataset_path = corpus_dir / dataset_name
            if dataset_path.is_dir():
                dataset_paths.append(dataset_path)
            else:
                logging.warning(f"Không tìm thấy dataset: {dataset_path}")
    else:
        dataset_paths = [d for d in corpus_dir.iterdir() if d.is_dir()]
        logging.info(f"Tìm thấy {len(dataset_paths)} dataset trong {corpus_dir}")
    
    if not dataset_paths:
        logging.error("Không có dataset nào để xử lý!")
        exit(1)
    
    for dataset_path in dataset_paths:
        logging.info(f"\n{'='*60}")
        logging.info(f"Đang xử lý dataset: {dataset_path.name}")
        logging.info(f"{'='*60}\n")
        
        prepare_dataset(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            language=args.language,
            normalize_text=args.normalize_text,
            num_jobs=args.num_jobs,
            resample_rate=target_sr,
            deduplicate=deduplicate,
            apply_espnet_filter=apply_espnet_filter,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Hoàn tất xử lý {len(dataset_paths)} dataset!")
    logging.info(f"{'='*60}")
