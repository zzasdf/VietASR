import argparse
import logging
import re
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm

VIETASR = (
   # "dev",
    "test",
   # "train",
)


def prepare_manifest(
    corpus_dir: Pathlike,
    language="en",
    output_dir: Optional[Pathlike] = None,
    normalize_text: str = "none",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: str, "none" or "lower",
        for "lower" the transcripts are converted to lower-case.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = set(VIETASR).intersection(
        path.name for path in corpus_dir.glob("*")
    )
    if not dataset_parts:
        raise ValueError(f"Could not find any of splits in: {corpus_dir}")

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir, prefix="vietASR"):
                logging.info(f"Subset: {part} already prepared - skipping.")
                continue
            recordings = []
            supervisions = []
            part_path = corpus_dir / part
            futures = []
            for trans_path in tqdm(
                part_path.rglob("*.trans.txt"), desc="Distributing tasks", leave=False
            ):
                # "trans_path" file contains lines like:
                #
                #   121-121726-0000 ALSO A POPULAR CONTRIVANCE
                #   121-121726-0001 HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE
                #   121-121726-0002 ANGOR PAIN PAINFUL TO HEAR
                #
                # We will create a separate Recording and SupervisionSegment for those.
                with open(trans_path) as f:
                    for line in f:
                        futures.append(
                            ex.submit(parse_utterance, trans_path, line, language)
                        )

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            # Normalize text to lowercase
            if normalize_text == "lower":
                to_lower = lambda text: text.lower()
                supervision_set = SupervisionSet.from_segments(
                    [s.transform_text(to_lower) for s in supervision_set]
                )

            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"vietASR_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"vietASR_recordings_{part}.jsonl.gz"
                )

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests


def parse_utterance(
    script_path: Path,
    line: str,
    language: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # Create the Recording first
    audio_path = script_path.parent / f"{recording_id}.wav"
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
        speaker=re.sub(r"-.*", r"", recording.id),
        text=text.strip(),
    )
    return recording, segment


def run(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    lanugage: str,
    normalize_text: str,
    num_jobs: int,
):
    prepare_manifest(
        corpus_dir,
        output_dir=output_dir,
        language=lanugage,
        num_jobs=num_jobs,
        normalize_text=normalize_text,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, help="Path to the data dir.")
    parser.add_argument(
        "--output-dir", type=Path, help="Path where to write the manifests."
    )
    parser.add_argument("--language", type=str, help="dataset language")
    parser.add_argument(
        "--normalize-text",
        type=str,
        help="Conversion of transcripts to lower-case (originally in upper-case)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="How many threads to use (can give good speed-ups with slow disks).",
    )
    args = parser.parse_args()

    run(
        args.corpus_dir,
        args.output_dir,
        args.language,
        args.normalize_text,
        args.num_jobs,
    )
