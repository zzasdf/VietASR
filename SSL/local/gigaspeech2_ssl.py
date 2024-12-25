import json
import logging
import os
import glob
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations


def _parse_utterance(
    part_path: Pathlike,
    line: str,
    lang: Optional[str] = None,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id = f"{os.path.basename(os.path.dirname(str(part_path)))}-{os.path.splitext(os.path.basename(str(part_path)))[0]}"
    text = ""
    audio_path = part_path
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=recording_id,
    )

    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=lang,
        text=text.strip(),
    )

    return recording, segment


def _prepare_subset(
    corpus_dir: Pathlike,
    lang: Optional[str] = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    wav_paths = glob.iglob(f"{str(corpus_dir)}/**/*.wav", recursive=True)

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for wav_path in tqdm(wav_paths, desc="Distributing tasks"):
            futures.append(ex.submit(_parse_utterance, Path(wav_path), "", lang))

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_vietnam(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    part: str = "train",
    lang: Optional[str] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the Gigaspeech2 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing Gigaspeech2...")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    logging.info(f"Processing Gigaspeech2 subset: {part}")
    if manifests_exist(
        part=part,
        output_dir=output_dir,
        prefix="gigaspeech2-ssl",
        suffix="jsonl.gz",
    ):
        logging.info(f"Gigaspeech2 subset: {part} already prepared - skipping.")
        return manifests

    recording_set, supervision_set = _prepare_subset(
        corpus_dir, lang, num_jobs
    )

    if output_dir is not None:
        supervision_set.to_file(
            output_dir / f"gigaspeech2-ssl_supervisions_{part}.jsonl.gz"
        )
        recording_set.to_file(
            output_dir / f"gigaspeech2-ssl_recordings_{part}.jsonl.gz"
        )

    manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests

if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("corpus_dir", type=str)
    parser.add_argument("part", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("-j", "--num-jobs", type=int, default=1)
    args = parser.parse_args()
    prepare_vietnam(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        part = args.part,
        lang=args.lang,
        num_jobs=args.num_jobs,
    )


# @prepare.command(context_settings=dict(show_default=True))
# @click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
# @click.argument("output_dir", type=click.Path())
# @click.option("--lang", type=str)
# @click.option(
#     "-j",
#     "--num-jobs",
#     type=int,
#     default=1,
#     help="How many threads to use (can give good speed-ups with slow disks).",
# )
# def gigaspeech2(
#     corpus_dir: Pathlike,
#     output_dir: Optional[Pathlike] = None,
#     lang: Optional[str] = None,
#     num_jobs: int = 1,
# ):
#     """GigaSpeech2 data preparation."""
#     prepare_gigaspeech2(
#         corpus_dir=corpus_dir,
#         output_dir=output_dir,
#         lang=lang,
#         num_jobs=num_jobs,
#     )
