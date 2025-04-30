# Copyright      2024  Xiaomi Corporation        (authors: Yifan Yang)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Any, Dict, Optional, List, Tuple, Callable, Union

import numpy as np
import torch
import torch.nn.functional as F
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data.dataloader import default_collate
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures


def disc_to_label(text):
    pre = ""
    label = []
    for item in text.split():
        if item!=pre:
            label.append(item)
            pre = item
    return " ".join(label)

class PseudoRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task using k2 library.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - single-channel:
                        - features: (B, T, F)
                        - audio: (B, T)
                      - multi-channel: currently not supported
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S

                    # For feature input strategies
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)

                    # For audio input strategies
                    'start_sample': Tensor[int] of shape (S,)
                    'num_samples': Tensor[int] of shape (S,)

                    # Optionally, when return_cuts=True
                    'cut': List[AnyCut] of len S
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        """
        k2 ASR IterableDataset constructor.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        validate_for_asr(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": disc_to_label(cut.custom["kmeans"]),
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        has_word_alignments = all(
            s.alignment is not None and "word" in s.alignment
            for c in cuts
            for s in c.supervisions
        )
        if has_word_alignments:
            # TODO: might need to refactor BatchIO API to move the following conditional logic
            #       into these objects (e.g. use like: self.input_strategy.convert_timestamp(),
            #       that returns either num_frames or num_samples depending on the strategy).
            words, starts, ends = [], [], []
            frame_shift = cuts[0].frame_shift
            sampling_rate = cuts[0].sampling_rate
            if frame_shift is None:
                try:
                    frame_shift = self.input_strategy.extractor.frame_shift
                except AttributeError:
                    raise ValueError(
                        "Can't determine the frame_shift -- it is not present either in cuts or the input_strategy. "
                    )
            for c in cuts:
                for s in c.supervisions:
                    words.append([aliword.symbol for aliword in s.alignment["word"]])
                    starts.append(
                        [
                            compute_num_frames(
                                aliword.start,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
                    ends.append(
                        [
                            compute_num_frames(
                                aliword.end,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
            batch["supervisions"]["word"] = words
            batch["supervisions"]["word_start"] = starts
            batch["supervisions"]["word_end"] = ends

        return batch


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )

            # Supervision start time is relative to Cut ...
            # https://lhotse.readthedocs.io/en/v0.10_e/cuts.html
            #
            # 'supervision.end' is end of supervision inside the Cut
            assert supervision.end <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )


class HubertDataset(torch.utils.data.Dataset):
    """
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'features': (B, T, F) float tensor
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 100,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.random_crop = random_crop
        self.pad_feature = pad_audio
        self.num_classes = num_classes
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = [cut.num_frames for cut in cuts]

        if self.pad_feature:
            feature_size = min(max(feature_lens), self.max_sample_size)
        else:
            feature_size = min(min(feature_lens), self.max_sample_size)

        try:
            features, padding_mask, feature_starts = self.collater_feature(
                features, feature_lens, feature_size
            )
        except:
            print("--------------------------------------------------------")
            print(cuts)
            raise

        try:
            kmeans = [cut.custom["kmeans"] for cut in cuts]
        except:
            print("--------------------------------------------------------")
            for cut in cuts:
                print(cut.id)
                print(list(cut.custom))
            raise
        if type(kmeans[0]) == list:
            kmeans = [
                torch.tensor([int(item) for item in label], dtype=torch.int64)
                for label in kmeans
            ]
        else:
            kmeans = [
                torch.tensor([int(item) for item in label.split()], dtype=torch.int64)
                for label in kmeans
            ]
        kmeans, kmeans_lens = self.collater_frm_label(kmeans, feature_size, feature_starts)

        return {
            "cuts": cuts,
            "features": features,
            "padding_mask": padding_mask,
            "kmeans": kmeans,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, feature, target_size):
        size = len(feature)
        diff = size - target_size
        if diff <= 0:
            return feature, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return feature[start:end, :], start

    def collater_feature(self, features, feature_lens, feature_size):
        feature_dim = features[0].shape[-1]

        try:
            collated_features = features[0].new_zeros(len(features), feature_size, feature_dim)
        except:
            print((len(features), feature_size, feature_dim))
            raise

        padding_mask = (
            torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
            # if self.pad_feature else None
        )
        feature_starts = [0 for _ in features]
        for i, (feature, feature_len) in enumerate(zip(features, feature_lens)):
            diff = feature_len - feature_size
            if diff == 0:
                collated_features[i] = feature
            elif diff < 0:
                assert self.pad_feature
                collated_features[i] = torch.cat([feature, feature.new_full((-diff, feature_dim), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_features[i], feature_starts[i] = self.crop_to_max_size(
                    feature, feature_size
                )
        return collated_features, padding_mask, feature_starts


    def collate_tokens(
        self,
        values,
        pad_idx,
        eos_idx=None,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
        return res


    def collater_frm_label(self, kmeans, feature_size, feature_starts):
        assert self.label_rate > 0
        pad = self.num_classes[0] - 1
        s2f = self.label_rate / self.sample_rate

        frm_starts = [int(round(s * s2f)) for s in feature_starts]
        frm_size = int(round(feature_size * s2f))
        if not self.pad_feature:
            rem_size = [len(t) - s for t, s in zip(kmeans, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        kmeans = [t[s : s + frm_size] for t, s in zip(kmeans, frm_starts)]

        lengths = torch.LongTensor([len(t) for t in kmeans])
        kmeans = self.collate_tokens(kmeans, pad_idx=pad, left_pad=False)
        return kmeans, lengths


class HubertAsrDataset(torch.utils.data.Dataset):
    """
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
        }
    """

    def __init__(
        self,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 16000,
        random_crop: bool = True,
        pad_audio: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.pad_feature = pad_audio
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = [cut.num_frames for cut in cuts]

        if self.pad_feature:
            feature_size = min(max(feature_lens), self.max_sample_size)
        else:
            feature_size = min(min(feature_lens), self.max_sample_size)

        features, padding_mask, feature_starts = self.collater_feature(
            features, feature_lens, feature_size
        )

        return {
            "cuts": cuts,
            "audio": features,
            "padding_mask": padding_mask,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)

    def crop_to_max_size(self, feature, target_size):
        size = len(feature)
        diff = size - target_size
        if diff <= 0:
            return feature, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return feature[start:end, :], start

    def collater_feature(self, features, feature_lens, feature_size):
        feature_dim = features[0].shape[-1]

        try:
            collated_features = features[0].new_zeros(len(features), feature_size, feature_dim)
        except:
            print((len(features), feature_size, feature_dim))
            raise

        padding_mask = (
            torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
            # if self.pad_feature else None
        )
        feature_starts = [0 for _ in features]
        for i, (feature, feature_len) in enumerate(zip(features, feature_lens)):
            diff = feature_len - feature_size
            if diff == 0:
                collated_features[i] = feature
            elif diff < 0:
                assert self.pad_feature
                collated_features[i] = torch.cat([feature, feature.new_full((-diff, feature_dim), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_features[i], feature_starts[i] = self.crop_to_max_size(
                    feature, feature_size
                )
        return collated_features, padding_mask, feature_starts

if __name__ == "__main__":
    from lhotse import load_manifest_lazy
    from lhotse.dataset import DynamicBucketingSampler
    from torch.utils.data import DataLoader

    dataset = HubertAsrDataset(max_sample_size=1562)
    cuts = load_manifest_lazy("data/fbank/librispeech_cuts_train-clean-100.jsonl.gz")
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=300,
        shuffle=False,
    )
    dl = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=0,
    )

    for batch_idx, batch in enumerate(dl):
        print(batch["audio"].shape)
        print(batch["padding_mask"].shape)
        # print(batch["kmeans"].shape)
