# Copyright      2023  Xiaomi Corp.        (authors: Zengrui Jin)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, fbank_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - AR (10h)
            - arabic-speech-corpus (4h)
            - gale-p4 (29h)
        """
        self.fbank_dir = Path(fbank_dir)


    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # AR
        logging.info("Loading AR in lazy mode")
        AR_train_cuts = load_manifest_lazy(
            self.fbank_dir / "AR_cuts_train.jsonl.gz"
        )

        # arabic-speech-corpus
        logging.info("Loading arabic-speech-corpus in lazy mode")
        arabic_train_cuts = load_manifest_lazy(
            self.fbank_dir / "arabic-speech-corpus_cuts_train.jsonl.gz"
        )

        # gale-p4
        logging.info("Loading gale-p4 in lazy mode")
        gale_p4_train_cuts = load_manifest_lazy(
            self.fbank_dir / "gale_p4_cuts_train.jsonl.gz"
        )

        return CutSet.mux(
            AR_train_cuts,
            arabic_train_cuts,
            gale_p4_train_cuts,
            # quran_train_cuts,
            weights=[
                len(AR_train_cuts),
                len(arabic_train_cuts),
                len(gale_p4_train_cuts),
            ],
        )


    def dev_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # MASC
        logging.info("Loading MASC clean dev in lazy mode")
        masc_clean_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_clean_dev.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC noisy dev in lazy mode")
        masc_noisy_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_noisy_dev.jsonl.gz"
        )

        return CutSet.mux(
            masc_clean_dev_cuts,
            masc_noisy_dev_cuts,
            weights=[
                len(masc_clean_dev_cuts),
                len(masc_noisy_dev_cuts),
            ],
        )


    def test_cuts(self) -> Dict[str, CutSet]:
        logging.info("About to get multidataset test cuts")

        # mgb2
        logging.info("Loading mgb2 in lazy mode")
        mgb2_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "mgb2_cuts_dev.jsonl.gz"
        )

        # arabic-speech-corpus
        logging.info("Loading arabic-speech-corpus in lazy mode")
        arabic_test_cuts = load_manifest_lazy(
            self.fbank_dir / "arabic-speech-corpus_cuts_test.jsonl.gz"
        )

        # tunisian
        logging.info("Loading tunisian in lazy mode")
        tunisian_test_cuts = load_manifest_lazy(
            self.fbank_dir / "tunisian_cuts_test.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC clean dev in lazy mode")
        masc_clean_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_clean_dev.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC noisy dev in lazy mode")
        masc_noisy_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_noisy_dev.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC clean test in lazy mode")
        masc_clean_test_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_clean_test.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC noisy test in lazy mode")
        masc_noisy_test_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_noisy_test.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC MASC in lazy mode")
        masc_MSA_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_MSA.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC algeria in lazy mode")
        masc_algeria_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_algeria.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC iraq in lazy mode")
        masc_iraq_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_iraq.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC morocco in lazy mode")
        masc_morocco_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_morocco.jsonl.gz"
        )

        # MASC
        logging.info("Loading MASC saudi in lazy mode")
        masc_saudi_cuts = load_manifest_lazy(
            self.fbank_dir / "masc_cuts_saudi.jsonl.gz"
        )

        # SADA
        logging.info("Loading SADA dev in lazy mode")
        sada_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "sada_cuts_dev.jsonl.gz"
        )

        # SADA
        logging.info("Loading SADA test in lazy mode")
        sada_test_cuts = load_manifest_lazy(
            self.fbank_dir / "sada_cuts_test.jsonl.gz"
        )

        return {
            "mgb2_dev": mgb2_dev_cuts,
            "arabic-speech-corpus_test": arabic_test_cuts,
            "tunisian_test": tunisian_test_cuts,
            "masc-clean-dev": masc_clean_dev_cuts,
            "masc-noisy-dev": masc_noisy_dev_cuts,
            "masc-clean-test": masc_clean_test_cuts,
            "masc-noisy-test": masc_noisy_test_cuts,
            "masc-MSA": masc_MSA_cuts,
            "masc-algeria": masc_algeria_cuts,
            "masc-morocco": masc_morocco_cuts,
            "masc-iraq": masc_iraq_cuts,
            "masc-saudi": masc_saudi_cuts,
            "sada-dev": sada_dev_cuts,
            "sada-test": sada_test_cuts,
        }
