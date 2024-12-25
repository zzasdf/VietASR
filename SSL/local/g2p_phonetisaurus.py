#!/usr/bin/env python
import phonetisaurus
import torch
from tqdm import tqdm
import json
import gzip

src_path_list = [
    "/workdir/data/vi/ssl_finetune/fbank_50h/gigaspeech2_cuts_train.jsonl.gz",
    "/workdir/data/vi/ssl_finetune/fbank_200h/gigaspeech2-vi_cuts_dev.jsonl.gz",
    "/workdir/data/vi/ssl_finetune/fbank_200h/gigaspeech2-vi_cuts_test.jsonl.gz"
    ]
tgt_path_list = [
    "/workdir/data/vi/ssl_finetune/fbank_50h/gigaspeech2_cuts_train_phone2.jsonl.gz",
    "/workdir/data/vi/ssl_finetune/fbank_200h/gigaspeech2-vi_cuts_dev_phone2.jsonl.gz",
    "/workdir/data/vi/ssl_finetune/fbank_200h/gigaspeech2-vi_cuts_test_phone2.jsonl.gz"
    ]

def Phoneticize(model, args) :
    """Python wrapper function for g2p bindings.

    Python wrapper function for g2p bindings.  Most basic possible example.
    Intended as a template for doing something more useful.

    Args:
        model(str): The g2p fst model to load.
        args(obj): The argparse object with user specified options.
    """

    results = model.Phoneticize(
        args.token,
        args.nbest,
        args.beam,
        args.thresh,
        args.write_fsts,
        args.accumulate,
        args.pmass
    )

    for result in results :
        uniques = [model.FindOsym(u) for u in result.Uniques]
    return "".join(uniques)
        # print("{0:0.2f}\t{1}".format(result.PathWeight, " ".join(uniques)))
        # print("-------")

        #Should always be equal length
        # for ilab, olab, weight in zip(result.ILabels,
        #                                 result.OLabels,
        #                                 result.PathWeights) :
        #     print("{0}:{1}:{2:0.2f}".format(
        #         model.FindIsym(ilab),
        #         model.FindOsym(olab),
        #         weight
        #     ))



if __name__ == "__main__" :
    import argparse, sys

    parser  = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="Phonetisaurus G2P model.",
                         required=True)
    # group   = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--nbest", "-n", help="NBest",
                         default=1, type=int)
    parser.add_argument("--beam", "-b", help="Search beam",
                         default=500, type=int)
    parser.add_argument("--thresh", "-t", help="NBest threshold.",
                         default=10., type=float)
    parser.add_argument("--write_fsts", "-wf", help="Write decoded fsts "
                         "to disk", default=False, action="store_true")
    parser.add_argument("--accumulate", "-a", help="Accumulate probs across "
                         "unique pronunciations.", default=False,
                         action="store_true")
    parser.add_argument("--pmass", "-p", help="Target probability mass.",
                         default=0.0, type=float)
    args = parser.parse_args()

    model = phonetisaurus.Phonetisaurus(args.model)

    for src_path, tgt_path in zip(src_path_list, tgt_path_list):
        # src_cuts = CutSet.from_file(src_path)
        with gzip.open(src_path, 'rt') as f:
            lines = f.read().splitlines()
        src_cuts = [json.loads(line) for line in lines]


        tgt_cuts = []
        for cut in tqdm(src_cuts):
            words = cut['supervisions'][0]['text'].lower().split()
            phones = []
            for word in words:
                args.token = word
                phones.append(Phoneticize(model, args))

            # words = ['<vie-c>: '+words]
            # words = ["".join(words)]
            cut['supervisions'][0]['text'] = ' '.join(phones)
            tgt_cuts.append(cut)
        
        tgt_cuts = [json.dumps(cut, ensure_ascii=False) for cut in tgt_cuts]
        with gzip.open(tgt_path, 'wt') as f:
            for line in tgt_cuts:
                print(line, file=f)