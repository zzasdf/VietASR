#!/usr/bin/env python3

import sys
import os
from icefall.utils import write_error_stats
from normalizers import EnglishTextNormalizer, IndonesianTextNormalizer, ThaiTextNormalizer

cuts = []
result = dict()
def to_key(item):
    item = item.split('-')
    re = []
    for k in item:
        try:
            kk=float(k)
            re.append(kk)
        except:
            re.append(k)
    return tuple(re)

def iterate_files_in_directory(directory):
    print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            yield file_path
def record_to_data(line):
    index = line.find("=")
    return eval(line[index+1:])

def sub_normalize(refs, hyps, rnormalizer, hnormalizer, entry, dataset_name, norm_method, log_home):
    refs = [' '.join(item).lower() for item in refs]
    hyps = [' '.join(item).lower() for item in hyps]
    refs = [rnormalizer(text) for text in refs]
    hyps = [hnormalizer(text) for text in hyps]
    # data["reference_clean"] = data["reference"]

    results = []
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        results.append((i, ref.split(), hyp.split()))
    errs_filename=f"errs-{entry}-{norm_method}.txt"
    with open(os.path.join(log_home, errs_filename), "w") as f:
        wer = write_error_stats(
            f, dataset_name, results, enable_log=True
        )
        # print(f"wer for {args.method}: {wer}")
    return wer

def normalize_wer(filename: str, result_dict, cut_record):
    # read lines from file
    # compute wer
    # normal 1
    # compute wer
    # save error, no new records
    # normal 2
    # compute wer
    # save error, no new records
    # cut name = dataset_norm method
    entry = os.path.splitext(os.path.basename(filename))[0].split('-', maxsplit=1)[1]
    log_home = os.path.dirname(filename)
    cut_base = entry.split('-')[0]

    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    refs = lines[0::2]
    hyps = lines[1::2]

    refs = [record_to_data(item) for item in refs]
    hyps = [record_to_data(item) for item in hyps]

    if entry not in result:
        result[entry] = dict()

    cut = cut_base
    normalizer = lambda x:x
    if cut not in cuts:
        cuts.append(cut)
    result_dict[entry][cut] = sub_normalize(refs, hyps, normalizer, normalizer, entry, cut_base, "base", log_home)

    cut = cut_base+"_norm_word"
    normalizer = IndonesianTextNormalizer(do_number_normalize=False)
    if cut not in cuts:
        cuts.append(cut)
    result_dict[entry][cut] = sub_normalize(refs, hyps, normalizer, normalizer, entry, cut_base, "norm_word", log_home)

    cut = cut_base+"_norm_number"
    normalizer = IndonesianTextNormalizer(do_number_normalize=True)
    if cut not in cut_record:
        cut_record.append(cut)

    result_dict[entry][cut] = sub_normalize(refs, hyps, normalizer, normalizer, entry, cut_base, "norm_number", log_home)

for item in iterate_files_in_directory(sys.argv[1]):
    entry = os.path.basename(item)
    if os.path.isfile(item) and entry.startswith("recogs-"):
        normalize_wer(item, result, cuts)

cuts.sort()

for item in sorted(list(result), key=to_key):
    print(item)
    print(' '.join([f"{cut}:{result[item][cut]}" for cut in cuts if cut in result[item]]))

import os
log_dir = sys.argv[1]
with open(f"{log_dir}/wer_norm.csv", 'w') as f:
    header = "\t" + "\t".join(cuts)
    print(header, file=f)
    for item in sorted(list(result), key=to_key):
        line = []
        for cut in cuts:
            if cut in result[item]:
                line.append(f"{result[item][cut]}")
            else:
                line.append("")
        line = f"{item}\t"+"\t".join(line)
        print(line, file=f)

