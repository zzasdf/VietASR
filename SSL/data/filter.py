import os, sys
import lhotse


def filter_perturbed_speed(cut):
    if cut.id.endswith('sp1.1') or cut.id.endswith('sp0.9'):
        return False
    return True


if __name__ == "__main__":
    cuts = lhotse.load_manifest_lazy('asr_45h/mgb2_cuts_train.jsonl.gz')
    cuts = cuts.filter(filter_perturbed_speed)
    # print(len(cuts.to_eager()))
    print(cuts.describe())

    splits = cuts.split(num_splits=6, shuffle=True, drop_last=False)
    for i, cut in enumerate(splits):
        print(len(cut))
        cut.to_file(f'asr_45h/mgb2_cuts_train_{i}.jsonl.gz')
