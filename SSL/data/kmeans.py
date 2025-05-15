import os, sys
import lhotse


def filter_perturbed_speed(cut):
    if cut.id.endswith('sp1.1'):
        return True
    return False


if __name__ == "__main__":
    cutset = lhotse.load_manifest_lazy('data/pretrain/pretraining_split_7_km_iter1.jsonl.gz')
    # cutset = cutset.filter(filter_perturbed_speed).to_eager()
    print(len(cutset))
    for cut in cutset:
        kmeans = cut.custom['kmeans']
        # print(len(kmeans.split()))
        # print(cut.duration)     # 50Hz label

        feat = cut.load_features()
        # print(feat.shape)       # (100*dur, 80) 100Hz feature

        if len(kmeans.split()) > 53 * float(cut.duration):
            print(f'{cut.duration}\t{len(kmeans.split())}\t{feat.shape}\t{cut.id}')
