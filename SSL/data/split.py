import os, sys, random
import lhotse


def split_cuts(manifest_dir, output_dir, num_splits=8):
    random.seed(142)
    print(f"About to get train cuts from {manifest_dir}")
    if os.path.isfile(manifest_dir):
        cut_list = [manifest_dir]
        manifest_dir = os.path.dirname(manifest_dir)

    elif os.path.isdir(manifest_dir):
        pool_list = os.listdir(manifest_dir)
        cut_list = [os.path.join(manifest_dir, item) for item in pool_list if item.endswith('jsonl.gz')]
        cut_list = sorted(cut_list)
        random.shuffle(cut_list)

    print(f"Loading {len(cut_list)} splits in lazy mode")

    cuts = lhotse.combine(lhotse.load_manifest_lazy(p) for p in cut_list)
    print(len(cuts))

    splits = cuts.split(num_splits=num_splits, shuffle=True, drop_last=False)
    for i, cut in enumerate(splits):
        print(len(cut))
        cut.to_file(f'{output_dir}/pretraining_split_{i}.jsonl.gz')


if __name__ == "__main__":
    manifest = sys.argv[1]
    output = sys.argv[2]
    split_num = int(sys.argv[3])
    split_cuts(manifest, output, split_num)
