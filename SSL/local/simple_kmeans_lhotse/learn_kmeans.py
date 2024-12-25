# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from lhotse import CutSet
from lhotse.workarounds import Hdf5MemoryIssueFix

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(cut_file, percent):
    cuts = CutSet.from_file(cut_file)
    feat_lis = []
    batch_size = 64
    log_step = 1000
    hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
    tem_batch = []
    if percent < 0:
        for i, cut in enumerate(cuts): 
            if i% log_step==0:
                logging.info(f"loaded cut {i}")

            if i%batch_size==0:
                hdf5_fix.update()
                if len(tem_batch)>0:
                    feat_lis.append(np.concatenate(
                        tem_batch,
                        axis=0,
                    ))
                tem_batch = []
            tem_batch.append(cut.load_features())
        
        if len(tem_batch)>0:
            feat_lis.append(np.concatenate(
                tem_batch,
                axis=0,
            ))
    else:
        large_batch_size = int(batch_size/percent)
        for i, cut in enumerate(cuts): 
            if i% log_step==0:
                logging.info(f"loaded cut {i}")

            if i%large_batch_size==0:
                hdf5_fix.update()
                if len(tem_batch)>0:
                    indices = np.random.choice(large_batch_size, batch_size, replace=False)
                    tem_batch = [tem_batch[i] for i in indices]
                    feat_lis.append(np.concatenate(
                        tem_batch,
                        axis=0,
                    ))
                    tem_batch = []
            # if i>=100:
            #     break
            tem_batch.append(cut.load_features())

        if len(tem_batch)>0:
            indices = np.random.choice(len(tem_batch), int(len(tem_batch)*percent), replace=False)
            tem_batch = [tem_batch[i] for i in indices]
            feat_lis.append(np.concatenate(
                tem_batch,
                axis=0,
            ))
    
    feat = np.concatenate(
        feat_lis,
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat

def load_feature(cut_files, src_dir, percent):
    if cut_files is None:
        cut_files = os.listdir(src_dir)
        cut_files = [os.path.join(src_dir, item) for item in cut_files if item.endswith(".jsonl.gz") and item.find("_raw")<=0]
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(cut_file, percent)
            for cut_file in cut_files
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    files,
    src_dir,
    km_path,
    percent,
    n_clusters,
    seed,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    feat = load_feature(files, src_dir, percent)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("km_path", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--files", type=str, nargs="*", default = None)
    parser.add_argument("--src_dir", type=str, default = None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
