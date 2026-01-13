
import logging
from pathlib import Path
from lhotse.recipes.utils import read_manifests_if_cached

logging.basicConfig(level=logging.INFO)

dataset_parts = ("test",)
src_dir = Path("../data6/bactrungbo/manifests")
prefix = "vietASR"
suffix = "jsonl.gz"

print(f"Checking in: {src_dir.resolve()}")
print(f"Looking for parts: {dataset_parts}")

manifests = read_manifests_if_cached(
    dataset_parts=dataset_parts,
    output_dir=src_dir,
    prefix=prefix,
    suffix=suffix,
)

print(f"Result: {manifests}")

# Manual check
rec_path = src_dir / f"{prefix}_recordings_test.{suffix}"
sup_path = src_dir / f"{prefix}_supervisions_test.{suffix}"
print(f"Recordings path: {rec_path} exists? {rec_path.exists()}")
print(f"Supervisions path: {sup_path} exists? {sup_path.exists()}")
