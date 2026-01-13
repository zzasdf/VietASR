#!/usr/bin/env python3
"""
Script gộp manifests từ nhiều dataset với đường dẫn train/test riêng biệt.
TƯƠNG THÍCH VỚI LHOTSE 1.31.1.dev (sửa lỗi lazy iterator sau shuffle)
"""

from pathlib import Path
from typing import List, Tuple
from lhotse import CutSet, load_manifest_lazy
from lhotse.utils import fix_random_seed


TRAIN_DIR = Path("manifests/")
TEST_DIR = Path("manifests/")
TRAIN_DATASETS: List[str] = ["regions_63","regions.bac_trung_bo","regions.bac_trung_bo_112024","regions.dong_nam_bo","tongdai","tongdai_02112022","tongdai_25112024","regions.tongdai"]
TEST_DATASETS: List[str] = []
TRAIN_RATIO = 0.95
OUTPUT_DIR = Path("data4/manifests")
# ====== END CONFIG ======


def load_dataset(data_dir: Path, name: str) -> CutSet:
    """Load một dataset từ recordings và supervisions."""
    rec_file = data_dir / name / f"recordings_{name}.jsonl.gz"
    sup_file = data_dir / name / f"supervisions_{name}.jsonl.gz"
    
    if not rec_file.exists() or not sup_file.exists():
        raise FileNotFoundError(f"Dataset {name} thiếu file: {rec_file} hoặc {sup_file}")
    
    print(f"  📂 Loading {data_dir.name}/{name}...")
    return CutSet.from_manifests(
        recordings=load_manifest_lazy(rec_file),
        supervisions=load_manifest_lazy(sup_file)
    )


def merge_datasets(data_dir: Path, names: List[str]) -> CutSet:
    """Gộp nhiều dataset từ cùng một folder thành một CutSet."""
    if not names:
        print("  ⚠️  Không có dataset nào để gộp!")
        return CutSet([])
    
    cuts_list = [load_dataset(data_dir, name) for name in names]
    print(f"  🔀 Gộp {len(names)} datasets từ {data_dir.name}...")
    
    # TƯƠNG THÍCH MỌI PHIÊN BẢN: Dùng toán tử + qua sum()
    return sum(cuts_list, CutSet([]))


def split_train_dev(cuts: CutSet, train_ratio: float) -> Tuple[CutSet, CutSet]:
    """Chia CutSet thành train và dev theo tỷ lệ.
    
    FIX CHO LHOTSE 1.31.1.dev:
    - Sau shuffle() trở thành lazy iterator
    - Phải dùng subset(first=N) thay vì slicing
    - Hoặc materialize toàn bộ với to_eager()
    """
    if len(cuts) == 0:
        return CutSet([]), CutSet([])
    
    # Set seed toàn cục trước khi shuffle
    fix_random_seed(42)
    cuts = cuts.shuffle()  # Trả về lazy iterator
    
    # SOLUTION 1: Dùng subset() với first/last
    total = len(cuts)
    train_size = int(train_ratio * total)
    
    # Lấy train_size phần tử đầu
    train_cuts = cuts.subset(first=train_size)
    
    # Lấy phần còn lại: skip train_size phần tử đầu
    dev_cuts = cuts.subset(last=total - train_size)
    
    return train_cuts, dev_cuts


def main():
    print("=" * 70)
    print("Bắt đầu gộp manifests...")
    print("Lhotse version: 1.31.1.dev (fix lazy iterator)")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📍 Train folder: {TRAIN_DIR}")
    print(f"📍 Test folder:  {TEST_DIR}")
    
    # ===== XỬ LÝ TRAIN/DEV =====
    print("\n🎯 Xử lý train/dev datasets:")
    if TRAIN_DATASETS:
        print(f"  Danh sách: {', '.join(TRAIN_DATASETS)}")
        
        combined_cuts = merge_datasets(TRAIN_DIR, TRAIN_DATASETS)
        print(f"  Tổng số utterances: {len(combined_cuts)}")
        
        train_cuts, dev_cuts = split_train_dev(combined_cuts, TRAIN_RATIO)
        print(f"  → Train: {len(train_cuts)} utterances ({TRAIN_RATIO:.0%})")
        print(f"  → Dev: {len(dev_cuts)} utterances ({1-TRAIN_RATIO:.0%})")
        
        train_cuts.to_file(OUTPUT_DIR / "vietASR_cuts_train.jsonl.gz")
        dev_cuts.to_file(OUTPUT_DIR / "vietASR_cuts_dev.jsonl.gz")
        print("  ✅ Đã lưu train/dev cuts")
    else:
        print("  ⚠️  Không có dataset nào cho train/dev")
    
    # ===== XỬ LÝ TEST =====
    print("\n🧪 Xử lý test datasets:")
    if TEST_DATASETS:
        print(f"  Danh sách: {', '.join(TEST_DATASETS)}")
        
        test_cuts = merge_datasets(TEST_DIR, TEST_DATASETS)
        print(f"  Tổng số utterances: {len(test_cuts)}")
        
        test_cuts.to_file(OUTPUT_DIR / "vietASR_cuts_test.jsonl.gz")
        print("  ✅ Đã lưu test cuts")
    else:
        print("  ⚠️  Không có dataset nào cho test")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH!")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()