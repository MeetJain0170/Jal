"""
STEP 5: Split merged dataset into train (80%) and val (20%)
- Input:  data/merged/images/ and data/merged/labels/
- Output: data/final/train/images|labels and data/final/val/images|labels
- Stratified by dataset prefix so each source is proportionally represented
"""

import os
import glob
import shutil
import random
from pathlib import Path
from collections import defaultdict

MERGED_IMAGES_DIR = "data/merged/images"
MERGED_LABELS_DIR = "data/merged/labels"
FINAL_DIR         = "data/final"
TRAIN_RATIO       = 0.80
SEED              = 42

SPLITS = {
    "train": os.path.join(FINAL_DIR, "train"),
    "val":   os.path.join(FINAL_DIR, "val"),
}


def setup_dirs():
    for split_dir in SPLITS.values():
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)


def copy_sample(stem, ext, split):
    src_img = os.path.join(MERGED_IMAGES_DIR, stem + ext)
    src_lbl = os.path.join(MERGED_LABELS_DIR, stem + ".txt")
    dst_img = os.path.join(SPLITS[split], "images", stem + ext)
    dst_lbl = os.path.join(SPLITS[split], "labels", stem + ".txt")
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)


def split_dataset():
    setup_dirs()
    random.seed(SEED)

    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_images = []
    for ext in img_extensions:
        all_images.extend(glob.glob(os.path.join(MERGED_IMAGES_DIR, ext)))

    # Group by dataset prefix (first token before _)
    by_prefix = defaultdict(list)
    for img_path in all_images:
        stem = Path(img_path).stem
        ext  = Path(img_path).suffix
        lbl_path = os.path.join(MERGED_LABELS_DIR, stem + ".txt")
        if not os.path.exists(lbl_path):
            print(f"  [SKIP] No label for: {stem}")
            continue
        prefix = stem.split("_")[0]
        by_prefix[prefix].append((stem, ext))

    train_total, val_total = 0, 0

    for prefix, samples in by_prefix.items():
        random.shuffle(samples)
        n_train = max(1, int(len(samples) * TRAIN_RATIO))
        train_samples = samples[:n_train]
        val_samples   = samples[n_train:]

        for stem, ext in train_samples:
            copy_sample(stem, ext, "train")
        for stem, ext in val_samples:
            copy_sample(stem, ext, "val")

        print(f"  [{prefix}] total={len(samples)} | train={len(train_samples)} | val={len(val_samples)}")
        train_total += len(train_samples)
        val_total   += len(val_samples)

    print(f"\n  Overall -> train: {train_total} | val: {val_total}")
    print(f"  Output: {FINAL_DIR}")


if __name__ == "__main__":
    split_dataset()
    print("\n[DONE] Step 5 complete.")
