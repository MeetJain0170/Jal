"""
STEP 4: Merge all datasets into one unified folder
- Combines all images into data/merged/images/
- Combines all labels into data/merged/labels/
- Adds dataset prefix to filenames to avoid duplicates
- Handles train + valid/val splits from each dataset
"""

import os
import glob
import shutil
from pathlib import Path
from step2_class_config import DATASET_CONFIG

MERGED_IMAGES_DIR = "data/merged/images"
MERGED_LABELS_DIR = "data/merged/labels"

# Prefix used per dataset to avoid filename collisions
DATASET_PREFIX = {
    "DeepFish":       "df",
    "Fish4Knowledge": "f4k",
    "TrashCan":       "tc",
}


def merge_split(dataset_name, split_dir, merged_img_dir, merged_lbl_dir):
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return 0

    prefix = DATASET_PREFIX.get(dataset_name, dataset_name.lower())
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    copied = 0
    skipped = 0

    for img_path in image_files:
        stem = Path(img_path).stem
        ext = Path(img_path).suffix
        label_path = os.path.join(labels_dir, stem + ".txt")

        if not os.path.exists(label_path):
            skipped += 1
            continue

        new_stem = f"{prefix}_{stem}"
        new_img_dst = os.path.join(merged_img_dir, new_stem + ext)
        new_lbl_dst = os.path.join(merged_lbl_dir, new_stem + ".txt")

        # Handle rare collision (e.g., same filename across splits of same dataset)
        counter = 0
        while os.path.exists(new_img_dst) or os.path.exists(new_lbl_dst):
            counter += 1
            new_img_dst = os.path.join(merged_img_dir, f"{new_stem}_{counter}{ext}")
            new_lbl_dst = os.path.join(merged_lbl_dir, f"{new_stem}_{counter}.txt")

        shutil.copy2(img_path, new_img_dst)
        shutil.copy2(label_path, new_lbl_dst)
        copied += 1

    print(f"    Copied: {copied} | Skipped (no label): {skipped}")
    return copied


def merge_all_datasets():
    os.makedirs(MERGED_IMAGES_DIR, exist_ok=True)
    os.makedirs(MERGED_LABELS_DIR, exist_ok=True)

    total = 0

    for name, config in DATASET_CONFIG.items():
        root = config["root"]
        if not os.path.isdir(root):
            print(f"[WARNING] Skipping {name}: not found at {root}")
            continue

        print(f"\n{'='*60}")
        print(f"Merging: {name}")
        print(f"{'='*60}")

        for split in ["train", "valid", "val", "test"]:
            split_dir = os.path.join(root, split)
            if os.path.isdir(split_dir):
                print(f"  Split: {split}")
                count = merge_split(name, split_dir, MERGED_IMAGES_DIR, MERGED_LABELS_DIR)
                total += count

    print(f"\n{'='*60}")
    print(f"Total merged samples: {total}")
    print(f"Images -> {MERGED_IMAGES_DIR}")
    print(f"Labels -> {MERGED_LABELS_DIR}")


if __name__ == "__main__":
    merge_all_datasets()
    print("\n[DONE] Step 4 complete.")
