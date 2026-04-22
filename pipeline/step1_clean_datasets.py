"""
STEP 1: Clean each dataset
- Remove corrupt/unreadable images
- Remove empty label files
- Remove images without labels (and vice versa)
"""

import os
import glob
from pathlib import Path
from PIL import Image

DATASET_ROOTS = [
    "data/DeepFish",
    "data/Fish4Knowledge",
    "data/TrashCan",
]

def is_image_valid(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def is_label_empty(label_path):
    if not os.path.exists(label_path):
        return True
    return os.path.getsize(label_path) == 0

def clean_split(images_dir, labels_dir):
    removed = {"corrupt": 0, "no_label": 0, "empty_label": 0, "orphan_label": 0}

    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    for img_path in image_files:
        stem = Path(img_path).stem
        label_path = os.path.join(labels_dir, stem + ".txt")

        # Remove corrupt images
        if not is_image_valid(img_path):
            print(f"  [CORRUPT] Removing: {img_path}")
            os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            removed["corrupt"] += 1
            continue

        # Remove if no label exists
        if not os.path.exists(label_path):
            print(f"  [NO LABEL] Removing image: {img_path}")
            os.remove(img_path)
            removed["no_label"] += 1
            continue

        # Remove if label is empty
        if is_label_empty(label_path):
            print(f"  [EMPTY LABEL] Removing: {img_path} and {label_path}")
            os.remove(img_path)
            os.remove(label_path)
            removed["empty_label"] += 1
            continue

    # Remove orphan labels (label exists but no image)
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for label_path in label_files:
        stem = Path(label_path).stem
        found = any(
            os.path.exists(os.path.join(images_dir, stem + ext))
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]
        )
        if not found:
            print(f"  [ORPHAN LABEL] Removing: {label_path}")
            os.remove(label_path)
            removed["orphan_label"] += 1

    return removed

def clean_dataset(root):
    print(f"\n{'='*60}")
    print(f"Cleaning: {root}")
    print(f"{'='*60}")
    total = {"corrupt": 0, "no_label": 0, "empty_label": 0, "orphan_label": 0}

    for split in ["train", "valid", "val", "test"]:
        images_dir = os.path.join(root, split, "images")
        labels_dir = os.path.join(root, split, "labels")
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            print(f"\n  Split: {split}")
            result = clean_split(images_dir, labels_dir)
            for k in total:
                total[k] += result[k]

    print(f"\n  Summary for {root}:")
    for k, v in total.items():
        print(f"    {k}: {v} removed")

if __name__ == "__main__":
    for dataset in DATASET_ROOTS:
        if os.path.isdir(dataset):
            clean_dataset(dataset)
        else:
            print(f"[WARNING] Dataset not found: {dataset}")
    print("\n[DONE] Step 1 complete.")
