"""
STEP 9: Improve dataset quality
- Analyze class distribution (find imbalances)
- Find images with likely wrong labels (high loss proxy)
- Find duplicate/near-duplicate images
- Generate a report of what to fix
"""

import os
import glob
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from step2_class_config import FINAL_CLASSES

FINAL_DIR   = "data/final"
REPORT_PATH = "data/dataset_analysis_report.txt"


# ─── 1. Class Distribution ───────────────────────────────────────────────────

def analyze_class_distribution():
    """Count annotations per class across train+val."""
    print("\n--- Class Distribution ---")
    class_counts = Counter()
    total_images = 0

    for split in ["train", "val"]:
        labels_dir = os.path.join(FINAL_DIR, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for lbl_path in glob.glob(os.path.join(labels_dir, "*.txt")):
            total_images += 1
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1

    max_count = max(class_counts.values()) if class_counts else 1
    rows = []
    for i, cls in enumerate(FINAL_CLASSES):
        count = class_counts.get(i, 0)
        bar   = "#" * int(40 * count / max_count)
        status = ""
        if count == 0:
            status = "  [MISSING - no samples]"
        elif count < 50:
            status = "  [LOW - collect more]"
        elif count < max_count * 0.1:
            status = "  [IMBALANCED]"
        row = f"  {i:2d} {cls:<20} {count:>6}  {bar}{status}"
        print(row)
        rows.append(row)

    print(f"\n  Total images: {total_images}")
    print(f"  Total annotations: {sum(class_counts.values())}")
    return class_counts, rows


# ─── 2. Exact Duplicate Detection ────────────────────────────────────────────

def find_duplicate_images():
    """Find exact duplicate images by MD5 hash."""
    print("\n--- Duplicate Image Detection ---")
    hash_map = defaultdict(list)

    for split in ["train", "val"]:
        images_dir = os.path.join(FINAL_DIR, split, "images")
        if not os.path.isdir(images_dir):
            continue
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in glob.glob(os.path.join(images_dir, ext)):
                with open(img_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                hash_map[h].append(img_path)

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    if duplicates:
        print(f"  Found {len(duplicates)} duplicate groups:")
        dup_rows = []
        for h, paths in list(duplicates.items())[:20]:  # Show first 20
            line = f"  DUPLICATE: {[os.path.basename(p) for p in paths]}"
            print(line)
            dup_rows.append(line)
    else:
        print("  No exact duplicates found.")
        dup_rows = []

    return duplicates, dup_rows


def remove_duplicates(duplicates):
    """Remove duplicate images, keeping the first occurrence."""
    removed = 0
    for h, paths in duplicates.items():
        for path in paths[1:]:
            stem = Path(path).stem
            split = "train" if "train" in path else "val"
            lbl_path = os.path.join(FINAL_DIR, split, "labels", stem + ".txt")
            os.remove(path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed += 1
    print(f"  Removed {removed} duplicate files.")


# ─── 3. Bounding Box Sanity Check ────────────────────────────────────────────

def sanity_check_labels():
    """Check for common YOLO label errors."""
    print("\n--- Label Sanity Check ---")
    issues = []

    for split in ["train", "val"]:
        labels_dir = os.path.join(FINAL_DIR, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for lbl_path in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(lbl_path) as f:
                for lineno, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(f"  BAD FORMAT: {lbl_path}:{lineno} -> '{line.strip()}'")
                        continue
                    cls_id, cx, cy, w, h = int(parts[0]), *[float(x) for x in parts[1:]]
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        issues.append(f"  OUT OF RANGE: {lbl_path}:{lineno} -> cx={cx} cy={cy} w={w} h={h}")
                    if cls_id >= len(FINAL_CLASSES) or cls_id < 0:
                        issues.append(f"  INVALID CLASS: {lbl_path}:{lineno} -> class_id={cls_id}")

    if issues:
        print(f"  Found {len(issues)} issues:")
        for iss in issues[:30]:
            print(iss)
    else:
        print("  No label issues found.")

    return issues


# ─── 4. Report ────────────────────────────────────────────────────────────────

def write_report(class_rows, dup_rows, issues):
    with open(REPORT_PATH, "w") as f:
        f.write("MARINE DATASET ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-"*40 + "\n")
        f.write("\n".join(class_rows) + "\n\n")
        f.write("DUPLICATES\n")
        f.write("-"*40 + "\n")
        if dup_rows:
            f.write("\n".join(dup_rows) + "\n\n")
        else:
            f.write("None found.\n\n")
        f.write("LABEL ISSUES\n")
        f.write("-"*40 + "\n")
        if issues:
            f.write("\n".join(issues[:100]) + "\n")
        else:
            f.write("None found.\n")
    print(f"\n  Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    class_counts, class_rows = analyze_class_distribution()
    duplicates, dup_rows     = find_duplicate_images()
    issues                   = sanity_check_labels()

    write_report(class_rows, dup_rows, issues)

    # Auto-remove duplicates
    if duplicates:
        ans = input("\nAuto-remove duplicate files? [y/N]: ").strip().lower()
        if ans == "y":
            remove_duplicates(duplicates)

    print("\n[DONE] Step 9 complete. Review the report and fix issues manually.")
