"""
STEP 3: Convert labels to YOLO format and remap class IDs
- All datasets assumed to already be in YOLO format (.txt)
- This step remaps class IDs based on DATASET_CONFIG
- Drops annotations for classes mapped to None
- Saves remapped labels in-place (backs up originals first)
"""

import os
import glob
import shutil
from pathlib import Path
from step2_class_config import DATASET_CONFIG


def remap_label_file(src_path, dst_path, class_map):
    """Read a YOLO label file, remap class IDs, write result."""
    new_lines = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            orig_class = int(parts[0])
            new_class = class_map.get(orig_class, None)
            if new_class is None:
                continue  # Drop this annotation
            new_line = f"{new_class} " + " ".join(parts[1:])
            new_lines.append(new_line)

    # Write only if there are annotations remaining
    if new_lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        return True
    else:
        # Empty after remapping — remove or create empty
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False


def backup_labels(labels_dir):
    """Backup original labels to labels_backup/"""
    backup_dir = labels_dir + "_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(labels_dir, backup_dir)
        print(f"    Backed up to: {backup_dir}")
    else:
        print(f"    Backup already exists: {backup_dir}")


def remap_dataset(name, config):
    root = config["root"]
    class_map = config["class_map"]

    print(f"\n{'='*60}")
    print(f"Remapping: {name}  |  root: {root}")
    print(f"{'='*60}")

    total_processed = 0
    total_dropped_files = 0

    for split in ["train", "valid", "val", "test"]:
        labels_dir = os.path.join(root, split, "labels")
        images_dir = os.path.join(root, split, "images")
        if not os.path.isdir(labels_dir):
            continue

        print(f"\n  Split: {split}")
        backup_labels(labels_dir)

        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        kept, dropped = 0, 0

        for label_path in label_files:
            stem = Path(label_path).stem
            had_content = remap_label_file(label_path, label_path, class_map)

            if not had_content:
                # Also remove the corresponding image if label becomes empty
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    img_path = os.path.join(images_dir, stem + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        break
                os.remove(label_path) if os.path.exists(label_path) else None
                dropped += 1
            else:
                kept += 1

        total_processed += kept + dropped
        total_dropped_files += dropped
        print(f"    Kept: {kept} | Dropped (all annotations removed): {dropped}")

    print(f"\n  Total files processed: {total_processed}")
    print(f"  Total files dropped: {total_dropped_files}")


if __name__ == "__main__":
    for name, config in DATASET_CONFIG.items():
        if os.path.isdir(config["root"]):
            remap_dataset(name, config)
        else:
            print(f"[WARNING] Skipping {name}: directory not found at {config['root']}")

    print("\n[DONE] Step 3 complete. Labels remapped.")
