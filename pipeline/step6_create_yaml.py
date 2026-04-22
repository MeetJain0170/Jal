"""
STEP 6: Create data.yaml for YOLOv8 training
- Points to final train/val folders
- Defines number of classes and class names
"""

import os
import yaml
from step2_class_config import FINAL_CLASSES

FINAL_DIR  = "data/final"
YAML_PATH  = "data/marine_detection.yaml"


def create_data_yaml():
    # Use absolute paths — avoids issues when running from different CWDs
    abs_final = os.path.abspath(FINAL_DIR)
    train_path = os.path.join(abs_final, "train", "images")
    val_path   = os.path.join(abs_final, "val",   "images")

    data = {
        "path":  abs_final,
        "train": train_path,
        "val":   val_path,
        "nc":    len(FINAL_CLASSES),
        "names": FINAL_CLASSES,
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Written: {YAML_PATH}")
    print(f"  nc    : {data['nc']}")
    print(f"  train : {data['train']}")
    print(f"  val   : {data['val']}")
    print(f"  names : {data['names']}")


if __name__ == "__main__":
    create_data_yaml()
    print("\n[DONE] Step 6 complete.")
