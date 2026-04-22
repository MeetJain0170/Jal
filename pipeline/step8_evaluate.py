"""
STEP 8: Evaluate the trained model on validation set
- Computes mAP50, mAP50-95, precision, recall per class
- Saves prediction visualizations for manual inspection
- Flags classes with poor performance
"""

import os
import glob
import random
from pathlib import Path
from ultralytics import YOLO
from step2_class_config import FINAL_CLASSES

# ─── CONFIG ─────────────────────────────────────────────────────────────────
WEIGHTS_PATH  = "runs/marine/v1/weights/best.pt"
DATA_YAML     = "data/marine_detection.yaml"
VAL_IMG_DIR   = "data/final/val/images"
EVAL_OUTPUT   = "runs/marine/v1/eval"
N_PREVIEW     = 50    # Number of val images to save predictions for
CONF_THRESH   = 0.25
IOU_THRESH    = 0.5
MAP_WARN_THRESH = 0.4  # Flag classes with mAP50 below this
# ────────────────────────────────────────────────────────────────────────────


def run_validation(model):
    """Full dataset validation with metrics."""
    print("\n--- Running full validation ---")
    metrics = model.val(
        data=DATA_YAML,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        plots=True,
        save_json=True,
        project=EVAL_OUTPUT,
        name="val_run",
        exist_ok=True,
    )
    return metrics


def print_per_class_metrics(metrics):
    """Print per-class AP and flag poor performers."""
    print("\n--- Per-class mAP50 ---")
    print(f"{'Class':<20} {'mAP50':>8} {'Status'}")
    print("-" * 40)

    class_map50 = metrics.box.maps  # Per-class mAP50
    poor_classes = []

    for i, cls_name in enumerate(FINAL_CLASSES):
        if i < len(class_map50):
            ap = class_map50[i]
            status = "[OK]" if ap >= MAP_WARN_THRESH else "[POOR - needs more data]"
            if ap < MAP_WARN_THRESH:
                poor_classes.append((cls_name, ap))
            print(f"  {cls_name:<20} {ap:>8.3f}  {status}")

    print(f"\n  Overall mAP50    : {metrics.box.map50:.3f}")
    print(f"  Overall mAP50-95 : {metrics.box.map:.3f}")
    print(f"  Precision        : {metrics.box.mp:.3f}")
    print(f"  Recall           : {metrics.box.mr:.3f}")

    if poor_classes:
        print("\n  [ACTION NEEDED] Classes with low mAP50:")
        for cls, ap in poor_classes:
            print(f"    - {cls}: {ap:.3f}  -> Add more training data or fix labels")

    return poor_classes


def save_prediction_previews(model):
    """Run inference on random val images and save predictions."""
    os.makedirs(EVAL_OUTPUT + "/previews", exist_ok=True)
    img_extensions = ["*.jpg", "*.jpeg", "*.png"]
    all_imgs = []
    for ext in img_extensions:
        all_imgs.extend(glob.glob(os.path.join(VAL_IMG_DIR, ext)))

    sample = random.sample(all_imgs, min(N_PREVIEW, len(all_imgs)))
    print(f"\n--- Saving {len(sample)} prediction previews ---")

    results = model.predict(
        source=sample,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=True,
        save_conf=True,
        project=EVAL_OUTPUT + "/previews",
        name="predictions",
        exist_ok=True,
    )
    print(f"  Saved to: {EVAL_OUTPUT}/previews/predictions/")


def evaluate():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Weights not found: {WEIGHTS_PATH}")
        print("  Run step7_train.py first.")
        return

    print(f"Loading model: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    metrics = run_validation(model)
    poor_classes = print_per_class_metrics(metrics)
    save_prediction_previews(model)

    print(f"\n  Full evaluation results saved to: {EVAL_OUTPUT}")
    return poor_classes


if __name__ == "__main__":
    evaluate()
    print("\n[DONE] Step 8 complete.")
