"""
STEP 7: Train YOLOv8 on the merged marine dataset

Base model options (in JalDrishti/):
  yolov8n.pt  — nano,   fastest, least accurate  (good for testing)
  yolov8s.pt  — small,  balanced
  yolov8l.pt  — large,  better accuracy (you have this as yolov8l-worldv2.pt)
  yolov8x.pt  — xlarge, best accuracy, slowest

Recommendation:
  - Start with yolov8s.pt for quick iteration
  - Move to yolov8l or yolov8x for final training
"""

from ultralytics import YOLO, settings
import torch
import os


# Disable TQDM iteration bars completely to fix console log spanning
os.environ['TQDM_DISABLE'] = '1'

# Override global ultralytics settings so output doesn't bleed into Neurosearch or other projects
settings.update({'runs_dir': os.path.abspath('.')})

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_YAML   = "data/marine_detection.yaml"
BASE_MODEL  = "yolov8s.pt"        # Change to yolov8l.pt for better results
PROJECT_DIR = "runs/marine"
RUN_NAME    = "v1"
FORCE_CUDA  = True               # If True, error out instead of silently using CPU

TRAIN_CONFIG = {
    "data":      DATA_YAML,
    "epochs":    100,
    "patience":  20,             # Early stopping: stop if no improvement for 20 epochs
    "imgsz":     640,
    "batch":     8,              # Reduce to 8 if GPU OOM
    "workers":   8,              # Increased to speed up dataloading
    "cache":     "ram",           # 🔥 Cache images in RAM to eliminate disk I/O bottleneck
    "amp":       True,           # Automatic Mixed Precision for faster tensor cores
    # Ultralytics accepts int GPU index (0) or string like "0"/"0,1".
    # We default to GPU 0 when CUDA is available.
    "device":    0 if torch.cuda.is_available() else "cpu",
    "project":   PROJECT_DIR,
    "name":      RUN_NAME,
    "exist_ok":  True,
    "optimizer": "AdamW",
    "lr0":       0.001,
    "lrf":       0.01,           # Final LR = lr0 * lrf
    "warmup_epochs": 3,
    "cos_lr":    True,
    "augment":   True,
    # Augmentation suitable for underwater imagery
    "hsv_h":     0.015,
    "hsv_s":     0.7,
    "hsv_v":     0.4,
    "degrees":   10.0,
    "translate": 0.1,
    "scale":     0.5,
    "flipud":    0.1,
    "fliplr":    0.5,
    "mosaic":    1.0,
    "mixup":     0.1,
    "copy_paste": 0.1,
    "save":      True,
    "save_period": 10,           # Save checkpoint every 10 epochs
    "val":       True,
    "plots":     True,
    "verbose":   False,
}
# ────────────────────────────────────────────────────────────────────────────


def train():
    cuda_available = torch.cuda.is_available()
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    if FORCE_CUDA and not cuda_available:
        raise RuntimeError(
            "CUDA was requested (FORCE_CUDA=True) but is not available. "
            "This usually means you installed CPU-only PyTorch or your CUDA drivers/toolkit "
            "aren't compatible. Install a CUDA-enabled PyTorch build and ensure `nvidia-smi` works."
        )

    if cuda_available:
        try:
            torch.cuda.set_device(0)
        except Exception as e:
            raise RuntimeError(f"Failed to select CUDA device 0: {e}") from e

        print("Device: GPU (CUDA)")
        print(f"GPU[0]: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("Device: CPU")

    # Look for last.pt to gracefully resume after OOMs or crashes
    custom_run_path = r"C:\Users\meetj\Documents\Career\Projects\JalDrishti\runs\marine\v1\weights\last.pt"
    local_run_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "last.pt")

    last_pt = custom_run_path if os.path.exists(custom_run_path) else local_run_path

    if os.path.exists(last_pt):
        print(f"\nFound existing checkpoint: {last_pt}")
        print("Resuming training from the last saved epoch. Memory (batch size) adjustments have been applied.")
        model = YOLO(last_pt)
        results = model.train(resume=True)
    else:
        print(f"\nLoading base model: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)

        print(f"\nStarting fresh training — output: {PROJECT_DIR}/{RUN_NAME}")
        TRAIN_CONFIG["batch"] = 8  # Safe default to prevent future OOMs
        results = model.train(**TRAIN_CONFIG)

    best_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\nTraining complete.")
    print(f"Best weights: {best_path}")
    return results


if __name__ == "__main__":
    train()
    print("\n[DONE] Step 7 complete.")
