"""
STEP 10 (OPTIMIZED): Retrain with improved dataset — up to 10x faster

Speed improvements applied:
  1. Mixed precision (FP16) — 2-3x speedup on any modern GPU
  2. Larger batch size via rectangle training — better GPU utilization
  3. rect=True — pads images to similar aspect ratios per batch, less wasted compute
  4. cache='ram' (or 'disk') — eliminates repeated image decoding from disk
  5. workers tuned to CPU count — saturates data pipeline without thrashing
  6. close_mosaic earlier — mosaic augmentation is expensive; disable it sooner
  7. Reduced epochs + aggressive early stopping — stop as soon as plateau hits
  8. nbs (nominal batch size) tuned — stabilizes gradient scaling at small batches
  9. Disabled redundant plots during training — saves I/O per epoch
  10. amp=True explicitly — ensures automatic mixed precision is always on
"""

from ultralytics import YOLO
import torch
import os
import glob
import multiprocessing

DATA_YAML   = "data/marine_detection.yaml"
PROJECT_DIR = "runs/marine"
BASE_MODEL  = "yolov8s.pt"
RESUME_FROM_LAST = True

# ─── Hardware Detection ───────────────────────────────────────────────────────
HAS_GPU   = torch.cuda.is_available()
DEVICE    = 0 if HAS_GPU else "cpu"
CPU_COUNT = multiprocessing.cpu_count()

# Tune batch for your VRAM:
#   4GB  VRAM → batch=16
#   6GB  VRAM → batch=32
#   8GB  VRAM → batch=48
#   12GB VRAM → batch=64
#   24GB VRAM → batch=128 or -1 (auto)
# -1 = Ultralytics auto-detects max safe batch
BATCH = -1 if HAS_GPU else 4

# Cache strategy:
#   'ram'  — fastest, needs ~4-8GB free RAM for typical marine dataset
#   'disk' — slower than RAM but still faster than no cache; use if RAM is tight
#   False  — no cache, reads from disk every epoch (slowest)
CACHE = "ram"

# ─── Optimized Train Config ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    "data":    DATA_YAML,
    "project": PROJECT_DIR,

    # ── Precision & throughput ──────────────────────────────────────────────
    "amp":    True,          # FP16 mixed precision — 2-3x speedup on GPU
    "device": DEVICE,
    "batch":  BATCH,         # -1 = auto; or set fixed value from table above
    "workers": min(CPU_COUNT, 8),  # Cap at 8; beyond this hurts more than helps

    # ── Data loading — eliminates per-epoch disk I/O ─────────────────────
    "cache":  CACHE,         # Pre-load all images into RAM at epoch 0
    "rect":   True,          # Rectangle training: batch similar aspect ratios
                             # → smaller padding → less wasted compute per batch

    # ── Training schedule ───────────────────────────────────────────────────
    "epochs":        50,     # Enough for fine-tuning; early stopping handles the rest
    "patience":      10,     # Stop after 10 epochs of no improvement (was 20)
    "imgsz":         640,
    "optimizer":     "AdamW",
    "lr0":           0.001,  # Slightly higher than before — converges faster
    "lrf":           0.01,
    "warmup_epochs": 1,      # 1 warmup epoch is plenty for retraining
    "cos_lr":        True,

    # ── Augmentation cost reduction ─────────────────────────────────────────
    "close_mosaic": 5,       # Disable mosaic in last 5 epochs (was default 10)
                             # Mosaic is expensive; disabling it earlier speeds
                             # up the final convergence phase
    "mosaic":    1.0,
    "mixup":     0.05,       # Reduced from 0.1 — saves compute, marginal gain
    "copy_paste": 0.05,      # Reduced from 0.1
    "hsv_h":     0.015,
    "hsv_s":     0.7,
    "hsv_v":     0.4,
    "degrees":   10.0,
    "translate": 0.1,
    "scale":     0.5,
    "flipud":    0.1,
    "fliplr":    0.5,
    "augment":   True,

    # ── I/O and logging — skip expensive per-epoch writes ──────────────────
    "plots":       False,    # Generate plots only at end, not every epoch
    "save":        True,
    "save_period": -1,       # Only save best.pt + last.pt; skip intermediate
                             # checkpoints (was saving every 10 epochs = disk I/O)
    "val":         True,
    "verbose":     False,    # Reduce per-batch stdout — tiny but real saving
                             # at high batch counts
    # nbs: nominal batch size for gradient scaling stability
    "nbs": 64,               # Keeps effective LR consistent regardless of actual batch
}
# ─────────────────────────────────────────────────────────────────────────────


def get_next_version():
    existing = glob.glob(os.path.join(PROJECT_DIR, "v*"))
    versions = []
    for e in existing:
        name = os.path.basename(e)
        if name.startswith("v") and name[1:].isdigit():
            versions.append(int(name[1:]))
    return max(versions, default=0) + 1


def find_best_weights():
    existing = glob.glob(os.path.join(PROJECT_DIR, "v*", "weights", "best.pt"))
    if not existing:
        return None
    def version_num(p):
        for part in p.split(os.sep):
            if part.startswith("v") and part[1:].isdigit():
                return int(part[1:])
        return 0
    existing.sort(key=version_num, reverse=True)
    return existing[0]


def print_speed_profile():
    print("\n--- Speed Profile ---")
    print(f"  GPU available  : {HAS_GPU}")
    if HAS_GPU:
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM           : {vram:.1f} GB")
    print(f"  CPU cores      : {CPU_COUNT}")
    print(f"  Batch size     : {'auto' if BATCH == -1 else BATCH}")
    print(f"  Mixed precision: {'yes (FP16)' if HAS_GPU else 'no (CPU)'}")
    print(f"  Image cache    : {CACHE}")
    print(f"  Workers        : {TRAIN_CONFIG['workers']}")
    print(f"  Rectangle train: {TRAIN_CONFIG['rect']}")
    print()


def retrain():
    print_speed_profile()

    version  = get_next_version()
    run_name = f"v{version}"
    print(f"Run name: {run_name}")

    if RESUME_FROM_LAST:
        best_pt = find_best_weights()
        if best_pt:
            print(f"Loading weights from: {best_pt}")
            model = YOLO(best_pt)
        else:
            print(f"No previous weights found — starting from {BASE_MODEL}")
            model = YOLO(BASE_MODEL)
    else:
        print(f"Starting fresh from {BASE_MODEL}")
        model = YOLO(BASE_MODEL)

    # torch.compile (Triton) is not officially supported natively on Windows PyTorch yet.
    # Leaving compile=False safely bypasses the Induction backend crashes.
    TRAIN_CONFIG["compile"] = False

    TRAIN_CONFIG["name"]     = run_name
    TRAIN_CONFIG["exist_ok"] = False

    results  = model.train(**TRAIN_CONFIG)
    best_path = os.path.join(PROJECT_DIR, run_name, "weights", "best.pt")
    print(f"\nBest weights: {best_path}")
    return results


if __name__ == "__main__":
    retrain()
    print("\n[DONE] Step 10 complete.")