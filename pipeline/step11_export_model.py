"""
STEP 11: Export best model to multiple formats
- Copies best.pt to models/ directory in JalDrishti
- Optionally exports to ONNX for deployment (recommended for production)
- Optionally exports to TensorRT for GPU inference speedup
"""

import os
import shutil
import glob
from ultralytics import YOLO

PROJECT_DIR  = "runs/marine"
MODELS_DIR   = "models/detection"
EXPORT_ONNX  = True     # Recommended for deployment / cross-platform
EXPORT_TRT   = False    # Requires TensorRT install; fastest on NVIDIA GPUs


def find_best_weights():
    """Find the best.pt from the highest-version run."""
    candidates = glob.glob(os.path.join(PROJECT_DIR, "v*", "weights", "best.pt"))
    if not candidates:
        return None
    def version_num(p):
        for part in p.split(os.sep):
            if part.startswith("v") and part[1:].isdigit():
                return int(part[1:])
        return 0
    candidates.sort(key=version_num, reverse=True)
    return candidates[0]


def export_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_pt = find_best_weights()
    if not best_pt:
        print("[ERROR] No best.pt found. Run step7 or step10 first.")
        return

    print(f"Best weights found: {best_pt}")

    # ── Copy best.pt ──────────────────────────────────────────────────────
    dst_pt = os.path.join(MODELS_DIR, "marine_detector.pt")
    shutil.copy2(best_pt, dst_pt)
    print(f"Copied: {dst_pt}")

    model = YOLO(best_pt)

    # ── Export ONNX ───────────────────────────────────────────────────────
    if EXPORT_ONNX:
        print("\nExporting to ONNX...")
        onnx_path = model.export(
            format="onnx",
            imgsz=640,
            simplify=True,   # Simplify graph for faster inference
            dynamic=False,   # Fixed batch size (1) for deployment
        )
        dst_onnx = os.path.join(MODELS_DIR, "marine_detector.onnx")
        shutil.copy2(onnx_path, dst_onnx)
        print(f"ONNX exported: {dst_onnx}")

    # ── Export TensorRT ───────────────────────────────────────────────────
    if EXPORT_TRT:
        print("\nExporting to TensorRT (this can take 10+ minutes)...")
        try:
            trt_path = model.export(
                format="engine",
                imgsz=640,
                half=True,   # FP16 for speed
                simplify=True,
            )
            dst_trt = os.path.join(MODELS_DIR, "marine_detector.engine")
            shutil.copy2(trt_path, dst_trt)
            print(f"TensorRT exported: {dst_trt}")
        except Exception as e:
            print(f"[WARNING] TensorRT export failed: {e}")
            print("  Install tensorrt and try again, or use ONNX instead.")

    print(f"\nAll exports saved to: {MODELS_DIR}/")


if __name__ == "__main__":
    export_model()
    print("\n[DONE] Step 11 complete.")
