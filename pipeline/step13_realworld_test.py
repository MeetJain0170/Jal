"""
STEP 13: Real-world testing on unseen underwater images/videos
- Test on image folder, single image, or video file
- Logs failure cases (low confidence, unexpected classes)
- Saves annotated outputs for review
- Generates a simple test report
"""

import os
import cv2
import glob
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add parent dir to path so we can import detector
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from step12_yolo_detector import MarineYOLODetector, MARINE_CLASSES

WEIGHTS_PATH   = "models/detection/marine_detector.pt"
OUTPUT_DIR     = "runs/realworld_test"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.45
FAILURE_CONF   = 0.40   # Detections below this are flagged as uncertain


def test_images(detector, image_dir, output_dir):
    """Run detection on all images in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not img_paths:
        print(f"No images found in: {image_dir}")
        return

    print(f"\nTesting {len(img_paths)} images...")
    stats = defaultdict(int)
    failure_log = []

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue

        t0 = time.time()
        detections = detector.detect(frame)
        elapsed_ms = (time.time() - t0) * 1000

        stats["images"] += 1
        stats["detections"] += len(detections)

        # Flag failures
        for det in detections:
            stats[f"cls_{det.class_name}"] += 1
            if det.confidence < FAILURE_CONF:
                failure_log.append({
                    "image": os.path.basename(img_path),
                    "class": det.class_name,
                    "conf":  det.confidence,
                    "bbox":  det.bbox,
                })

        # Save annotated image
        annotated = detector.draw_detections(frame, detections)
        summary = detector.summary(detections)

        # Overlay summary text
        info = f"Total:{summary['total']} Trash:{summary['trash']} Animals:{summary['animals']} {elapsed_ms:.0f}ms"
        cv2.putText(annotated, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, annotated)

    print(f"\n--- Test Summary ---")
    print(f"  Images tested : {stats['images']}")
    print(f"  Total detections: {stats['detections']}")
    print(f"  Avg detections/image: {stats['detections']/max(1,stats['images']):.1f}")
    print(f"\n  Per-class counts:")
    for cls in MARINE_CLASSES:
        count = stats.get(f"cls_{cls}", 0)
        if count > 0:
            print(f"    {cls:<20}: {count}")
    print(f"\n  Low-confidence detections (< {FAILURE_CONF}): {len(failure_log)}")
    if failure_log:
        print("  First 10 failures:")
        for f in failure_log[:10]:
            print(f"    {f['image']} | {f['class']} | conf={f['conf']:.2f}")

    # Write failure log
    log_path = os.path.join(output_dir, "failure_log.txt")
    with open(log_path, "w") as f:
        for item in failure_log:
            f.write(f"{item['image']} | {item['class']} | conf={item['conf']:.3f} | bbox={item['bbox']}\n")
    print(f"\n  Results saved to: {output_dir}")
    print(f"  Failure log: {log_path}")


def test_video(detector, video_path, output_dir):
    """Run detection on a video file."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(output_dir, Path(video_path).stem + "_detected.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"\nProcessing video: {video_path}")
    print(f"  Frames: {total} | FPS: {fps} | Size: {width}x{height}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated  = detector.draw_detections(frame, detections)
        summary    = detector.summary(detections)

        info = f"Frame {frame_idx} | {summary['total']} objects | Trash:{summary['trash']} Animals:{summary['animals']}"
        cv2.putText(annotated, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total} frames")

    cap.release()
    writer.release()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JalDrishti Marine Detector - Real World Test")
    parser.add_argument("--source", required=True, help="Path to image folder or video file")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Path to .pt weights")
    parser.add_argument("--conf",    default=CONF_THRESHOLD, type=float)
    parser.add_argument("--output",  default=OUTPUT_DIR)
    args = parser.parse_args()

    detector = MarineYOLODetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=IOU_THRESHOLD,
    )

    if os.path.isdir(args.source):
        test_images(detector, args.source, args.output)
    elif os.path.isfile(args.source):
        ext = Path(args.source).suffix.lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            test_video(detector, args.source, args.output)
        else:
            # Single image
            test_images(detector, os.path.dirname(args.source), args.output)
    else:
        print(f"[ERROR] Source not found: {args.source}")

    print("\n[DONE] Step 13 complete.")
