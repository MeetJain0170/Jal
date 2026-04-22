"""
STEP 12: Integrate trained model into JalDrishti's detection/yolo_detector.py

This file is the REPLACEMENT for detection/yolo_detector.py
Drop this into your JalDrishti/detection/ folder.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─── Class list (must match data/marine_detection.yaml) ──────────────────────
MARINE_CLASSES = [
    "fish",           # 0
    "trash_bag",      # 1
    "trash_bottle",   # 2
    "trash_can",      # 3
    "trash_cup",      # 4
    "trash_net",      # 5
    "trash_rope",     # 6
    "trash_pipe",     # 7
    "trash_wreckage", # 8
    "trash_other",    # 9
    "crab",           # 10
    "starfish",       # 11
    "eel",            # 12
    "echinus",        # 13
    "holothurian",    # 14
    "scallop",        # 15
    "animal_other",   # 16
]

TRASH_CLASSES  = {1, 2, 3, 4, 5, 6, 7, 8, 9}
ANIMAL_CLASSES = {0, 10, 11, 12, 13, 14, 15, 16}

# Class colors for visualization (BGR)
CLASS_COLORS = {
    0:  (0, 200, 100),    # fish        - green
    1:  (0, 0, 200),      # trash_bag   - red
    2:  (0, 0, 180),      # trash_bottle
    3:  (0, 0, 160),      # trash_can
    4:  (0, 0, 140),      # trash_cup
    5:  (0, 0, 220),      # trash_net
    6:  (0, 0, 200),      # trash_rope
    7:  (0, 0, 180),      # trash_pipe
    8:  (0, 0, 160),      # trash_wreckage
    9:  (0, 0, 140),      # trash_other
    10: (100, 200, 0),    # crab        - lime
    11: (200, 100, 0),    # starfish    - orange
    12: (200, 0, 100),    # eel         - pink
    13: (100, 0, 200),    # echinus     - purple
    14: (0, 100, 200),    # holothurian - cyan
    15: (200, 200, 0),    # scallop     - yellow
    16: (150, 150, 150),  # animal_other - gray
}


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]  # x1, y1, x2, y2
    is_trash:   bool = field(init=False)
    is_animal:  bool = field(init=False)

    def __post_init__(self):
        self.is_trash  = self.class_id in TRASH_CLASSES
        self.is_animal = self.class_id in ANIMAL_CLASSES

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class MarineYOLODetector:
    """
    YOLOv8 detector for marine objects (fish, trash, underwater creatures).
    Drop-in replacement for detection/yolo_detector.py in JalDrishti.
    """

    # Project root = parent of either `pipeline/` or `detection/` folder (depending on where
    # this file lives after you "drop it in").
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Prefer the latest trained model from Step 7 if it exists.
    DEFAULT_TRAINED_WEIGHTS = os.path.join(
        _ROOT, "runs", "marine", "v1", "weights", "best.pt"
    )

    # Fallback path for a packaged/exported model.
    DEFAULT_PACKAGED_WEIGHTS = os.path.join(
        _ROOT, "models", "detection", "marine_detector.pt"
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold:  float = 0.45,
        device: Optional[str] = None,
    ):
        if weights_path is not None:
            self.weights_path = weights_path
        else:
            self.weights_path = (
                self.DEFAULT_TRAINED_WEIGHTS
                if os.path.exists(self.DEFAULT_TRAINED_WEIGHTS)
                else self.DEFAULT_PACKAGED_WEIGHTS
            )
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device = device or ("cuda:0" if self._has_cuda() else "cpu")

        print(f"[MarineYOLODetector] Loading: {self.weights_path}")
        self.model = YOLO(self.weights_path)
        self.model.to(self.device)
        print(f"[MarineYOLODetector] Ready on device: {self.device}")

    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame (OpenCV format).
        Returns list of Detection objects.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=MARINE_CLASSES[cls_id] if cls_id < len(MARINE_CLASSES) else "unknown",
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames."""
        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        batch_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=MARINE_CLASSES[cls_id] if cls_id < len(MARINE_CLASSES) else "unknown",
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
            batch_detections.append(detections)
        return batch_detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame. Returns annotated copy."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_id, (200, 200, 200))
            label = f"{det.class_name} {det.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated

    def filter_trash(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_trash]

    def filter_animals(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_animal]

    def summary(self, detections: List[Detection]) -> dict:
        """Return count summary by class."""
        from collections import Counter
        counts = Counter(d.class_name for d in detections)
        return {
            "total":   len(detections),
            "trash":   len(self.filter_trash(detections)),
            "animals": len(self.filter_animals(detections)),
            "by_class": dict(counts),
        }


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = MarineYOLODetector(conf_threshold=0.3)
    # Test on a blank frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_frame)
    print(f"Test detection on blank frame: {len(dets)} detections (expected 0)")
    print("[DONE] Step 12 integration ready.")
