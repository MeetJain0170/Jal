from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_YOLO = None
_BACKEND = None


def _load_backend(weights: str):
    global _YOLO, _BACKEND
    if _YOLO is not None:
        return _YOLO
    wants_world = "world" in os.path.basename(weights).lower()
    if wants_world:
        try:
            from ultralytics import YOLOWorld
            _YOLO = YOLOWorld
            _BACKEND = "world"
            logger.info("Using YOLOWorld backend.")
        except Exception:
            from ultralytics import YOLO
            _YOLO = YOLO
            _BACKEND = "yolo"
            logger.info("YOLOWorld unavailable; using standard YOLO backend.")
    else:
        from ultralytics import YOLO
        _YOLO = YOLO
        _BACKEND = "yolo"
        logger.info("Using standard YOLO backend.")
    return _YOLO


CLASSES = [
    "diver", "person underwater", "person swimming",
    "fish", "shark", "whale", "dolphin", "sea turtle", "ray", "jellyfish", "octopus",
    "submarine", "torpedo", "naval mine", "weapon",
    "boat", "ship", "vessel", "debris", "equipment",
]

REMAP = {
    "diver": "Diver",
    "person underwater": "Diver",
    "person swimming": "Diver",
    "person": "Diver",
    "fish": "Fish",
    "shark": "Shark",
    "whale": "Whale",
    "dolphin": "Dolphin",
    "sea turtle": "Sea Turtle",
    "turtle": "Sea Turtle",
    "ray": "Ray",
    "manta ray": "Ray",
    "stingray": "Ray",
    "jellyfish": "Jellyfish",
    "octopus": "Octopus",
    "submarine": "Submarine",
    "torpedo": "Torpedo",
    "naval mine": "Naval Mine",
    "mine": "Naval Mine",
    "weapon": "Weapon",
    "boat": "Surface Vessel",
    "ship": "Surface Vessel",
    "vessel": "Surface Vessel",
    "debris": "Debris",
    "equipment": "Equipment",
}

SECURITY = {"Submarine", "Torpedo", "Naval Mine", "Weapon"}
MARINE = {"Fish", "Shark", "Whale", "Dolphin", "Sea Turtle", "Ray", "Jellyfish", "Octopus"}
DIVERS = {"Diver"}
VESSELS = {"Surface Vessel"}


def _map_label(raw: str) -> str:
    r = raw.lower().strip()
    if r in REMAP:
        return REMAP[r]
    for k, v in REMAP.items():
        if k in r:
            return v
    # Handle custom-trained class names like `trash_rope` → `Rope`
    if r.startswith("trash_"):
        r = r[len("trash_"):]
    # Make underscores human-friendly: `animal_other` → `Animal Other`
    r = r.replace("_", " ").strip()
    return r.title()


def _category(display: str) -> str:
    if display in SECURITY:
        return "Security Threat"
    if display in DIVERS:
        return "Diver"
    if display in MARINE:
        return "Marine Life"
    if display in VESSELS:
        return "Surface Vessel"
    return "Object"


def _preprocess(img_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return out


def _nms(dets: List[Dict[str, Any]], iou: float) -> List[Dict[str, Any]]:
    if not dets:
        return []
    # Class-aware NMS to avoid collapsing dense fish schools into 1-2 boxes.
    out: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        key = str(d.get("display_class") or d.get("category") or "Object")
        groups.setdefault(key, []).append(d)

    for _, group in groups.items():
        boxes = []
        scores = []
        for d in group:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append([int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))])
            scores.append(float(d["confidence"]))
        idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou)
        if len(idxs) == 0:
            out.extend(group)
            continue
        keep = {int(i if np.isscalar(i) else i[0]) for i in idxs}
        out.extend([d for i, d in enumerate(group) if i in keep])

    out.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return out


def _pass_gate(d: Dict[str, Any], w: int, h: int) -> bool:
    conf = float(d["confidence"])
    x1, y1, x2, y2 = d["bbox"]
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(float(w * h), 1.0)
    cat = d["category"]
    if cat == "Marine Life":
        # More permissive for dense schools/small fauna in wide scenes.
        return conf >= 0.03 and area_ratio >= 0.0002
    if cat == "Diver":
        return conf >= 0.45 and area_ratio >= 0.003
    if cat == "Security Threat":
        return conf >= 0.20 and area_ratio >= 0.001
    return conf >= 0.15 and area_ratio >= 0.0008


class MaritimeDetector:
    def __init__(
        self,
        weights: str,
        conf_thresh: float = 0.01,
        iou_thresh: float = 0.01,
        img_size: int = 1280,
    ):
        self.weights = weights
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.model = None

    def _load(self):
        if self.model is not None:
            return self.model
        Y = _load_backend(self.weights)
        self.model = Y(self.weights)
        if _BACKEND == "world" and hasattr(self.model, "set_classes"):
            self.model.set_classes(CLASSES)
        return self.model

    def detect(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        h, w = img_np.shape[:2]
        infer = _preprocess(img_np)
        model = self._load()
        res = model(
            infer,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            verbose=False,
            augment=False,
        )

        dets: List[Dict[str, Any]] = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0])
                raw = str(model.names[cls_id])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                disp = _map_label(raw)
                cat = _category(disp)
                dets.append({
                    "class": raw,
                    "display_class": disp,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "category": cat,
                    "threat_score": round(conf if cat == "Security Threat" else conf * 0.2, 3),
                    "threat_level": "HIGH" if cat == "Security Threat" and conf > 0.6 else "LOW",
                })

        dets = _nms(dets, self.iou_thresh)
        raw_ranked = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        dets = [d for d in dets if _pass_gate(d, w, h)]
        # If all gates remove detections, keep multiple top marine candidates
        # to avoid collapsing busy fish scenes to 1-2 boxes.
        if not dets and raw_ranked:
            marineish = []
            for d in raw_ranked:
                lbl = (d.get("display_class") or "").lower()
                if d.get("category") == "Marine Life" or lbl in {"fish", "eel", "ray", "shark", "marine animal"}:
                    if float(d.get("confidence", 0.0)) >= 0.03:
                        marineish.append(d)
                if len(marineish) >= 8:
                    break
            if marineish:
                dets = marineish
            elif raw_ranked[0]["confidence"] >= 0.05:
                dets = [raw_ranked[0]]
        dets.sort(key=lambda x: x["confidence"], reverse=True)
        return dets

    def detect_and_annotate(
        self,
        inference_img: np.ndarray,
        original_img: Optional[np.ndarray] = None,
        context: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        dets = self.detect(inference_img)
        canvas = original_img if original_img is not None else inference_img
        ann = _draw(canvas, dets)
        pil = Image.fromarray(ann)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return dets, f"data:image/png;base64,{b64}"


def _draw(img_np: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    im = Image.fromarray(img_np)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    colors = {
        "Marine Life": (52, 211, 153),
        "Diver": (251, 191, 36),
        "Security Threat": (239, 68, 68),
        "Surface Vessel": (96, 165, 250),
        "Object": (156, 163, 175),
    }
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        c = colors.get(d["category"], (156, 163, 175))
        dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
        label = f"{d['display_class']} {int(d['confidence']*100)}%"
        dr.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 7 + 6, y1], fill=c)
        dr.text((x1 + 3, max(0, y1 - 14)), label, fill=(0, 0, 0), font=font)
    return np.array(im)


_detector: Optional[MaritimeDetector] = None
_sig: Optional[Tuple[Any, ...]] = None


def get_detector(
    weights: Optional[str] = None,
    conf_thresh: float = 0.08,
    iou_thresh: float = 0.45,
    img_size: int = 1280,
    enhance: bool = True,
    tta: bool = False,
    multi_scale: bool = False,
) -> MaritimeDetector:
    global _detector, _sig

    if weights is None:
        weights = r"C:\Users\meetj\Documents\Career\Projects\JalDrishti\detect\runs\marine\v2-5\weights\best.pt"

    sig = (weights, conf_thresh, iou_thresh, img_size)

    if _detector is None or _sig != sig:
        _detector = MaritimeDetector(weights, conf_thresh, iou_thresh, img_size)
        _sig = sig

    return _detector
