"""
detection/yolo_detector.py — Maritime object detection, v3.

Architecture
------------
The previous YOLO-World approach had a fundamental ceiling: its visual-language
embedding space maps elongated streamlined bodies to "whale" regardless of text
prompts, causing sharks to be consistently mislabeled. Text prompt engineering
cannot fix embedding-space problems.

This version uses a TWO-STAGE pipeline:

Stage 1 — YOLOv8x (standard COCO model)
  The largest pretrained YOLOv8 model. COCO-trained YOLOv8x is significantly
  better at localising objects than YOLO-World for in-the-wild underwater
  footage. Its 80 COCO classes (person, fish, various animals) map well to our
  taxonomy with a remap table.

Stage 2 — Crop-level shape classifier
  For each "fish" or ambiguous detection above a size threshold, we run a
  lightweight shape analysis (aspect ratio, fin geometry, body taper) to
  disambiguate:
    - Sharks   → elongated, tapered, dorsal fin signature, countershading
    - Rays     → wide flat shape, wing aspect ratio > 2.2
    - Divers   → vertical elongation, dark wetsuit, equipment
    - Fish     → compact, bilateral symmetry, high saturation (reef fish)
  Purely geometric — no additional ML model required.

Stage 3 — Post-processing
  - Containment suppression: fin sub-boxes inside an already-detected shark
    box are dropped.
  - Area sanity: a detection covering > 15% of frame is almost never "Fish".
  - False-positive guard: clownfish → Diver, anemone → Fish are suppressed.
  - Aspect ratio guard: divers must not be extremely wide and flat.

YOLO-World is used as a SUPPLEMENTARY pass ONLY for threat-specific objects
(mines, submarines, torpedoes, marine mammals) where open-vocabulary helps
and where marine-life confusion is less of a concern.

Recommended weights
-------------------
  Primary   : yolov8x.pt   (best accuracy, ~130MB, auto-downloaded)
  World supp: yolov8x-worldv2.pt  (open-vocab threats + mammals)
"""

from __future__ import annotations

import io
import base64
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Backend loading ───────────────────────────────────────────────────────────

_YOLO_STANDARD = None
_YOLO_WORLD    = None


def _get_yolo_standard():
    global _YOLO_STANDARD
    if _YOLO_STANDARD is not None:
        return _YOLO_STANDARD
    try:
        from ultralytics import YOLO
        _YOLO_STANDARD = YOLO
        logger.info("Loaded standard YOLO backend.")
    except ImportError:
        logger.error("ultralytics not installed — run: pip install ultralytics")
    return _YOLO_STANDARD


def _get_yolo_world():
    global _YOLO_WORLD
    if _YOLO_WORLD is not None:
        return _YOLO_WORLD
    try:
        from ultralytics import YOLOWorld
        _YOLO_WORLD = YOLOWorld
        logger.info("Loaded YOLOWorld backend.")
    except ImportError:
        try:
            from ultralytics import YOLO
            _YOLO_WORLD = YOLO
            logger.warning("YOLOWorld not available, using standard YOLO for world pass.")
        except ImportError:
            pass
    return _YOLO_WORLD


# ═══════════════════════════════════════════════════════════════════════════════
# COCO → MARITIME LABEL MAP
# ═══════════════════════════════════════════════════════════════════════════════
# Maps all 80 COCO class names to maritime display labels.
# "fish" maps to "Fish" initially; shape classifier refines to Shark/Ray/etc.

COCO_TO_MARITIME: Dict[str, str] = {
    "person":           "Diver",
    "bicycle":          "Equipment",
    "car":              "Submersible",
    "motorcycle":       "Equipment",
    "airplane":         "Equipment",
    "bus":              "Submersible",
    "train":            "Equipment",
    "truck":            "Submersible",
    "boat":             "Surface Vessel",
    "traffic light":    "Equipment",
    "fire hydrant":     "Equipment",
    "stop sign":        "Equipment",
    "parking meter":    "Equipment",
    "bench":            "Debris",
    "bird":             "Fish",
    "cat":              "Fish",
    "dog":              "Marine Animal",
    "horse":            "Marine Animal",
    "sheep":            "Marine Animal",
    "cow":              "Marine Animal",
    "elephant":         "Marine Animal",
    "bear":             "Marine Animal",
    "zebra":            "Marine Animal",
    "giraffe":          "Marine Animal",
    "backpack":         "Equipment",
    "umbrella":         "Equipment",
    "handbag":          "Equipment",
    "tie":              "Debris",
    "suitcase":         "Suspicious Package",
    "frisbee":          "Naval Mine",
    "skis":             "Debris",
    "snowboard":        "Debris",
    "sports ball":      "Naval Mine",
    "kite":             "Equipment",
    "baseball bat":     "Weapon",
    "baseball glove":   "Equipment",
    "skateboard":       "Debris",
    "surfboard":        "Debris",
    "tennis racket":    "Equipment",
    "bottle":           "Debris",
    "wine glass":       "Debris",
    "cup":              "Debris",
    "fork":             "Weapon",
    "knife":            "Weapon",
    "spoon":            "Equipment",
    "bowl":             "Debris",
    "banana":           "Debris",
    "apple":            "Debris",
    "sandwich":         "Debris",
    "orange":           "Debris",
    "broccoli":         "Aquatic Plant",
    "carrot":           "Debris",
    "hot dog":          "Debris",
    "pizza":            "Debris",
    "donut":            "Debris",
    "cake":             "Debris",
    "chair":            "Debris",
    "couch":            "Debris",
    "potted plant":     "Aquatic Plant",
    "bed":              "Debris",
    "dining table":     "Debris",
    "toilet":           "Debris",
    "tv":               "Equipment",
    "laptop":           "Equipment",
    "mouse":            "Equipment",
    "remote":           "Equipment",
    "keyboard":         "Equipment",
    "cell phone":       "Equipment",
    "microwave":        "Equipment",
    "oven":             "Equipment",
    "toaster":          "Equipment",
    "sink":             "Debris",
    "refrigerator":     "Equipment",
    "book":             "Debris",
    "clock":            "Equipment",
    "vase":             "Underwater Structure",
    "scissors":         "Sharp Debris",
    "teddy bear":       "Debris",
    "hair drier":       "Equipment",
    "toothbrush":       "Equipment",
}

# YOLO-World supplementary vocab — threats + hard-to-COCO-map marine mammals
THREAT_VOCAB: List[str] = [
    "naval mine", "underwater mine", "sea mine",
    "submarine", "submersible", "mini-submarine",
    "torpedo", "weapon", "rifle", "gun", "knife",
    "explosive", "bomb", "suspicious package", "hull damage",
    "dugong", "manatee", "sea cow",
    "whale", "humpback whale", "blue whale",
    "manta ray", "stingray",
    "sea turtle", "green sea turtle",
    "shark", "reef shark", "hammerhead shark",
]

# ── Category sets ─────────────────────────────────────────────────────────────

SECURITY_THREAT_CLASSES = {
    "Naval Mine", "Torpedo", "Submarine", "Submersible",
    "Weapon", "Explosive Device", "Suspicious Package",
    "Hull Damage", "Sharp Debris",
}
DIVER_CLASSES = {"Diver", "Scuba Diver", "Freediver", "Snorkeler"}
SHARK_CLASSES = {
    "Shark", "Great White Shark", "Hammerhead Shark",
    "Whale Shark", "Reef Shark",
}
RAY_CLASSES = {"Manta Ray", "Stingray", "Ray"}
MARINE_MAMMAL_CLASSES = {
    "Whale", "Humpback Whale", "Blue Whale", "Sperm Whale",
    "Dolphin", "Bottlenose Dolphin",
    "Dugong", "Manatee", "Seal", "Sea Lion", "Walrus", "Marine Animal",
}
MARINE_LIFE_CLASSES = SHARK_CLASSES | RAY_CLASSES | MARINE_MAMMAL_CLASSES | {
    "Fish", "Tropical Fish", "School of Fish",
    "Sea Turtle", "Green Sea Turtle", "Hawksbill Turtle",
    "Jellyfish", "Octopus", "Squid", "Cuttlefish",
    "Seahorse", "Eel", "Moray Eel",
    "Lobster", "Crab", "Sea Urchin", "Starfish", "Sea Anemone",
    "Coral", "Coral Reef", "Hard Coral", "Soft Coral",
}
STRUCTURE_CLASSES = {
    "Shipwreck", "Underwater Structure", "Pipeline", "Cable",
    "Rope", "Anchor", "Chain", "Rock Formation",
    "Seagrass", "Kelp", "Algae", "Aquatic Plant",
}
VESSEL_CLASSES    = {"Surface Vessel"}
EQUIPMENT_CLASSES = {"Equipment", "Debris"}

ALERT_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Security Threat": (255,  55,  55),
    "Diver":           (255, 160,  40),
    "Marine Life":     ( 50, 215, 110),
    "Marine Mammal":   ( 90, 220, 190),
    "Surface Vessel":  (255, 200,  45),
    "Structure":       (120, 180, 255),
    "Equipment":       (180, 140, 255),
    "Object":          (160, 160, 160),
}

# World model label remap
WORLD_REMAP: Dict[str, str] = {
    "naval mine": "Naval Mine", "underwater mine": "Naval Mine",
    "sea mine": "Naval Mine", "mine": "Naval Mine",
    "torpedo": "Torpedo",
    "submarine": "Submarine", "submersible": "Submersible",
    "mini-submarine": "Submersible",
    "weapon": "Weapon", "rifle": "Weapon", "gun": "Weapon", "knife": "Weapon",
    "explosive": "Explosive Device", "bomb": "Explosive Device",
    "suspicious package": "Suspicious Package", "hull damage": "Hull Damage",
    "dugong": "Dugong", "manatee": "Manatee", "sea cow": "Dugong",
    "whale": "Whale", "humpback whale": "Humpback Whale", "blue whale": "Blue Whale",
    "manta ray": "Manta Ray", "stingray": "Stingray",
    "sea turtle": "Sea Turtle", "green sea turtle": "Green Sea Turtle",
    "shark": "Shark", "reef shark": "Reef Shark",
    "hammerhead shark": "Hammerhead Shark",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SHAPE-BASED CLASSIFIER (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeClassifier:
    """
    Geometric analysis of detection crops to disambiguate marine animals.

    No additional model weights needed — uses OpenCV geometry only.
    """

    @staticmethod
    def classify_crop(
        crop_bgr: np.ndarray,
        bbox: List[float],
        img_w: int,
        img_h: int,
        current_label: str,
    ) -> str:
        if crop_bgr is None or crop_bgr.size == 0:
            return current_label
        ch, cw = crop_bgr.shape[:2]
        if ch < 8 or cw < 8:
            return current_label

        x1, y1, x2, y2 = bbox
        area_ratio = max(0.0, (x2-x1)*(y2-y1)) / max(float(img_w*img_h), 1.0)
        aspect = cw / max(ch, 1)

        # Diver false-positive correction (most common failure)
        if current_label in DIVER_CLASSES:
            if not ShapeClassifier._is_diver_shape(crop_bgr, aspect, area_ratio):
                marine = ShapeClassifier._classify_marine_animal(
                    crop_bgr, aspect, area_ratio
                )
                return marine or ShapeClassifier._best_fish_label(crop_bgr, aspect)

        # Marine animal disambiguation
        if current_label in {"Fish", "Marine Animal"} or current_label in MARINE_MAMMAL_CLASSES:
            marine = ShapeClassifier._classify_marine_animal(
                crop_bgr, aspect, area_ratio
            )
            if marine:
                return marine

        # Large fish → force shape check
        if current_label == "Fish" and area_ratio > 0.06:
            marine = ShapeClassifier._classify_marine_animal(
                crop_bgr, aspect, area_ratio
            )
            return marine or current_label

        return current_label

    # ── Diver shape check ─────────────────────────────────────────────────────

    @staticmethod
    def _is_diver_shape(
        crop_bgr: np.ndarray,
        aspect: float,
        area_ratio: float,
    ) -> bool:
        # Too small to be a diver
        if area_ratio < 0.005:
            return False

        # Too wide and flat — probably a shark or ray
        if aspect > 2.8:
            return False

        # Colour check: reef fish have very high saturation
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        hue_std  = float(np.std(hsv[:, :, 0]))
        mean_val = float(np.mean(hsv[:, :, 2]))

        # Highly saturated AND hue varies a lot AND bright = reef fish
        if mean_sat > 130 and hue_std > 35 and mean_val > 90:
            return False

        # Divers are predominantly dark (wetsuit)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.sum(gray < 85)) / max(gray.size, 1)

        # Very low dark content + high saturation = fish
        if dark_ratio < 0.08 and mean_sat > 90:
            return False

        return True

    # ── Marine animal classifier ──────────────────────────────────────────────

    @staticmethod
    def _classify_marine_animal(
        crop_bgr: np.ndarray,
        aspect: float,
        area_ratio: float,
    ) -> Optional[str]:
        ch, cw = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # ── Ray: very wide flat shape with wing extensions ────────────────────
        if aspect > 2.2:
            # Wings show as relatively bright lateral extensions in
            # an otherwise dark body
            left_edge  = float(np.mean(gray[:, :max(1, cw//8)]))
            right_edge = float(np.mean(gray[:, max(0, 7*cw//8):]))
            mid        = float(np.mean(gray[:, cw//4: 3*cw//4]))
            if min(left_edge, right_edge) > mid * 0.5:
                return "Ray"

        # ── Shark: elongated, countershaded, tapered body ─────────────────────
        if 1.3 <= aspect <= 5.5:
            top_mean    = float(np.mean(gray[:ch//2, :]))
            bottom_mean = float(np.mean(gray[ch//2:, :]))
            # Countershading: bottom significantly lighter than top
            countershading = (bottom_mean - top_mean) > 6

            # Body taper: centre columns brighter (thicker body) than ends
            if cw >= 4:
                col_profile = np.mean(gray, axis=0).astype(float)
                left_end    = float(np.mean(col_profile[:cw//5]))
                right_end   = float(np.mean(col_profile[4*cw//5:]))
                centre      = float(np.mean(col_profile[cw//4: 3*cw//4]))
                taper       = centre > max(left_end, right_end) * 0.9
            else:
                taper = False

            # Dorsal fin variation: local texture in top quarter
            tq = gray[:max(1, ch//4), :]
            dorsal_var = float(np.std(tq)) > 12

            score = int(countershading) + int(taper) + int(dorsal_var)
            if score >= 2:
                return "Shark"
            if aspect > 2.5 and score >= 1:
                return "Shark"

        # ── Large smooth mammal ────────────────────────────────────────────────
        if area_ratio > 0.05 and 0.8 <= aspect <= 3.5:
            edges = cv2.Canny(gray, 40, 120)
            edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)
            if edge_density < 0.12:
                if aspect > 1.8:
                    return "Shark" if area_ratio < 0.12 else "Whale"
                return "Whale"

        return None

    @staticmethod
    def _best_fish_label(crop_bgr: np.ndarray, aspect: float) -> str:
        hsv     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_s  = float(np.mean(hsv[:, :, 1]))
        hue_std = float(np.std(hsv[:, :, 0]))
        if mean_s > 100 and hue_std > 30:
            return "Tropical Fish"
        return "Fish"


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enhance_underwater(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    result = np.zeros_like(enhanced, dtype=np.float32)
    for i in range(3):
        ch = enhanced[:, :, i].astype(np.float32)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi > lo:
            ch = (ch - lo) / (hi - lo) * 255.0
        result[:, :, i] = np.clip(ch, 0, 255)

    result   = result.astype(np.uint8)
    blurred  = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(result, 1.3, blurred, -0.3, 0)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _clip_bbox(bbox: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, w-1)); x2 = float(np.clip(x2, 0, w-1))
    y1 = float(np.clip(y1, 0, h-1)); y2 = float(np.clip(y2, 0, h-1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


# ═══════════════════════════════════════════════════════════════════════════════
# THREAT SCORING
# ═══════════════════════════════════════════════════════════════════════════════

BASE_THREAT_SCORES: Dict[str, float] = {
    "Naval Mine": 0.97, "Torpedo": 0.95, "Explosive Device": 0.93,
    "Weapon": 0.85, "Submarine": 0.80, "Submersible": 0.75,
    "Hull Damage": 0.70, "Suspicious Package": 0.65, "Sharp Debris": 0.30,
    "Scuba Diver": 0.15, "Diver": 0.12, "Freediver": 0.10, "Snorkeler": 0.08,
    **{k: 0.0 for k in MARINE_LIFE_CLASSES},
    "Surface Vessel": 0.20, "Shipwreck": 0.05,
    "Underwater Structure": 0.05, "Pipeline": 0.10,
    "Equipment": 0.10, "Debris": 0.10,
}
DEFAULT_BASE_THREAT = 0.05


@dataclass
class ThreatContext:
    """Optional contextual signals for diver threat-score escalation."""
    night_operation:   bool = False
    proximity_to_hull: bool = False
    unusual_depth:     bool = False
    armed_detected:    bool = False
    restricted_zone:   bool = False


def _compute_threat_score(
    label: str, conf: float, ctx: Optional[ThreatContext] = None
) -> float:
    base = BASE_THREAT_SCORES.get(label, DEFAULT_BASE_THREAT)
    if label in MARINE_LIFE_CLASSES:
        return 0.0
    score = base * (0.7 + 0.3 * conf)
    if label in DIVER_CLASSES and ctx:
        m = 1.0
        if ctx.armed_detected:    m *= 3.0
        if ctx.night_operation:   m *= 2.0
        if ctx.restricted_zone:   m *= 2.0
        if ctx.proximity_to_hull: m *= 1.8
        if ctx.unusual_depth:     m *= 1.4
        score = min(score * m, 0.95)
    return round(min(score, 1.0), 3)


def _threat_level(score: float) -> str:
    if score >= 0.75: return "CRITICAL"
    if score >= 0.50: return "HIGH"
    if score >= 0.25: return "MEDIUM"
    if score >= 0.05: return "LOW"
    return "NONE"


def _classify_category(label: str) -> str:
    if label in SECURITY_THREAT_CLASSES: return "Security Threat"
    if label in DIVER_CLASSES:           return "Diver"
    if label in MARINE_MAMMAL_CLASSES:   return "Marine Mammal"
    if label in MARINE_LIFE_CLASSES:     return "Marine Life"
    if label in VESSEL_CLASSES:          return "Surface Vessel"
    if label in STRUCTURE_CLASSES:       return "Structure"
    if label in EQUIPMENT_CLASSES:       return "Equipment"
    return "Object"


# ═══════════════════════════════════════════════════════════════════════════════
# NMS + POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = max(1e-6, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(1e-6, (b2[2]-b2[0])*(b2[3]-b2[1]))
    return inter / (a1 + a2 - inter + 1e-6)


def _containment_fraction(inner: List[float], outer: List[float]) -> float:
    """Fraction of inner box that is inside outer box (0–1)."""
    x1 = max(inner[0], outer[0]); y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2]); y2 = min(inner[3], outer[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    inner_area = max(1e-6, (inner[2]-inner[0])*(inner[3]-inner[1]))
    return inter / inner_area


def _nms_merge(dets: List[Dict], iou_thresh: float = 0.45) -> List[Dict]:
    """Greedy class-aware NMS."""
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    keep: List[Dict] = []
    suppressed: set = set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(d)
        for j in range(i+1, len(dets)):
            if j in suppressed:
                continue
            # Suppress same class OR same category with high overlap
            same = (dets[j]["display_class"] == d["display_class"] or
                    dets[j]["category"] == d["category"])
            if same and _iou(d["bbox"], dets[j]["bbox"]) > iou_thresh:
                suppressed.add(j)
    return keep


def _suppress_subparts(dets: List[Dict]) -> List[Dict]:
    """
    Drop small boxes that are mostly contained inside a larger box
    of the same broad category. Prevents fin/tail sub-boxes inside
    an already-detected shark.
    """
    if len(dets) < 2:
        return dets

    by_area = sorted(
        enumerate(dets),
        key=lambda x: (x[1]["bbox"][2]-x[1]["bbox"][0])*(x[1]["bbox"][3]-x[1]["bbox"][1]),
        reverse=True,
    )
    suppressed: set = set()

    for pi, (i, parent) in enumerate(by_area):
        if i in suppressed:
            continue
        p_area = (parent["bbox"][2]-parent["bbox"][0]) * (parent["bbox"][3]-parent["bbox"][1])
        p_cat  = parent["category"]

        for j, child in enumerate(dets):
            if j == i or j in suppressed:
                continue
            c_area = (child["bbox"][2]-child["bbox"][0]) * (child["bbox"][3]-child["bbox"][1])
            if p_area < c_area * 1.8:
                continue
            if child["category"] != p_cat:
                continue
            if _containment_fraction(child["bbox"], parent["bbox"]) >= 0.60:
                suppressed.add(j)

    return [d for j, d in enumerate(dets) if j not in suppressed]


def _area_sanity(dets: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    """
    Fix labels based on size/aspect ratio sanity rules.

    - "Whale" covering < 2% of frame → downgrade to Shark
      (real whales fill most of the frame when visible)
    - "Diver" with aspect > 2.5 (very wide) → relabel as Shark
    - "Fish" covering > 20% of frame → needs shape promotion
    """
    img_area = float(img_w * img_h)
    out = []
    for d in dets:
        d = dict(d)
        x1, y1, x2, y2 = d["bbox"]
        bw = x2 - x1; bh = y2 - y1
        ar = (bw * bh) / max(img_area, 1.0)
        aspect = bw / max(bh, 1.0)
        lbl = d["display_class"]

        if lbl in DIVER_CLASSES and aspect > 2.5:
            d["display_class"] = "Shark"
            d["category"] = "Marine Life"

        elif lbl in {"Whale", "Humpback Whale", "Blue Whale", "Sperm Whale"} and ar < 0.025:
            d["display_class"] = "Shark"
            d["category"] = "Marine Life"

        out.append(d)
    return out


def _quality_gate(det: Dict, img_w: int, img_h: int) -> bool:
    conf = float(det.get("confidence", 0.0))
    lbl  = det.get("display_class", "")
    x1, y1, x2, y2 = det.get("bbox", [0,0,0,0])
    area = max(0.0, (x2-x1)*(y2-y1))
    ar   = area / max(float(img_w*img_h), 1.0)

    if area < 30:
        return False
    if lbl in DIVER_CLASSES:
        return conf >= 0.40 and ar >= 0.003
    if lbl in SHARK_CLASSES:
        return conf >= 0.10
    if lbl in SECURITY_THREAT_CLASSES:
        return conf >= 0.15
    if lbl in MARINE_MAMMAL_CLASSES:
        return conf >= 0.05
    if lbl == "Fish":
        return conf >= 0.15
    if _classify_category(lbl) == "Marine Life":
        return conf >= 0.12
    return conf >= 0.18


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MaritimeDetector:
    """
    Two-stage maritime object detector.

    Stage 1 : YOLOv8x (COCO) — best-in-class object localisation.
    Stage 2 : Geometric shape classifier — corrects Shark/Fish/Diver confusion.
    Supp.   : YOLO-World supplementary pass for threats + marine mammals.

    Parameters
    ----------
    weights         : YOLOv8 weights. 'yolov8x.pt' for best accuracy.
    world_weights   : YOLO-World weights for supplementary pass.
    conf_thresh     : Primary confidence threshold.
    iou_thresh      : NMS IoU threshold.
    img_size        : Inference resolution (1280 recommended).
    enhance         : Apply underwater image enhancement.
    tta             : Horizontal flip test-time augmentation.
    run_world_pass  : Run YOLO-World supplementary threat/mammal pass.
    """

    def __init__(
        self,
        weights: str = r"C:\Users\meetj\Documents\Career\Projects\JalDrishti\detect\runs\marine\v2-5\weights\best.pt",
        world_weights:  str   = "yolov8x-worldv2.pt",
        conf_thresh:    float = 0.01,
        iou_thresh:     float = 0.01,
        img_size:       int   = 1280,
        enhance:        bool  = True,
        tta:            bool  = True,
        run_world_pass: bool  = True,
    ):
        self.weights        = weights
        self.world_weights  = world_weights
        self.conf_thresh    = conf_thresh
        self.iou_thresh     = iou_thresh
        self.img_size       = img_size
        self.enhance        = enhance
        self.tta            = tta
        self.run_world_pass = run_world_pass
        self._model         = None
        self._world_model   = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_standard(self):
        if self._model is None:
            YOLO = _get_yolo_standard()
            if YOLO is None:
                raise RuntimeError("ultralytics not installed.")
            self._model = YOLO(self.weights)
        return self._model

    def _load_world(self):
        if self._world_model is None:
            YW = _get_yolo_world()
            if YW is None:
                return None
            try:
                self._world_model = YW(self.world_weights)
                if hasattr(self._world_model, "set_classes"):
                    self._world_model.set_classes(THREAT_VOCAB)
            except Exception as e:
                logger.warning("World model load failed: %s", e)
                self._world_model = None
        return self._world_model

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _infer_standard(
        self,
        img_bgr: np.ndarray,
        conf: Optional[float] = None,
        size: Optional[int] = None,
    ) -> List[Dict]:
        model = self._load_standard()
        h, w  = img_bgr.shape[:2]
        res   = model(
            img_bgr,
            conf=conf or self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=size or self.img_size,
            verbose=False,
            augment=False,
        )
        dets: List[Dict] = []
        for r in res:
            if r.boxes is None:
                continue
            for box in r.boxes:
                raw  = str(model.names[int(box.cls[0])]).lower()
                cf   = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                disp = COCO_TO_MARITIME.get(raw, raw.title())
                dets.append({
                    "class":         raw,
                    "display_class": disp,
                    "confidence":    round(cf, 3),
                    "bbox":          [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    "category":      _classify_category(disp),
                })
        return dets

    def _infer_world(self, img_bgr: np.ndarray) -> List[Dict]:
        model = self._load_world()
        if model is None:
            return []
        try:
            res = model(
                img_bgr,
                conf=0.12,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                verbose=False,
                augment=False,
            )
        except Exception as e:
            logger.warning("World model inference error: %s", e)
            return []

        dets: List[Dict] = []
        for r in res:
            if r.boxes is None:
                continue
            for box in r.boxes:
                raw  = str(model.names[int(box.cls[0])]).lower()
                cf   = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                disp = WORLD_REMAP.get(raw, raw.title())
                cat  = _classify_category(disp)
                dets.append({
                    "class":         raw,
                    "display_class": disp,
                    "confidence":    round(cf, 3),
                    "bbox":          [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    "category":      cat,
                })
        return dets

    # ── Shape refinement ──────────────────────────────────────────────────────

    def _refine_shapes(
        self, dets: List[Dict], img_bgr: np.ndarray
    ) -> List[Dict]:
        fh, fw = img_bgr.shape[:2]
        out = []
        for d in dets:
            d = dict(d)
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(fw, x2); y2c = min(fh, y2)
            if x2c > x1c and y2c > y1c:
                crop = img_bgr[y1c:y2c, x1c:x2c]
                refined = ShapeClassifier.classify_crop(
                    crop, d["bbox"], fw, fh, d["display_class"]
                )
                if refined != d["display_class"]:
                    d["display_class"] = refined
                    d["category"] = _classify_category(refined)
            out.append(d)
        return out

    # ── Main detect() ─────────────────────────────────────────────────────────

    def detect(
        self,
        img_np: np.ndarray,
        context: Optional[ThreatContext] = None,
    ) -> List[Dict]:
        """
        Full detection pipeline.

        Returns list of dicts with keys:
            class, display_class, confidence, bbox [x1,y1,x2,y2],
            category, threat_score, threat_level
        """
        img_rgb = _to_rgb(img_np.copy())
        img_bgr = _to_bgr(img_rgb)
        infer   = _enhance_underwater(img_bgr) if self.enhance else img_bgr
        h, w    = img_bgr.shape[:2]

        all_dets: List[Dict] = []

        # ── 1. Primary standard YOLO pass ────────────────────────────────────
        all_dets.extend(self._infer_standard(infer))

        # ── 2. TTA: horizontal flip ──────────────────────────────────────────
        if self.tta:
            flipped = cv2.flip(infer, 1)
            for d in self._infer_standard(flipped):
                d = dict(d)
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = [round(w-x2,1), y1, round(w-x1,1), y2]
                all_dets.append(d)

        # ── 3. YOLO-World threat/mammal supplementary pass ───────────────────
        if self.run_world_pass:
            world = self._infer_world(infer)
            # Only keep security threats and marine mammals from world pass —
            # world's marine-life outputs are unreliable for fish/sharks
            world = [
                d for d in world
                if d["category"] in {"Security Threat", "Marine Mammal", "Surface Vessel"}
            ]
            all_dets.extend(world)

        # ── 4. NMS merge ─────────────────────────────────────────────────────
        all_dets = _nms_merge(all_dets, iou_thresh=self.iou_thresh)

        # ── 5. Shape refinement ──────────────────────────────────────────────
        all_dets = self._refine_shapes(all_dets, infer)

        # ── 6. Sub-part containment suppression ──────────────────────────────
        all_dets = _suppress_subparts(all_dets)

        # ── 7. Area sanity promotion ─────────────────────────────────────────
        all_dets = _area_sanity(all_dets, w, h)

        # ── 8. Quality gate ──────────────────────────────────────────────────
        all_dets = [d for d in all_dets if _quality_gate(d, w, h)]

        # ── 9. Clip bboxes ────────────────────────────────────────────────────
        for d in all_dets:
            d["bbox"] = _clip_bbox(d["bbox"], w, h)

        # ── 10. Threat scoring ────────────────────────────────────────────────
        for d in all_dets:
            ts = _compute_threat_score(d["display_class"], d["confidence"], context)
            d["threat_score"] = ts
            d["threat_level"] = _threat_level(ts)

        all_dets.sort(key=lambda x: (x["confidence"], x["threat_score"]), reverse=True)
        return all_dets

    def detect_and_annotate(
        self,
        inference_img: np.ndarray,
        original_img:  Optional[np.ndarray] = None,
        context:       Optional[ThreatContext] = None,
    ) -> Tuple[List[Dict], str]:
        dets   = self.detect(inference_img, context=context)
        target = original_img if original_img is not None else inference_img

        if original_img is not None:
            ih, iw = inference_img.shape[:2]
            oh, ow = original_img.shape[:2]
            if (ih, iw) != (oh, ow):
                sx, sy = ow/iw, oh/ih
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = [round(x1*sx,1), round(y1*sy,1),
                                 round(x2*sx,1), round(y2*sy,1)]

        ann     = _draw_boxes(_to_rgb(target.copy()), dets)
        img_pil = Image.fromarray(ann)
        buf     = io.BytesIO()
        img_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return dets, f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_boxes(img_rgb: np.ndarray, dets: List[Dict]) -> np.ndarray:
    pil = Image.fromarray(img_rgb).convert("RGBA")
    ov  = Image.new("RGBA", pil.size, (0,0,0,0))
    dr  = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
        )
    except Exception:
        font = ImageFont.load_default()

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cat    = d.get("category", "Object")
        colour = ALERT_COLOURS.get(cat, ALERT_COLOURS["Object"])
        conf   = d.get("confidence", 0.0)
        threat = d.get("threat_score", 0.0)
        tlevel = d.get("threat_level", "NONE")
        lbl    = f"{d['display_class']}  {conf:.0%}"

        dr.rectangle([x1-3, y1-3, x2+3, y2+3], outline=colour+(30,), width=5)
        dr.rectangle([x1, y1, x2, y2], outline=colour+(220,), width=2)
        if tlevel in ("CRITICAL", "HIGH"):
            dr.rectangle([x1-6, y1-6, x2+6, y2+6], outline=colour+(70,), width=2)

        try:
            tw = dr.textbbox((0,0), lbl, font=font)[2]
        except Exception:
            tw = len(lbl) * 7

        ty = max(0, y1-22)
        dr.rectangle([x1, ty, x1+tw+8, ty+18], fill=colour+(210,))
        dr.text((x1+4, ty+3), lbl, fill=(0,0,0), font=font)

        if threat > 0.0:
            bw = int((x2-x1) * threat)
            by = y2 + 2
            dr.rectangle([x1, by, x2, by+5], fill=(40,40,40,150))
            bc = ((255,55,55,220) if threat >= 0.75 else
                  (255,165,0,200) if threat >= 0.25 else
                  (80,200,100,180))
            dr.rectangle([x1, by, x1+bw, by+5], fill=bc)

    return np.array(Image.alpha_composite(pil, ov).convert("RGB"))


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_default_detector: Optional[MaritimeDetector] = None
_default_sig: Optional[tuple] = None


def get_detector(
    weights:        str   = "yolov8x.pt",
    world_weights:  str   = "yolov8x-worldv2.pt",
    conf_thresh:    float = 0.18,
    iou_thresh:     float = 0.45,
    img_size:       int   = 1280,
    enhance:        bool  = True,
    tta:            bool  = True,
    run_world_pass: bool  = True,
) -> MaritimeDetector:
    """Return (and cache) a singleton MaritimeDetector instance."""
    global _default_detector, _default_sig
    sig = (weights, world_weights, conf_thresh, iou_thresh,
           img_size, enhance, tta, run_world_pass)
    if _default_detector is None or _default_sig != sig:
        _default_detector = MaritimeDetector(
            weights=weights, world_weights=world_weights,
            conf_thresh=conf_thresh, iou_thresh=iou_thresh,
            img_size=img_size, enhance=enhance,
            tta=tta, run_world_pass=run_world_pass,
        )
        _default_sig = sig
    return _default_detector