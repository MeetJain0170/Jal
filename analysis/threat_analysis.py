"""
analysis/threat_analysis.py — Maritime threat scoring from detection results.

Input  : list of detection dicts (from yolo_detector.py)
Output : threat_score (0-100), alert_level, breakdown
"""
from __future__ import annotations
from typing import List, Dict, Any

# Threat weights per YOLO class label
THREAT_WEIGHTS: Dict[str, float] = {
    # High-threat
    "person":        18.0,   # diver
    "diver":         20.0,
    "mine":          25.0,
    "naval_mine":    25.0,
    "submersible":   22.0,
    "submarine":     22.0,
    "torpedo":       28.0,
    "weapon":        20.0,
    "hull_damage":   18.0,
    # Medium-threat
    "boat":           8.0,
    "ship":           6.0,
    "fishing_net":    5.0,
    "debris":         4.0,
    # Low-threat / marine life (ignored in scoring)
    "fish":           0.5,
    "shark":          1.0,
    "jellyfish":      0.2,
    "coral":          0.0,
}

SECURITY_CLASSES = {
    "person", "diver", "mine", "naval_mine", "submersible",
    "submarine", "torpedo", "weapon", "hull_damage",
}

MARINE_LIFE_CLASSES = {
    "fish", "shark", "whale", "dolphin", "jellyfish",
    "coral", "sea_turtle", "ray",
}


def _category(label: str) -> str:
    if label in SECURITY_CLASSES:
        return "Security Threat"
    if label in MARINE_LIFE_CLASSES:
        return "Marine Life"
    return "Object"


def _proximity_factor(bbox: List[float], img_w: int = 640, img_h: int = 480) -> float:
    """
    Proximity score: 0 (far/small) to 1 (large/centred).
    Computed from bounding box area relative to image area.
    """
    if len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = bbox[:4]
    box_area = abs((x2 - x1) * (y2 - y1))
    img_area = max(img_w * img_h, 1)
    return min(1.0, box_area / img_area * 10)


def _visibility_penalty(turbidity_index: float) -> float:
    """Low visibility = harder to respond = higher threat."""
    return turbidity_index * 15.0


def compute_threat_score(
    detections: List[Dict[str, Any]],
    turbidity_index: float = 0.0,
    img_w: int = 640,
    img_h: int = 480,
) -> Dict[str, Any]:
    """
    Compute a threat score from YOLO detections and water conditions.

    Parameters
    ----------
    detections      : list of dicts with keys: class, confidence, bbox, category
    turbidity_index : from water_quality module (0..1)
    img_w, img_h    : image dimensions for proximity calculation

    Returns
    -------
    dict:
        threat_score      : float 0-100
        alert_level       : "Green" | "Yellow" | "Red"
        security_objects  : int — number of threat-class detections
        breakdown         : dict with sub-scores
        recommendations   : list of action strings
    """
    if not detections:
        raw_score = 0.0
        security_count = 0
        prox_score = 0.0
    else:
        security_objects = [d for d in detections if d.get("category") == "Security Threat"]
        security_count   = len(security_objects)

        # Base score from detection weights × confidence
        base = 0.0
        for d in detections:
            label = str(d.get("class", "")).lower()
            conf  = float(d.get("confidence", 0.5))
            w     = THREAT_WEIGHTS.get(label, 2.0)
            base += w * conf

        # Proximity bonus for closest/largest threat object
        prox_scores = [
            _proximity_factor(d.get("bbox", []), img_w, img_h) * 15.0
            for d in security_objects
        ]
        prox_score = max(prox_scores) if prox_scores else 0.0

        # Visibility penalty
        vis_penalty = _visibility_penalty(turbidity_index)

        raw_score = base + prox_score + vis_penalty

    # Clamp to 0-100
    threat_score = round(min(100.0, max(0.0, raw_score)), 1)

    if threat_score < 25:
        alert_level = "Green"
        recommendations = ["Continue routine monitoring."]
    elif threat_score < 60:
        alert_level = "Yellow"
        recommendations = [
            "Increase sensor sweep frequency.",
            "Alert duty officer.",
            "Prepare rapid-response team.",
        ]
    else:
        alert_level = "Red"
        recommendations = [
            "Immediate threat detected — activate security protocol.",
            "Dispatch response team.",
            "Notify command centre.",
            "Log all detections with timestamps.",
        ]

    return {
        "threat_score":     threat_score,
        "alert_level":      alert_level,
        "security_objects": security_count if detections else 0,
        "breakdown": {
            "detection_score":   round(raw_score - (prox_score if detections else 0) - _visibility_penalty(turbidity_index), 2),
            "proximity_bonus":   round(prox_score if detections else 0, 2),
            "visibility_penalty":round(_visibility_penalty(turbidity_index), 2),
        },
        "recommendations": recommendations,
    }