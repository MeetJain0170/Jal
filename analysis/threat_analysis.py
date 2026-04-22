"""
analysis/threat_analysis.py — Maritime threat scoring from detection results.

Input  : list of detection dicts (from yolo_detector.py)
Output : threat_score (0-100), alert_level, breakdown

Calibration philosophy
-----------------------
• Weapons, mines, torpedoes, unexploded ordnance → HIGH (25-30)
• Unauthorized submersibles/submarines          → HIGH (20-22)
• Hull damage, suspicious packages              → MEDIUM-HIGH (14-18)
• Human divers: context-dependent.
    A solo diver in open water is LOW threat (8) unless accompanied
    by weapons or until proven hostile.  Blanket "person = threat"
    is biologically and operationally wrong.
• Surface vessels                               → MEDIUM (6-8)
• Marine life (sharks, fish, etc.)              → VERY LOW (0-2)
    Sharks are not weapons. Marine biology ≠ security threat.
"""
from __future__ import annotations
from typing import List, Dict, Any

# ── Per-class base threat weights ─────────────────────────────────────────────
# Weight = maximum contribution per detection × confidence
# Divers get a low default weight; they are elevated only if
# contextually flagged (e.g., accompanied by weapons).
THREAT_WEIGHTS: Dict[str, float] = {
    # ─── Confirmed high-threat hardware ─────────────────────────────────────
    "torpedo":           30.0,
    "mine":              28.0,
    "naval mine":        28.0,
    "naval_mine":        28.0,
    "weapon":            22.0,
    "knife":             15.0,
    "explosive":         28.0,
    # ─── Unauthorized vessels / platforms ───────────────────────────────────
    "submarine":         22.0,
    "submersible":       22.0,
    # ─── Infrastructure / equipment anomalies ───────────────────────────────
    "hull damage":       16.0,
    "hull_damage":       16.0,
    "suspicious package":16.0,
    "package":           10.0,
    # ─── Human presence — low default; context escalates ────────────────────
    # A diver at a recreational dive site ≠ security threat.
    # Operators should escalate manually if context warrants.
    "person":             8.0,
    "diver":              8.0,
    "scuba diver":        8.0,
    "human":              8.0,
    # ─── Surface traffic ─────────────────────────────────────────────────────
    "boat":               8.0,
    "ship":               6.0,
    "vessel":             6.0,
    # ─── Debris / entanglement hazards ───────────────────────────────────────
    "fishing net":        4.0,
    "fishing_net":        4.0,
    "debris":             3.0,
    "sharp debris":       5.0,
    "equipment":          2.0,
    # ─── Marine life — NOT a security threat ─────────────────────────────────
    "fish":               0.3,
    "shark":              1.5,   # mild flag — operator curiosity, not danger
    "whale":              0.2,
    "dolphin":            0.1,
    "jellyfish":          0.1,
    "coral":              0.0,
    "turtle":             0.0,
    "sea turtle":         0.0,
    "ray":                0.1,
    "marine life":        0.2,
    "marine animal":      0.2,
    "rock":               0.0,
    "plant":              0.0,
}

# Divers reclassified as Security Threat ONLY when a weapon is co-detected
SECURITY_CLASSES = {
    "mine", "naval mine", "naval_mine", "submersible",
    "submarine", "torpedo", "weapon", "hull_damage", "hull damage",
    "explosive", "suspicious package",
}

# Diver is contextually flagged (Yellow at most alone)
DIVER_CLASSES = {"person", "diver", "scuba diver", "human"}

MARINE_LIFE_CLASSES = {
    "fish", "shark", "whale", "dolphin", "jellyfish",
    "coral", "sea_turtle", "sea turtle", "turtle", "ray",
    "marine life", "marine animal",
}


def _category(label: str) -> str:
    lbl = label.lower()
    if lbl in SECURITY_CLASSES:
        return "Security Threat"
    if lbl in DIVER_CLASSES:
        return "Diver"          # separate from blanket "Security Threat"
    if lbl in MARINE_LIFE_CLASSES:
        return "Marine Life"
    return "Object"


def _proximity_factor(bbox: List[float], img_w: int = 640, img_h: int = 480) -> float:
    """Proximity score 0-1 based on bounding-box area vs image area."""
    if len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = bbox[:4]
    box_area = abs((x2 - x1) * (y2 - y1))
    img_area = max(img_w * img_h, 1)
    return min(1.0, box_area / img_area * 10)


def _visibility_penalty(turbidity_index: float) -> float:
    """Low visibility amplifies operational risk (max +10, down from +15)."""
    return turbidity_index * 10.0


def compute_threat_score(
    detections: List[Dict[str, Any]],
    turbidity_index: float = 0.0,
    img_w: int = 640,
    img_h: int = 480,
) -> Dict[str, Any]:
    """
    Compute a calibrated maritime threat score.

    Logic:
    1. Base score = sum(weight[class] × confidence) for each detection
    2. Diver escalation: if a diver AND a weapon are co-detected,
       the diver's weight is doubled (armed diver scenario).
    3. Proximity bonus for nearest confirmed threat.
    4. Visibility penalty for low-light/turbid water.
    5. Marine life NEVER triggers Yellow or Red on its own.
    """
    if not detections:
        return {
            "threat_score":    0.0,
            "alert_level":     "Green",
            "security_objects": 0,
            "breakdown": {"detection_score": 0, "proximity_bonus": 0, "visibility_penalty": 0},
            "recommendations": ["Continue routine monitoring."],
        }

    # Check for weapon presence (armed diver escalation)
    has_weapon = any(
        str(d.get("class", "")).lower() in {"weapon", "knife", "torpedo", "explosive"}
        for d in detections
    )

    security_objects = [d for d in detections
                        if d.get("category") in {"Security Threat"}]
    security_count = len(security_objects)

    base = 0.0
    for d in detections:
        label = str(d.get("class", "")).lower()
        conf  = float(d.get("confidence", 0.5))
        w     = THREAT_WEIGHTS.get(label, 1.5)

        # Armed-diver escalation
        if label in {"person", "diver", "scuba diver", "human"} and has_weapon:
            w *= 2.0

        base += w * conf

    # Proximity bonus — only for confirmed high-threat hardware (not divers alone)
    hw_threats = [d for d in detections
                  if str(d.get("class", "")).lower() in SECURITY_CLASSES]
    prox_scores = [
        _proximity_factor(d.get("bbox", []), img_w, img_h) * 12.0
        for d in hw_threats
    ]
    prox_score = max(prox_scores) if prox_scores else 0.0

    vis_penalty = _visibility_penalty(turbidity_index)
    raw_score   = base + prox_score + vis_penalty
    threat_score = round(min(100.0, max(0.0, raw_score)), 1)

    # ── Alert level ───────────────────────────────────────────────────────────
    # Green  <20  : routine marine activity, lone diver, pure marine life
    # Yellow 20-55: unverified vessel, lone diver, minor debris
    # Red    >55  : weapons, mines, armed divers, unauthorized subs
    if threat_score < 20:
        alert_level = "Green"
        recommendations = ["Continue routine monitoring."]
    elif threat_score < 55:
        alert_level = "Yellow"
        recommendations = [
            "Increase sensor sweep frequency.",
            "Alert duty officer.",
            "Prepare rapid-response team on standby.",
        ]
    else:
        alert_level = "Red"
        recommendations = [
            "Confirmed high-priority threat — activate security protocol.",
            "Dispatch response team immediately.",
            "Notify command centre and log all detections.",
            "Do not approach without protective detail.",
        ]

    return {
        "threat_score":     threat_score,
        "alert_level":      alert_level,
        "security_objects": security_count,
        "breakdown": {
            "detection_score":    round(base, 2),
            "proximity_bonus":    round(prox_score, 2),
            "visibility_penalty": round(vis_penalty, 2),
        },
        "recommendations": recommendations,
    }