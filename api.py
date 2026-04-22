from __future__ import annotations

import io
import os
import json
import base64
import time
import uuid
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import cv2

import config
from train.model import UNet, count_parameters
from enhance import enhance_image, enhance_opencv, calculate_metrics_full
from model_loader import load_model as _shared_load_model

from analysis.water_quality import analyze_water_quality
from analysis.threat_analysis import compute_threat_score

# ------------------------------------------------------------------------------
# CONFIG + LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JalAPI")

torch.set_num_threads(1)

app = Flask(__name__, static_folder="static")
CORS(app)
DETECTOR_REVISION = "detector-2026-04-21-r4"

_MODEL: UNet | None = None
_DETECTOR_CACHE = {}
_DEPTH_MODEL = None


# ------------------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------------------

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        model = _shared_load_model()
        if model:
            model.to(config.DEVICE)
            model.eval()
        _MODEL = model
        return model
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None


def _load_detector(profile: str | None = None):
    global _DETECTOR_CACHE
    profile = (profile or getattr(config, "DETECTION_PROFILE", "full")).strip().lower()
    if profile in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[profile]

    try:
        if getattr(config, "ENABLE_DIVER_DETECTOR", False):
            from detection.hybrid_detector import get_detector
            use_shark = getattr(config, "ENABLE_SHARK_DETECTOR", False) and profile in {"full", "shark_focus"}
            use_fish = getattr(config, "ENABLE_FISH_DETECTOR", False) and profile in {"full", "fish_focus"}
            _DETECTOR_CACHE[profile] = get_detector(
                marine_weights=config.YOLO_WEIGHTS,
                marine_conf=config.YOLO_CONF_THRESH,
                marine_iou=config.YOLO_IOU_THRESH,
                marine_imgsz=config.YOLO_IMG_SIZE,
                diver_weights=getattr(config, "DIVER_WEIGHTS", "yolov8s.pt"),
                diver_conf=getattr(config, "DIVER_CONF_THRESH", 0.25),
                diver_iou=getattr(config, "DIVER_IOU_THRESH", 0.45),
                diver_imgsz=getattr(config, "DIVER_IMG_SIZE", config.YOLO_IMG_SIZE),
                shark_weights=getattr(config, "SHARK_WEIGHTS", None) if use_shark else None,
                shark_conf=getattr(config, "SHARK_CONF_THRESH", 0.16),
                shark_iou=getattr(config, "SHARK_IOU_THRESH", 0.45),
                shark_imgsz=getattr(config, "SHARK_IMG_SIZE", config.YOLO_IMG_SIZE),
                fish_weights=getattr(config, "FISH_WEIGHTS", None) if use_fish else None,
                fish_conf=getattr(config, "FISH_CONF_THRESH", 0.10),
                fish_iou=getattr(config, "FISH_IOU_THRESH", 0.45),
                fish_imgsz=getattr(config, "FISH_IMG_SIZE", config.YOLO_IMG_SIZE),
            )
        else:
            from detection.simple_detector import get_detector
            _DETECTOR_CACHE[profile] = get_detector(
                weights=config.YOLO_WEIGHTS,
                conf_thresh=config.YOLO_CONF_THRESH,
                iou_thresh=config.YOLO_IOU_THRESH,
                img_size=config.YOLO_IMG_SIZE,
                tta=getattr(config, "YOLO_TTA", False),
                multi_scale=getattr(config, "YOLO_MULTI_SCALE", True),
            )
        return _DETECTOR_CACHE[profile]
    except Exception as e:
        logger.warning(f"YOLO load failed: {e}")
        return None


def _load_depth():
    """Warm up the MiDaS depth model (lazy — loads on first real call)."""
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    try:
        # depth_estimator uses an internal _load_model() so we trigger it via
        # a tiny dummy call on the module level rather than importing load_midas
        # (which never existed as a public symbol).
        from depth import depth_estimator as _de
        _DEPTH_MODEL = _de._load_model()  # returns (model, transform) tuple
        return _DEPTH_MODEL
    except Exception as e:
        logger.warning(f"Depth preload skipped (will load on first request): {e}")
        return None


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def _read_image(file):
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    return img


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def enhancement_stats(orig, enhanced):
    diff = np.abs(orig.astype(np.float32) - enhanced.astype(np.float32))
    return {
        "mean_enhancement": float(np.mean(diff)),
        "enhanced_pct": float(np.mean(diff > 10) * 100),
    }


def _looks_over_stylized(orig_np, candidate_np, opencv_np):
    """
    Reject enhancement outputs that are unnaturally dark or heavily color-shifted.
    """
    mean_orig = float(np.mean(orig_np))
    mean_cand = float(np.mean(candidate_np))
    bright_ratio = mean_cand / max(mean_orig, 1e-6)

    cand_delta = float(np.mean(np.abs(candidate_np.astype(np.float32) - orig_np.astype(np.float32))))
    op_delta = float(np.mean(np.abs(opencv_np.astype(np.float32) - orig_np.astype(np.float32))))

    # Candidate should not be dramatically darker than input.
    too_dark = bright_ratio < 0.80
    # Candidate should not diverge wildly more than stable classical baseline.
    too_shifted = cand_delta > max(40.0, op_delta * 1.35)
    return too_dark or too_shifted


def _tone_balance_hybrid(orig_np, hybrid_np):
    """
    Keep hybrid color mood close to original underwater scene.
    Prevent warm/red cast while preserving enhancement detail.
    """
    h = hybrid_np.astype(np.float32)
    o = orig_np.astype(np.float32)

    h_means = np.mean(h, axis=(0, 1)) + 1e-6  # R, G, B
    o_means = np.mean(o, axis=(0, 1)) + 1e-6

    # If red channel is disproportionately boosted, pull it back.
    red_boost = (h_means[0] / max(h_means[2], 1e-6)) / (o_means[0] / max(o_means[2], 1e-6))
    if red_boost > 1.12:
        h[:, :, 0] *= 0.84
        h[:, :, 2] *= 1.04

    # Preserve overall brightness relative to original scene.
    ratio = float(np.mean(h) / max(np.mean(o), 1e-6))
    if ratio < 0.86:
        h = cv2.convertScaleAbs(h.astype(np.uint8), alpha=1.06, beta=4).astype(np.float32)
    elif ratio > 1.20:
        h *= 0.95

    h = np.clip(h, 0, 255).astype(np.uint8)
    # Slightly stronger enhancement while keeping natural tint.
    mixed = cv2.addWeighted(o.astype(np.uint8), 0.16, h, 0.84, 0.0)

    # Mild local contrast boost (underwater-safe)
    lab = cv2.cvtColor(mixed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l)
    mixed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return mixed


def _fallback_depth_result(img_np):
    """Create a heuristic depth map so UI depth is never empty."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, _ = gray.shape

    vertical = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    depth = np.clip(0.7 * (1.0 - gray) + 0.3 * vertical, 0.0, 1.0)

    # Smooth white->black gradient map (near=white, far=black)
    depth_u8 = np.clip((1.0 - depth) * 255.0, 0, 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)

    near_mask = depth >= 0.66
    mid_mask = (depth >= 0.33) & (depth < 0.66)
    far_mask = depth < 0.33

    return {
        "status": "ok",
        "depth_map": _pil_to_b64(Image.fromarray(depth_rgb)),
        "average_depth": float(np.mean(depth)),
        "object_distances": [
            {"zone": "Near", "pixels": int(np.sum(near_mask))},
            {"zone": "Mid", "pixels": int(np.sum(mid_mask))},
            {"zone": "Far", "pixels": int(np.sum(far_mask))},
        ],
        "depth_source": "fallback",
    }


def _fallback_marine_detection(infer_np, original_np=None):
    """
    Offline-safe fallback detector:
    find the largest salient foreground region and label as Marine Life.
    """
    h, w = infer_np.shape[:2]
    gray = cv2.cvtColor(infer_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    best_area = 0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area > best_area and area > 0.01 * w * h:
            best_area = area
            bbox = [float(x), float(y), float(x + bw), float(y + bh)]

    if bbox is None:
        # conservative central region fallback
        bbox = [0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h]

    det = {
        "class": "marine_life",
        "display_class": "Marine Life",
        "confidence": 0.35,
        "bbox": [round(v, 1) for v in bbox],
        "category": "Marine Life",
        "threat_score": 0.0,
        "threat_level": "LOW",
    }

    canvas = original_np if original_np is not None else infer_np
    ann = canvas.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(ann, (x1, y1), (x2, y2), (52, 211, 153), 2)
    cv2.rectangle(ann, (x1, max(0, y1 - 18)), (x1 + 130, y1), (52, 211, 153), -1)
    cv2.putText(ann, "Marine Life 35%", (x1 + 4, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    annotated = _pil_to_b64(Image.fromarray(ann))
    return [det], annotated


def _has_meaningful_detections(detections, img_np):
    if not detections:
        return False
    area_img = float(img_np.shape[0] * img_np.shape[1])
    best_ratio = 0.0
    for d in detections:
        x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
        area = max(0.0, (x2 - x1) * (y2 - y1))
        best_ratio = max(best_ratio, area / max(area_img, 1.0))
    return best_ratio >= 0.015


def _iou_xyxy(b1, b2):
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1e-6, (float(b1[2]) - float(b1[0])) * (float(b1[3]) - float(b1[1])))
    a2 = max(1e-6, (float(b2[2]) - float(b2[0])) * (float(b2[3]) - float(b2[1])))
    return inter / (a1 + a2 - inter + 1e-6)


def _rescue_underwater_structure(infer_np, detections):
    """
    Add one conservative 'Underwater Structure' box for large foreground
    statue/ruin-like regions when YOLO misses them.
    """
    if infer_np is None or infer_np.size == 0:
        return detections
    # Avoid structure rescue in diver scenes; this commonly boxes fish schools.
    if any(d.get("category") == "Diver" or d.get("display_class") == "Diver" for d in detections):
        return detections
    marine_count = sum(1 for d in detections if d.get("category") == "Marine Life")
    # Do not add structure rescue whenever marine life is already detected.
    if marine_count >= 1:
        return detections
    if any((d.get("display_class") == "Underwater Structure") for d in detections):
        return detections

    h, w = infer_np.shape[:2]
    y0 = int(0.38 * h)  # focus on lower foreground where statues/ruins appear
    roi = infer_np[y0:, :]
    if roi.size == 0:
        return detections

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox = None
    best_score = 0.0
    min_area = 0.030 * w * h

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < min_area:
            continue
        x1, y1 = float(x), float(y + y0)
        x2, y2 = float(x + bw), float(y + bh + y0)
        aspect = bw / max(float(bh), 1.0)
        # Statues/ruins are often broad and foreground-heavy.
        if aspect < 0.45 or aspect > 5.5:
            continue
        # Prefer wide lower-frame candidates.
        lower_bias = (y2 / max(float(h), 1.0))
        score = (area / max(float(w * h), 1.0)) * (0.7 + 0.3 * lower_bias)
        if score > best_score:
            best_score = score
            best_bbox = [x1, y1, x2, y2]

    if best_bbox is None:
        return detections

    # Avoid adding structure over a detected diver.
    for d in detections:
        if d.get("category") == "Diver":
            if _iou_xyxy(best_bbox, d.get("bbox", [0, 0, 0, 0])) > 0.22:
                return detections

    rescued = {
        "class": "underwater_structure",
        "display_class": "Underwater Structure",
        "confidence": 0.31,
        "bbox": [round(v, 1) for v in best_bbox],
        "category": "Object",
        "threat_score": 0.0,
        "threat_level": "LOW",
    }
    return detections + [rescued]


def _augment_marine_recall(detector, infer_np, detections):
    """
    If marine detections are too sparse, run an auxiliary marine-only pass and
    merge non-overlapping fish/marine candidates.
    """
    marine_now = [d for d in detections if d.get("category") == "Marine Life"]
    if len(marine_now) >= 3:
        return detections
    diver_present = any(d.get("category") == "Diver" or d.get("display_class") == "Diver" for d in detections)
    if detector is None or not hasattr(detector, "marine_detector"):
        return detections

    try:
        extra = detector.marine_detector.detect(infer_np)
    except Exception:
        return detections

    h, w = infer_np.shape[:2]
    img_area = max(1.0, float(h * w))
    merged = list(detections)
    added = 0
    for d in sorted(extra, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
        lbl = (d.get("display_class") or "").lower()
        is_marineish = (d.get("category") == "Marine Life") or (
            lbl in {"fish", "eel", "ray", "shark", "marine animal", "animal other"}
        )
        if not is_marineish:
            continue
        conf = float(d.get("confidence", 0.0))
        x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / img_area
        min_conf = max(0.15, 0.18 if diver_present else 0.15)
        min_area_ratio = 0.0015 if diver_present else 0.0008
        if conf < min_conf or area_ratio < min_area_ratio:
            continue
        if area_ratio > 0.28 and conf < 0.75:
            continue
        if any(_iou_xyxy(d.get("bbox", [0, 0, 0, 0]), e.get("bbox", [0, 0, 0, 0])) > 0.55 for e in merged):
            continue
        d = dict(d)
        d["source"] = "api_marine_recall"
        merged.append(d)
        added += 1
        if added >= (5 if diver_present else 8):
            break
    return merged


def _augment_world_marine(infer_np, detections, *, only_when_empty: bool = True):
    """
    Secondary marine recall using YOLO-World when fish count is too low but
    the scene clearly contains marine activity.
    """
    marine_now = [d for d in detections if d.get("category") == "Marine Life"]
    if len(marine_now) >= 4:
        return detections
    if only_when_empty and len(marine_now) > 0:
        return detections

    try:
        from detection.simple_detector import MaritimeDetector
        world = MaritimeDetector(
            weights="yolov8l-worldv2.pt",
            conf_thresh=0.02,
            iou_thresh=0.45,
            img_size=min(1280, config.YOLO_IMG_SIZE),
        )
        extra = world.detect(infer_np)
    except Exception:
        return detections

    merged = list(detections)
    added = 0
    for d in sorted(extra, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
        lbl = (d.get("display_class") or "").lower()
        if lbl not in {"fish", "ray", "shark", "marine animal", "animal other", "eel"}:
            continue
        x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
        area = max(0.0, (x2 - x1) * (y2 - y1))
        area_ratio = area / max(1.0, float(infer_np.shape[0] * infer_np.shape[1]))
        if float(d.get("confidence", 0.0)) < 0.20 or area < 900:
            continue
        if area_ratio > 0.30 and float(d.get("confidence", 0.0)) < 0.82:
            continue
        if any(_iou_xyxy(d.get("bbox", [0, 0, 0, 0]), e.get("bbox", [0, 0, 0, 0])) > 0.58 for e in merged):
            continue
        d = dict(d)
        d["category"] = "Marine Life"
        d["source"] = "api_world_marine"
        if d.get("display_class") == "Animal Other":
            d["display_class"] = "Marine Animal"
        merged.append(d)
        added += 1
        if added >= 12:
            break
    return merged


def _scene_postprocess(infer_np, detections):
    """
    Scene-aware cleanup:
    - deduplicate near-identical boxes
    - suppress low-value clutter in fish-dense reef scenes
    - relabel tiny false "Shark" boxes to "Fish"
    """
    if not detections:
        return detections

    h, w = infer_np.shape[:2]
    img_area = max(1.0, float(h * w))

    # 1) Sort by confidence and dedupe near-identical boxes.
    ordered = sorted(detections, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    dedup = []
    for d in ordered:
        db = d.get("bbox", [0, 0, 0, 0])
        same = False
        for k in dedup:
            kb = k.get("bbox", [0, 0, 0, 0])
            if _iou_xyxy(db, kb) > 0.65:
                same_label = (d.get("display_class") == k.get("display_class")) or (d.get("category") == k.get("category"))
                if same_label:
                    same = True
                    break
        if not same:
            dedup.append(d)

    marine_count = sum(1 for d in dedup if d.get("category") == "Marine Life")
    diver_items = [d for d in dedup if d.get("category") == "Diver" or d.get("display_class") == "Diver"]
    diver_present = len(diver_items) > 0

    cleaned = []
    for d in dedup:
        item = dict(d)
        lbl = (item.get("display_class") or "").lower()
        cat = item.get("category")
        conf = float(item.get("confidence", 0.0))
        if conf < 0.15:
            continue
        x1, y1, x2, y2 = item.get("bbox", [0, 0, 0, 0])
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / img_area
        aspect = bw / max(bh, 1e-6)

        # 2) In fish-dense scenes, drop weak clutter labels.
        if marine_count >= 5 and cat == "Object" and conf < 0.45:
            continue

        # 2b) In diver+macrofauna scenes, suppress tiny low-confidence marine speckles.
        if diver_present and marine_count >= 3 and cat == "Marine Life":
            if area_ratio < 0.0012 and conf < 0.16:
                continue

        # 3) Common reef false-positive: tiny fish labeled as Shark.
        if lbl == "shark" and area_ratio < 0.02 and aspect < 2.2 and conf < 0.75:
            item["display_class"] = "Fish"
            item["class"] = "fish"
            item["category"] = "Marine Life"

        # 3b) Hard sanity: fish-like boxes should not occupy huge frame area
        # unless confidence is very strong.
        if item.get("category") == "Marine Life" and lbl in {"fish", "tropical fish", "marine animal", "animal other", "eel"}:
            if area_ratio > 0.30 and conf < 0.82:
                continue
            # Normalize ambiguous marine labels to fish for cleaner UX.
            if lbl in {"animal other", "marine animal", "tropical fish", "eel"}:
                item["display_class"] = "Fish"
                item["class"] = "fish"
                item["category"] = "Marine Life"

        # 4) Promote very large marine body to Sea Turtle in diver scenes.
        if diver_present and item.get("category") == "Marine Life":
            if area_ratio > 0.11 and 0.7 <= aspect <= 2.6 and conf >= 0.12:
                item["display_class"] = "Sea Turtle"
                item["class"] = "sea_turtle"
                item["category"] = "Marine Life"

        # 5) Diver false-positive guard: turtle/large fauna often gets tagged as
        # diver when the shell/flippers dominate a horizontal box.
        if item.get("category") == "Diver" or item.get("display_class") == "Diver":
            overlap_marine = 0.0
            ib = item.get("bbox", [0, 0, 0, 0])
            for m in dedup:
                if m is item:
                    continue
                if m.get("category") != "Marine Life":
                    continue
                overlap_marine = max(overlap_marine, _iou_xyxy(ib, m.get("bbox", [0, 0, 0, 0])))
            # Re-label to Sea Turtle when diver box is broad and marine-overlapping.
            if (
                marine_count >= 2
                and aspect > 1.15
                and area_ratio > 0.018
                and overlap_marine > 0.18
            ):
                item["display_class"] = "Sea Turtle"
                item["class"] = "sea_turtle"
                item["category"] = "Marine Life"

        cleaned.append(item)

    # Keep output readable without collapsing valid schools.
    cleaned.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return cleaned[:40]


def _strict_marine_filter(detections):
    """
    Keep only diver + fish/shark marine outputs in strict mode.
    Suppresses noisy classes like Net/Underwater Structure for now.
    """
    out = []
    for d in detections:
        cls = (d.get("display_class") or "").lower().strip()
        cat = (d.get("category") or "").strip()

        if cls == "diver" or cat == "Diver":
            out.append(d)
            continue

        if cls in {"fish", "shark", "ray", "sea turtle", "whale", "dolphin", "jellyfish", "octopus"}:
            nd = dict(d)
            if cls not in {"shark", "ray", "sea turtle", "whale", "dolphin", "jellyfish", "octopus"}:
                nd["display_class"] = "Fish"
                nd["class"] = "fish"
            nd["category"] = "Marine Life"
            out.append(nd)
            continue

    return out


# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ------------------------------------------------------------------------------
# ENHANCE
# ------------------------------------------------------------------------------

@app.route("/api/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        t0 = time.perf_counter()

        img = _read_image(request.files["image"])
        orig_np = np.array(img)
        model = _load_model()

        fast_mode = request.form.get("fast", "false").lower() == "true"

        if model is None:
            logger.warning("Enhancement model unavailable, using OpenCV fallback output.")
            enhanced_hybrid = enhance_opencv(img)
            model_fallback = True
        else:
            enhanced_hybrid = enhance_image(model, img, use_hybrid=not fast_mode)
            model_fallback = False

        enhanced_op = enhance_opencv(img)

        # If model output looks visually unstable, snap to classical result.
        if not model_fallback:
            enh_np_check = np.array(enhanced_hybrid)
            op_np_check = np.array(enhanced_op)
            if _looks_over_stylized(orig_np, enh_np_check, op_np_check):
                logger.warning("Hybrid output over-stylized; falling back to OpenCV enhancement.")
                enhanced_hybrid = enhanced_op
                model_fallback = True
            else:
                # Keep hybrid distinct from OpenCV while still preventing harsh casts.
                # Use a light stabilization blend only when needed.
                delta_vs_op = float(np.mean(np.abs(
                    enh_np_check.astype(np.float32) - op_np_check.astype(np.float32)
                )))
                # Only blend if hybrid drifts too far from classical baseline.
                if delta_vs_op > 26.0:
                    blended = cv2.addWeighted(
                        op_np_check, 0.35,
                        enh_np_check, 0.65,
                        0.0
                    )
                    enhanced_hybrid = Image.fromarray(blended.astype(np.uint8))

            # Final tone harmonization for more natural underwater color.
            enhanced_hybrid = Image.fromarray(
                _tone_balance_hybrid(orig_np, np.array(enhanced_hybrid))
            )

        enh_np = np.array(enhanced_hybrid)
        elapsed = round(time.perf_counter() - t0, 3)

        metrics = calculate_metrics_full(img, enhanced_hybrid)
        stats = enhancement_stats(orig_np, enh_np)

        w, h = img.size  # PIL size is (width, height)

        return jsonify({
            "success": True,
            "enhanced_image_hybrid": _pil_to_b64(enhanced_hybrid),
            "enhanced_image_opencv": _pil_to_b64(enhanced_op),
            "enhancement_fallback": model_fallback,
            "processing_time": elapsed,
            "size": f"{w}×{h}",
            "psnr":  metrics["psnr"],
            "ssim":  metrics["ssim"],
            "uiqm":  metrics["uiqm"],
            "uciqe": metrics["uciqe"],
            "eps":   metrics["eps"],
            **stats
        })

    except Exception as e:
        logger.error(f"Enhance failed: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# DETECT
# ------------------------------------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # `image` is both inference and annotation image (enhanced path).
        infer_img = _read_image(request.files["image"])
        infer_np = np.array(infer_img)

        original_np = None
        if "original" in request.files:
            try:
                original_np = np.array(_read_image(request.files["original"]))
            except Exception:
                original_np = None

        if request.form.get("enhance_first", "false") == "true":
            model = _load_model()
            if model:
                infer_img = enhance_image(model, infer_img, use_hybrid=False)
                infer_np = np.array(infer_img)
        elif getattr(config, "DETECT_ON_OPENCV_POLISH", False):
            # Classical OpenCV polish at full resolution for detection
            polished = enhance_opencv(infer_img)
            infer_np = np.array(polished)

        detection_mode = request.form.get("mode", getattr(config, "DETECTION_MODE", "marine_clean")).strip().lower()
        if detection_mode not in {"marine_clean", "full_debug"}:
            detection_mode = "marine_clean"

        strict_labels_only = True if detection_mode == "marine_clean" else bool(getattr(config, "STRICT_MARINE_LABELS_ONLY", True))
        enable_world_marine = False if detection_mode == "marine_clean" else bool(getattr(config, "ENABLE_WORLD_MARINE_AUGMENT", False))
        enable_structure_rescue = False if detection_mode == "marine_clean" else bool(getattr(config, "ENABLE_STRUCTURE_RESCUE", False))

        detect_profile = request.form.get("detection_profile", getattr(config, "DETECTION_PROFILE", "full"))
        detector = _load_detector(profile=detect_profile)
        if detector is None:
            return jsonify({"error": "YOLO unavailable"}), 200

        try:
            detections, annotated = detector.detect_and_annotate(
                infer_np,
                original_img=infer_np
            )
        except Exception as det_err:
            logger.warning(f"Primary detector failed, trying fallback model: {det_err}")
            from detection.yolo_detector import get_detector
            fallback = get_detector(
                weights="yolov8s-worldv2.pt",
                conf_thresh=max(0.08, config.YOLO_CONF_THRESH),
                iou_thresh=config.YOLO_IOU_THRESH,
                img_size=min(1280, config.YOLO_IMG_SIZE),
                tta=False,
                multi_scale=True,
            )
            detections, annotated = fallback.detect_and_annotate(
                infer_np,
                original_img=infer_np
            )

        # Final fallback: if still weak, run a permissive world-model pass
        # tuned for marine fauna (ray, fish, turtle, shark).
        if not _has_meaningful_detections(detections, infer_np):
            from detection.simple_detector import MaritimeDetector
            soft = MaritimeDetector(
                weights="yolov8l-worldv2.pt",
                conf_thresh=0.03,
                iou_thresh=0.45,
                img_size=min(1280, config.YOLO_IMG_SIZE),
            )
            detections, annotated = soft.detect_and_annotate(
                infer_np,
                original_img=infer_np
            )
        if not _has_meaningful_detections(detections, infer_np):
            detections, annotated = _fallback_marine_detection(infer_np, infer_np)

        detections = _augment_marine_recall(detector, infer_np, detections)
        if enable_world_marine and not _has_meaningful_detections(detections, infer_np):
            detections = _augment_world_marine(infer_np, detections, only_when_empty=True)
        if enable_structure_rescue:
            detections = _rescue_underwater_structure(infer_np, detections)
        detections = _scene_postprocess(infer_np, detections)
        if strict_labels_only:
            detections = _strict_marine_filter(detections)
        min_conf = float(getattr(config, "DETECTION_MIN_CONFIDENCE", 0.15))
        detections = [d for d in detections if float(d.get("confidence", 0.0)) >= min_conf]
        try:
            from detection.hybrid_detector import _draw as _draw_hybrid
            annotated_np = _draw_hybrid(infer_np, detections)
            annotated = _pil_to_b64(Image.fromarray(annotated_np))
        except Exception:
            pass

        water = analyze_water_quality(infer_np)
        threat = compute_threat_score(
            detections,
            turbidity_index=water.get("turbidity_index", 0),
            img_w=infer_np.shape[1],
            img_h=infer_np.shape[0],
        )

        return jsonify({
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "detector_revision": DETECTOR_REVISION,
            "detection_mode": detection_mode,
            "detection_profile": detect_profile,
            "annotated_image": annotated,
            "threat_score": threat["threat_score"],
            "alert_level": threat["alert_level"],
            "recommendations": threat.get("recommendations", []),
            "security_objects": threat.get("security_objects", 0),
            "threat_breakdown": threat.get("breakdown", {}),
        })

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        # Keep UX functional even if detector dependencies fail.
        try:
            infer_np_local = locals().get("infer_np")
            original_np_local = locals().get("original_np")
            if infer_np_local is None:
                infer_img = _read_image(request.files["image"])
                infer_np_local = np.array(infer_img)
            if original_np_local is None and "original" in request.files:
                try:
                    original_np_local = np.array(_read_image(request.files["original"]))
                except Exception:
                    original_np_local = None
            detections, annotated = _fallback_marine_detection(infer_np_local, original_np_local)
            water = analyze_water_quality(infer_np_local)
            threat = compute_threat_score(
                detections,
                turbidity_index=water.get("turbidity_index", 0),
                img_w=infer_np_local.shape[1],
                img_h=infer_np_local.shape[0],
            )
            return jsonify({
                "success": True,
                "fallback_detection": True,
                "detections": detections,
                "detection_count": len(detections),
                "detector_revision": DETECTOR_REVISION,
                "annotated_image": annotated,
                "threat_score": threat["threat_score"],
                "alert_level": threat["alert_level"],
                "recommendations": threat.get("recommendations", []),
                "security_objects": threat.get("security_objects", 0),
                "threat_breakdown": threat.get("breakdown", {}),
            })
        except Exception:
            return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# DEPTH
# ------------------------------------------------------------------------------

@app.route("/api/depth", methods=["POST"])
def depth():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = _read_image(request.files["image"])
        img_np = np.array(img)

        t0 = time.perf_counter()

        try:
            from depth.depth_estimator import estimate_depth
            result = estimate_depth(img_np)
        except Exception as e:
            logger.warning(f"[depth] primary failed: {e}")
            result = {"status": "error", "error": str(e)}

        if result.get("status") != "ok" or not result.get("depth_map"):
            logger.warning("Depth model unavailable, returning heuristic fallback depth map.")
            result = _fallback_depth_result(img_np)

        elapsed = round(time.perf_counter() - t0, 3)

        result["processing_time"] = elapsed
        result["success"] = result.get("status", "ok") == "ok"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# WATER ANALYSIS
# ------------------------------------------------------------------------------

@app.route("/api/analyze-water", methods=["POST"])
def analyze_water():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = _read_image(request.files["image"])
        img_np = np.array(img)

        t0 = time.perf_counter()
        wq = analyze_water_quality(img_np)
        elapsed = round(time.perf_counter() - t0, 3)

        return jsonify({
            "success": True,
            **wq,
            "processing_time": elapsed,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# GALLERY
# ------------------------------------------------------------------------------

@app.route("/api/gallery")
def gallery_list():
    try:
        if os.path.exists(config.GALLERY_JSON):
            with open(config.GALLERY_JSON, "r") as f:
                entries = json.load(f)
        else:
            entries = []
        return jsonify({"success": True, "entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gallery/save", methods=["POST"])
def gallery_save():
    try:
        data = request.get_json(force=True)
        entry = {
            "enhanced_b64": data.get("enhanced_b64", ""),
            "filename":     data.get("filename", "image"),
            "saved_at":     time.strftime("%H:%M:%S"),
            "psnr":         data.get("psnr"),
            "ssim":         data.get("ssim"),
            "uiqm":         data.get("uiqm"),
            "uciqe":        data.get("uciqe"),
            "processing_time": data.get("processing_time"),
        }
        entries = []
        if os.path.exists(config.GALLERY_JSON):
            with open(config.GALLERY_JSON, "r") as f:
                entries = json.load(f)
        entries.insert(0, entry)          # newest first
        entries = entries[:100]           # cap at 100
        with open(config.GALLERY_JSON, "w") as f:
            json.dump(entries, f)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/gallery/clear", methods=["DELETE"])
def gallery_clear():
    try:
        if os.path.exists(config.GALLERY_JSON):
            os.remove(config.GALLERY_JSON)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# STATUS
# ------------------------------------------------------------------------------

@app.route("/api/status")
def status():
    model = _load_model()
    yolo_ok = _load_detector(profile=getattr(config, "DETECTION_PROFILE", "full")) is not None
    midas_ok = _load_depth() is not None

    return jsonify({
        "model_loaded": model is not None,
        "device": str(config.DEVICE),
        "parameters": count_parameters(model) if model else None,
        "modules": {
            "yolo": yolo_ok,
            "midas": midas_ok,
            "opencv": True,
        },
        "detection_profile": getattr(config, "DETECTION_PROFILE", "full"),
    })


# ------------------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nJalDrishti FULL API Running...\n")
    _load_model()        # Pre-load U-Net enhancement model
    # Depth model loads lazily on first /api/depth request
    app.run(host="0.0.0.0", port=5500, debug=False, threaded=True)

