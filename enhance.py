from __future__ import annotations

"""
JalDrishti v2 — Adaptive Underwater Enhancement Pipeline

Key upgrades over v1:
- Scene-aware preprocessing (prevents double enhancement)
- Safer white balance (no red explosion)
- Depth-aware contrast (bounded + stable)
- Modular pipeline stages
- Optional adaptive blending mask
- Robust fallbacks
- Logging-ready structure
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import logging

import config
from analysis.image_quality import compute_all_metrics
from depth.depth_estimator import estimate_depth_tensor

_to_tensor = transforms.ToTensor()

# ------------------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------------------
logger = logging.getLogger("JalDrishti")
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# CONFIG DEFAULTS (SAFE LIMITS)
# ------------------------------------------------------------------------------

CLAHE_CLIP = 1.0
MAX_SATURATION = 0.8
SHARPEN_AMOUNT = 0.3

DEPTH_ALPHA_BASE = 1.02
DEPTH_BETA_BASE = 6
DEPTH_ALPHA_SCALE = 0.05
DEPTH_BETA_SCALE = 8

# ------------------------------------------------------------------------------
# SCENE CLASSIFIER (CRUCIAL FIX)
# ------------------------------------------------------------------------------

def classify_scene(img_rgb_np):
    """
    Heuristic scene classifier:
    Detects hazy/deep vs normal scenes.

    Why:
    Prevents double enhancement (biggest flaw earlier).
    """
    img = img_rgb_np.astype(np.float32) / 255.0

    brightness = np.mean(img)
    contrast = np.std(img)

    # underwater haze signature
    blue_dominance = np.mean(img[:, :, 2]) > np.mean(img[:, :, 0]) * 1.2

    if contrast < 0.12 or blue_dominance:
        return "hazy"
    return "normal"

# ------------------------------------------------------------------------------
# IMPROVED WHITE BALANCE (FIXED)
# ------------------------------------------------------------------------------

def white_balance_underwater(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)

    mean_rg = (np.mean(r) + np.mean(g)) / 2.0
    gain_r = 1 + 0.5 * (mean_rg - np.mean(r)) / (mean_rg + 1e-6)

    r = np.clip(r * gain_r, 0, 255)

    return cv2.merge([b, g, r]).astype(np.uint8)

# ------------------------------------------------------------------------------
# SAFE CLAHE
# ------------------------------------------------------------------------------

def clahe_enhance(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    L = clahe.apply(L)

    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

# ------------------------------------------------------------------------------
# SATURATION CONTROL
# ------------------------------------------------------------------------------

def saturation_clamp(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    sat = hsv[:, :, 1] / 255.0
    sat = np.where(sat > MAX_SATURATION, MAX_SATURATION, sat)

    hsv[:, :, 1] = sat * 255.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ------------------------------------------------------------------------------
# DEHAZE (STABLE)
# ------------------------------------------------------------------------------

def dehaze(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    A = np.percentile(gray, 95)

    t = 1 - 0.75 * (gray / (A + 1e-6))
    t = np.clip(t, 0.4, 1.0)

    t = cv2.merge([t, t, t])
    J = (img.astype(np.float32) - A) / t + A

    return np.clip(J, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------------------
# SHARPEN (CONTROLLED)
# ------------------------------------------------------------------------------

def sharpen(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.addWeighted(img, 1 + SHARPEN_AMOUNT, blur, -SHARPEN_AMOUNT, 0)

# ------------------------------------------------------------------------------
# GAMMA
# ------------------------------------------------------------------------------

def gamma_correct(img, gamma=1.08):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ------------------------------------------------------------------------------
# CLASSICAL PIPELINE (CONDITIONAL)
# ------------------------------------------------------------------------------

def enhance_opencv_adaptive(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    scene = classify_scene(img)

    if scene == "hazy":
        logger.info("Applying full preprocessing")
        img_bgr = white_balance_underwater(img_bgr)
        img_bgr = clahe_enhance(img_bgr)
        img_bgr = dehaze(img_bgr)
    else:
        logger.info("Light preprocessing only")
        img_bgr = white_balance_underwater(img_bgr)

    img_bgr = saturation_clamp(img_bgr)
    img_bgr = sharpen(img_bgr)
    img_bgr = gamma_correct(img_bgr)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ------------------------------------------------------------------------------
# BACKSCATTER (UNCHANGED BUT STABLE)
# ------------------------------------------------------------------------------

def suppress_backscatter(img_rgb_np):
    img = img_rgb_np.astype(np.float32) / 255.0

    A = np.percentile(img, 99, axis=(0, 1))
    t = 1 - 0.85 * (img / (A + 1e-6))
    t = np.clip(t, 0.2, 1.0)

    J = (img - A) / t + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------------------
# DEPTH-AWARE CONTRAST (FIXED)
# ------------------------------------------------------------------------------

def depth_aware_contrast(img, depth):
    img_f = img.astype(np.float32)

    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    far = 1.0 - d

    far = cv2.GaussianBlur(far, (21, 21), 0)
    far = far[:, :, None]

    alpha = DEPTH_ALPHA_BASE + DEPTH_ALPHA_SCALE * far
    beta = DEPTH_BETA_BASE + DEPTH_BETA_SCALE * far

    return np.clip(img_f * alpha + beta, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------------------
# ADAPTIVE BLENDING (NEW 🔥)
# ------------------------------------------------------------------------------

def adaptive_blend(orig, pred):
    """
    Simple heuristic alpha map:
    More blending in low-contrast regions
    """
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    contrast = cv2.Laplacian(gray, cv2.CV_32F)

    alpha = cv2.normalize(np.abs(contrast), None, 0, 1, cv2.NORM_MINMAX)
    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)

    alpha = alpha[:, :, None]

    return (alpha * pred + (1 - alpha) * orig).astype(np.uint8)

# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------

def enhance_image(model, pil_img, use_hybrid=True):

    # STEP 1: Adaptive preprocessing (FIXED major flaw)
    if use_hybrid:
        img = enhance_opencv_adaptive(pil_img)
    else:
        img = pil_img.convert("RGB")

    orig_np = np.array(img)

    # STEP 2: MODEL
    tensor = _to_tensor(orig_np).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        pred = model(tensor)

    pred = torch.clamp(pred, 0, 1)[0].cpu().permute(1, 2, 0).numpy()
    pred = (pred * 255).astype(np.uint8)

    # STEP 3: ADAPTIVE BLENDING (NEW)
    out = adaptive_blend(orig_np, pred)

    # STEP 4: BACKSCATTER
    out = suppress_backscatter(out)

    # STEP 5: DEPTH-AWARE
    try:
        depth = estimate_depth_tensor(out)
        out = depth_aware_contrast(out, depth)
    except Exception:
        logger.warning("Depth unavailable — fallback used")
        out = cv2.convertScaleAbs(out, alpha=1.03, beta=6)

    # STEP 6: FINAL POLISH
    out = cv2.bilateralFilter(out, 5, 35, 35)
    # Keep enhanced luminance but anchor chroma to original scene to avoid
    # magenta/reddish drift on bright marine subjects.
    out_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab[:, :, 1] = 0.70 * out_lab[:, :, 1] + 0.30 * orig_lab[:, :, 1]
    out_lab[:, :, 2] = 0.70 * out_lab[:, :, 2] + 0.30 * orig_lab[:, :, 2]
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # STEP 7: GUARDRAIL AGAINST OVER-DARK / OVER-STYLIZED OUTPUT
    # Some checkpoints can produce cinematic but unrealistic dark-magenta casts.
    # If that happens, softly blend toward classical OpenCV output.
    opencv_ref = np.array(enhance_opencv_adaptive(pil_img))
    orig_ref = np.array(pil_img.convert("RGB"))
    mean_orig = float(np.mean(orig_ref))
    mean_out = float(np.mean(out))
    # brightness ratio below ~0.72 usually looks crushed on UI scenes
    if mean_orig > 1.0 and (mean_out / mean_orig) < 0.72:
        out = cv2.addWeighted(opencv_ref, 0.65, out, 0.35, 0)

    # Clamp excessive chroma shift against original to avoid purple cast
    mean_delta = float(np.mean(np.abs(out.astype(np.float32) - orig_ref.astype(np.float32))))
    if mean_delta > 62.0:
        out = cv2.addWeighted(orig_ref, 0.45, out, 0.55, 0)

    return Image.fromarray(out)

# ------------------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------------------

def calculate_metrics_full(original, enhanced):
    return compute_all_metrics(
        np.array(original.convert("RGB")),
        np.array(enhanced.convert("RGB"))
    )


# Backward-compat alias — api.py imports this name
enhance_opencv = enhance_opencv_adaptive