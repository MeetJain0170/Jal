"""
analysis/water_quality.py — Underwater water quality & visibility analysis.

Algorithms:
    • Dark Channel Prior — turbidity / haze density estimation
    • Attenuation model — visibility range estimation
    • RGB channel loss analysis

Public API:
    analyze_water_quality(img_np) -> dict
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def _dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Dark Channel Prior (He et al., 2009).
    img : HxWx3 float32 in [0, 1]
    Returns dark channel map (H, W) — higher = more haze.
    """
    min_c  = img[:, :, 1:].min(axis=2) # UDCP: exclude red channel 
    kernel = np.ones((patch_size, patch_size), dtype=np.float32)
    h, w   = min_c.shape
    pad    = patch_size // 2

    dark = np.full_like(min_c, 1.0)
    padded = np.pad(min_c, pad, mode='edge')
    for i in range(h):
        for j in range(w):
            dark[i, j] = padded[i:i+patch_size, j:j+patch_size].min()
    return dark


def _fast_dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Fast approximation using rank-order filtering via uniform minimum.
    Uses scipy if available, otherwise falls back to sliding minimum.
    """
    min_c = img[:, :, 1:].min(axis=2) # UDCP: exclude red channel 
    try:
        from scipy.ndimage import minimum_filter
        return minimum_filter(min_c, size=patch_size)
    except ImportError:
        return _dark_channel(img, patch_size)


def estimate_turbidity(img: np.ndarray) -> dict:
    """
    Estimate turbidity using Dark Channel Prior.

    Returns:
        turbidity_index    : 0 (clear) to 1 (very turbid)
        turbidity_level    : "Clear" | "Moderate" | "High" | "Severe"
        haze_density       : mean dark channel value
    """
    dark = _fast_dark_channel(img, patch_size=15)
    haze_density = float(dark.mean())

    if haze_density < 0.12:
        level = "Clear"
    elif haze_density < 0.25:
        level = "Moderate"
    elif haze_density < 0.45:
        level = "High"
    else:
        level = "Severe"

    return {
        "turbidity_index": round(haze_density, 4),
        "turbidity_level": level,
        "haze_density":    round(haze_density, 4),
    }


def estimate_visibility(img: np.ndarray) -> dict:
    """
    Estimate visibility range in metres using the image attenuation model.
    This is an approximation — real sonar/lidar gives ground truth.

    Model:
        contrast_loss = 1 - local_contrast_mean
        visibility_m  ≈ (1 - contrast_loss) / attenuation_coeff × depth_scale

    Returns:
        visibility_range_meters : estimated visibility (int, metres)
        contrast_loss           : normalised contrast reduction [0..1]
    """
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

    # Michelson contrast in 32×32 blocks
    h, w = gray.shape
    bh, bw = max(1, h//8), max(1, w//8)
    contrasts = []
    for i in range(8):
        for j in range(8):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            mn, mx = block.min(), block.max()
            if mn + mx > 1e-5:
                contrasts.append((mx - mn) / (mx + mn))

    mean_contrast  = float(np.mean(contrasts)) if contrasts else 0.0
    contrast_loss  = max(0.0, 1.0 - mean_contrast)

    # Rough Beer-Lambert attenuation — empirical calibration for coastal water
    # Assumes light absorption coefficient ~0.12 m⁻¹ in clear conditions
    attenuation_base = 0.12
    turbidity_factor = 1.0 + contrast_loss * 4.0
    attenuation      = attenuation_base * turbidity_factor

    visibility_m = max(1, int((mean_contrast / (attenuation + 1e-6)) * 5))
    visibility_m = min(visibility_m, 40)     # cap at realistic 40 m

    return {
        "visibility_range_meters": visibility_m,
        "contrast_loss":           round(contrast_loss, 4),
        "mean_contrast":           round(mean_contrast, 4),
    }


def estimate_color_attenuation(img: np.ndarray) -> dict:
    """
    Estimate per-channel colour attenuation relative to blue.
    Underwater: red attenuates first, then green, blue travels furthest.

    Returns attenuation values [0..1]: 0 = full transmission, 1 = full loss.
    """
    r_mean = float(img[:,:,0].mean())
    g_mean = float(img[:,:,1].mean())
    b_mean = float(img[:,:,2].mean())

    ref  = max(r_mean, g_mean, b_mean, 1e-5)
    return {
        "red":   round(1.0 - r_mean / ref, 4),
        "green": round(1.0 - g_mean / ref, 4),
        "blue":  round(1.0 - b_mean / ref, 4),
    }


def analyze_water_quality(img_np: np.ndarray) -> dict:
    """
    Full water quality analysis pipeline.

    Parameters
    ----------
    img_np : HxWx3 uint8 numpy array

    Returns
    -------
    dict:
        visibility_range_meters, turbidity_level, turbidity_index,
        contrast_loss, mean_contrast,
        attenuation: {red, green, blue},
        water_type : "Coastal" | "Offshore" | "Harbour"
    """
    img_f = img_np.astype(np.float32) / 255.0

    turb = estimate_turbidity(img_f)
    vis  = estimate_visibility(img_f)
    att  = estimate_color_attenuation(img_f)

    # Heuristic water type classification
    v = vis["visibility_range_meters"]
    if v < 5:
        water_type = "Harbour"
    elif v < 15:
        water_type = "Coastal"
    else:
        water_type = "Offshore"

    return {
        "visibility_range_meters": vis["visibility_range_meters"],
        "contrast_loss":           vis["contrast_loss"],
        "mean_contrast":           vis["mean_contrast"],
        "turbidity_level":         turb["turbidity_level"],
        "turbidity_index":         turb["turbidity_index"],
        "haze_density":            turb["haze_density"],
        "attenuation":             att,
        "water_type":              water_type,
    }