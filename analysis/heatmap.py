"""
analysis/heatmap.py — Enhancement difference heatmap generation.

Computes per-pixel enhancement magnitude and renders it as a
false-colour overlay using OpenCV colour maps.
"""
from __future__ import annotations
import io
import base64
import numpy as np
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def compute_difference_map(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """
    Compute signed L2 difference map.
    Both arrays: HxWx3 uint8.
    Returns: HxWx3 float32, values in [-1, 1].
    """
    diff = enhanced.astype(np.float32) - original.astype(np.float32)
    return diff / 255.0


def compute_magnitude_map(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """
    Per-pixel enhancement magnitude (L2 norm across channels).
    Returns: HxW float32 in [0, 1].
    """
    diff = compute_difference_map(original, enhanced)
    mag  = np.linalg.norm(diff, axis=2)        # (H, W)
    return np.clip(mag / (np.sqrt(3)), 0.0, 1.0)   # normalise to [0,1]


def generate_heatmap(
    original_np: np.ndarray,
    enhanced_np: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "JET",
) -> np.ndarray:
    """
    Generate a false-colour heatmap overlay on top of the enhanced image.

    Parameters
    ----------
    original_np : HxWx3 uint8
    enhanced_np : HxWx3 uint8
    alpha       : blend weight for heatmap overlay (0 = none, 1 = full)
    colormap    : OpenCV colormap name ('JET', 'TURBO', 'HOT', 'MAGMA', …)

    Returns
    -------
    HxWx3 uint8 — heatmap blended onto enhanced image
    """
    mag    = compute_magnitude_map(original_np, enhanced_np)
    mag_u8 = (mag * 255).astype(np.uint8)

    if _CV2_AVAILABLE:
        cmap_id = getattr(cv2, f"COLORMAP_{colormap}", cv2.COLORMAP_JET)
        coloured = cv2.applyColorMap(mag_u8, cmap_id)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
        result   = cv2.addWeighted(enhanced_np, 1.0 - alpha, coloured, alpha, 0)
    else:
        # Fallback: simple red-channel intensity map
        coloured         = np.zeros_like(enhanced_np)
        coloured[:,:,0]  = mag_u8                 # red channel only
        result = np.clip(
            enhanced_np.astype(np.float32) * (1.0 - alpha) +
            coloured.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)

    return result


def heatmap_to_base64(
    original_np: np.ndarray,
    enhanced_np: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "JET",
) -> str:
    """
    Returns a base64-encoded PNG data URL of the heatmap overlay.
    """
    hm  = generate_heatmap(original_np, enhanced_np, alpha, colormap)
    img = Image.fromarray(hm)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def enhancement_stats(original_np: np.ndarray, enhanced_np: np.ndarray) -> dict:
    """
    Summary statistics of the enhancement magnitude map.
    """
    mag = compute_magnitude_map(original_np, enhanced_np)
    return {
        "mean_enhancement":   round(float(mag.mean()), 4),
        "max_enhancement":    round(float(mag.max()),  4),
        "std_enhancement":    round(float(mag.std()),  4),
        "enhanced_pct":       round(float((mag > 0.05).mean() * 100), 1),
    }