"""
analysis/image_quality.py — Standalone maritime image quality metrics.

Exported functions:
    compute_all_metrics(original_np, enhanced_np) -> dict
        Returns: psnr, ssim, uiqm, uciqe, eps

Can be called directly from API handlers or imported in training scripts.
"""
from __future__ import annotations
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity  as _ssim


def compute_psnr(orig: np.ndarray, enh: np.ndarray) -> float:
    """PSNR in dB. Arrays: HxWx3 uint8."""
    return float(_psnr(orig, enh, data_range=255))


def compute_ssim(orig: np.ndarray, enh: np.ndarray) -> float:
    """SSIM. Arrays: HxWx3 uint8."""
    return float(_ssim(orig, enh, channel_axis=2, data_range=255))


def compute_uiqm(img: np.ndarray) -> float:
    """
    Underwater Image Quality Measure (Panetta 2016).
    img : HxWx3 float32 in [0, 1].
    Higher is better; typical range -2 to +3.
    """
    import math
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # UICM — chroma quality
    rg_mean, yb_mean = (r-g).mean(), (0.5*(r+g)-b).mean()
    rg_std,  yb_std  = (r-g).std(),  (0.5*(r+g)-b).std()
    uicm = -0.0268 * math.sqrt(rg_mean**2 + yb_mean**2) + 0.1586 * math.sqrt(rg_std**2 + yb_std**2)

    # UISM — sharpness
    try:
        from scipy.ndimage import sobel
        def _eme(ch):
            sx, sy = sobel(ch, axis=1), sobel(ch, axis=0)
            mag = np.hypot(sx, sy)
            h, w = mag.shape
            bh, bw = max(1, h//8), max(1, w//8)
            val = 0.0; cnt = 0
            for i in range(8):
                for j in range(8):
                    bl = mag[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                    mn, mx = bl.min(), bl.max()
                    if mn > 1e-5: val += math.log(mx / mn)
                    cnt += 1
            return val / max(cnt, 1)
        uism = 0.299*_eme(r) + 0.587*_eme(g) + 0.114*_eme(b)
    except ImportError:
        uism = float(np.var(g) * 10)

    # UIConM — contrast
    gray = 0.299*r + 0.587*g + 0.114*b
    uiconm = float(gray.std())

    return float(0.4680*uicm + 0.2745*uism + 0.2576*uiconm)


def compute_uciqe(img: np.ndarray) -> float:
    """
    Underwater Color Image Quality Evaluation (Yang 2015).
    img : HxWx3 float32 in [0, 1].
    Higher is better; range roughly 0..1.
    """
    import colorsys
    r_f = img[:,:,0].flatten()
    g_f = img[:,:,1].flatten()
    b_f = img[:,:,2].flatten()
    sats = np.zeros(len(r_f)); lums = np.zeros(len(r_f))
    for i in range(len(r_f)):
        _, s, l = colorsys.rgb_to_hls(float(r_f[i]), float(g_f[i]), float(b_f[i]))
        sats[i] = s; lums[i] = l
    return float(0.4680*sats.std() + 0.2745*(lums.max()-lums.min()) + 0.2576*sats.mean())


def compute_eps(enh: np.ndarray, orig: np.ndarray) -> float:
    """
    Edge Preservation Score.
    Both arrays: HxWx3 float32 in [0, 1].
    """
    def _energy(a):
        gray = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
        return float(np.mean(np.diff(gray, axis=0)**2) + np.mean(np.diff(gray, axis=1)**2))
    return _energy(enh) / max(_energy(orig), 1e-8)


def compute_all_metrics(original_np: np.ndarray, enhanced_np: np.ndarray) -> dict:
    """
    Compute all five quality metrics.

    Parameters
    ----------
    original_np : HxWx3 uint8 — original (degraded) image
    enhanced_np : HxWx3 uint8 — enhanced image

    Returns
    -------
    dict with keys: psnr, ssim, uiqm, uciqe, eps
    All values are float or None on failure.
    """
    result = dict(psnr=None, ssim=None, uiqm=None, uciqe=None, eps=None)
    try:
        result["psnr"] = compute_psnr(original_np, enhanced_np)
        result["ssim"] = compute_ssim(original_np, enhanced_np)
        orig_f = original_np.astype(np.float32) / 255.0
        enh_f  = enhanced_np.astype(np.float32) / 255.0
        result["uiqm"]  = compute_uiqm(enh_f)
        result["uciqe"] = compute_uciqe(enh_f)
        result["eps"]   = compute_eps(enh_f, orig_f)
    except Exception as exc:
        import sys
        print(f"[image_quality] metric error: {exc}", file=sys.stderr)
    return result