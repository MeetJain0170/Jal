"""
utils.py — JalDrishti utility library.

Metrics (numpy-based, for API / eval scripts):
  calculate_psnr_np, calculate_ssim_np
  calculate_uiqm     — Underwater Image Quality Measure (Panetta 2016)
  calculate_uciqe    — Underwater Color Image Quality Evaluation (Yang 2015)
  calculate_edge_score — Edge Preservation Score (Sobel magnitude ratio)

Metrics (torch-based, for training):
  psnr, ssim, evaluate_batch, evaluate_batch_full

Training helpers:
  visualize_results, plot_training_curves
  save_checkpoint, load_checkpoint
  MetricsTracker, AverageMeter, Timer
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Numpy-based quality metrics (used by enhance.py / api.py / train.py)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_uiqm(img: np.ndarray) -> float:
    """
    Underwater Image Quality Measure (UIQM).
    Panetta et al., 2016. Higher is better (typically -2 to +3).

    img : HxWx3 float32 array in [0, 1]
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # ── UICM: chroma measure ─────────────────────────────────────────────────
    rg = r - g
    yb = 0.5 * (r + g) - b
    for ch in (rg, yb):
        ch_flat = ch.flatten()
        n = len(ch_flat)
        # Asymmetric α-trimmed mean
        k1, k2 = int(0.0001 * n), int(0.9999 * n)
        sorted_ch = np.sort(ch_flat)
        trimmed = sorted_ch[k1:k2] if k2 > k1 else sorted_ch
        mu = trimmed.mean() if len(trimmed) else 0.0
    mu_rg, mu_yb = rg.mean(), yb.mean()
    sig_rg, sig_yb = rg.std(), yb.std()
    uicm = -0.0268 * math.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * math.sqrt(sig_rg ** 2 + sig_yb ** 2)

    # ── UISM: sharpness measure ───────────────────────────────────────────────
    def _sobel_mag(channel: np.ndarray) -> np.ndarray:
        from scipy.ndimage import sobel as sp_sobel
        sx = sp_sobel(channel, axis=1)
        sy = sp_sobel(channel, axis=0)
        return np.hypot(sx, sy)

    try:
        uism = 0.0
        for ch, w in zip([r, g, b], [0.299, 0.587, 0.114]):
            mag = _sobel_mag(ch)
            # EME: Enhancement Measure Estimation (max/min in 8×8 blocks)
            h, ww = mag.shape
            bh, bw = max(1, h // 8), max(1, ww // 8)
            eme = 0.0
            cnt = 0
            for bi in range(8):
                for bj in range(8):
                    block = mag[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw]
                    mn, mx = block.min(), block.max()
                    if mn > 1e-5:
                        eme += math.log(mx / mn)
                    cnt += 1
            uism += w * (eme / max(cnt, 1))
    except ImportError:
        # scipy not available — fall back to simple variance sharpness
        uism = float(np.var(g) * 10)

    # ── UIConM: contrast measure ──────────────────────────────────────────────
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    uiconm = float(gray.std())

    # ── Final UIQM ────────────────────────────────────────────────────────────
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * uicm + c2 * uism + c3 * uiconm)


def calculate_uciqe(img: np.ndarray) -> float:
    """
    Underwater Color Image Quality Evaluation (UCIQE).
    Yang et al., 2015. Higher is better (range roughly 0..1).

    img : HxWx3 float32 array in [0, 1]
    """
    import colorsys

    h, w, _ = img.shape
    sat_vals = np.zeros(h * w)
    lum_vals = np.zeros(h * w)

    r_flat = img[:, :, 0].flatten()
    g_flat = img[:, :, 1].flatten()
    b_flat = img[:, :, 2].flatten()

    for i in range(len(r_flat)):
        hue, sat, lum = colorsys.rgb_to_hls(
            float(r_flat[i]), float(g_flat[i]), float(b_flat[i])
        )
        sat_vals[i] = sat
        lum_vals[i] = lum

    sigma_c = float(sat_vals.std())
    con_l   = float(lum_vals.max() - lum_vals.min() + 1e-8)
    mu_s    = float(sat_vals.mean())

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * sigma_c + c2 * con_l + c3 * mu_s)


def calculate_edge_score(enhanced: np.ndarray, original: np.ndarray) -> float:
    """
    Edge Preservation Score — ratio of Sobel edge energy in enhanced vs original.
    Values > 1 mean edge amplification; 1.0 is perfect preservation.
    Range is [0, inf]; values around 0.9-1.2 are typical for good enhancement.

    Both arrays: HxWx3 float32 in [0, 1].
    """
    def _edge_energy(img: np.ndarray) -> float:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        gy = np.diff(gray, axis=0)
        gx = np.diff(gray, axis=1)
        return float(np.mean(gy ** 2) + np.mean(gx ** 2))

    e_enh  = _edge_energy(enhanced)
    e_orig = _edge_energy(original)
    return e_enh / max(e_orig, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Torch-based metrics (training loop)
# ─────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    with torch.no_grad():
        mse = F.mse_loss(pred, target, reduction="none")
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        psnr_vals = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))
    return psnr_vals.mean().item()


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


def ssim(
    pred: torch.Tensor, target: torch.Tensor,
    window_size: int = 11, sigma: float = 1.5,
    C1: float = 1e-4, C2: float = 9e-4,
) -> float:
    with torch.no_grad():
        B, C, H, W = pred.shape
        kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
        kernel = kernel.expand(C, 1, window_size, window_size)
        pad    = window_size // 2
        mu1 = F.conv2d(pred,   kernel, padding=pad, groups=C)
        mu2 = F.conv2d(target, kernel, padding=pad, groups=C)
        s1sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu1 ** 2
        s2sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2 ** 2
        s12  = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu1 * mu2
        num  = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den  = (mu1 ** 2 + mu2 ** 2 + C1) * (s1sq + s2sq + C2)
        return (num / (den + 1e-10)).mean().item()


def _sobel_energy_torch(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    B, C, H, W = x.shape
    xf = x.reshape(B*C, 1, H, W)
    gx = F.conv2d(xf, kx, padding=1)
    gy = F.conv2d(xf, ky, padding=1)
    return (gx**2 + gy**2 + 1e-8).sqrt().mean()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Return (PSNR, SSIM) for a batch."""
    return psnr(pred, target), ssim(pred, target)


def evaluate_batch_full(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    """Return (PSNR, SSIM, EPS) for a batch."""
    eps = float(_sobel_energy_torch(pred) / (_sobel_energy_torch(target) + 1e-8))
    return psnr(pred, target), ssim(pred, target), eps


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(t: torch.Tensor) -> np.ndarray:
    return np.clip(t.detach().cpu().permute(1,2,0).numpy() * 255, 0, 255).astype(np.uint8)


def visualize_results(
    raw: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    save_path: str,
    metrics: Optional[dict] = None,
    epoch: Optional[int] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    title = f"Epoch {epoch}  —  Raw | Enhanced | Ground-Truth" if epoch else \
            "Raw | Enhanced | Ground-Truth"
    if metrics:
        parts = [f"{k}={v:.3f}" for k, v in metrics.items()]
        title += "\n" + "  ".join(parts)
    fig.suptitle(title, fontsize=9)
    for ax, img, lbl in zip(axes, [raw, pred, target], ["Raw", "Enhanced", "Target"]):
        ax.imshow(_to_np(img)); ax.set_title(lbl, fontsize=9); ax.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_training_curves(
    train_losses, val_losses, psnr_history, ssim_history,
    edge_history=None, uiqm_history=None, save_path: str = "curves.png",
) -> None:
    n_plots = 2 + bool(edge_history) + bool(uiqm_history)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train'); axes[0].plot(epochs, val_losses, label='Val')
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, psnr_history, color='green'); axes[1].plot(epochs, ssim_history, color='orange')
    axes[1].set_title("PSNR (green) / SSIM (orange)"); axes[1].grid(True)
    idx = 2
    if edge_history:
        axes[idx].plot(epochs[:len(edge_history)], edge_history, color='purple')
        axes[idx].set_title("Edge Preservation Score"); axes[idx].grid(True); idx += 1
    if uiqm_history:
        axes[idx].plot(range(1, len(uiqm_history)+1), uiqm_history, color='teal')
        axes[idx].set_title("UIQM"); axes[idx].grid(True)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, path, scheduler=None, extra=None):
    payload = {
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "val_loss":         val_loss,
    }
    if scheduler:
        payload["sched_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path, optimizer=None, scheduler=None, device=None):
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optim_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state_dict"])
    if scheduler and "sched_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state_dict"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# MetricsTracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self, select_by: str = "psnr"):
        self.select_by   = select_by
        self.train_losses: list = []
        self.val_losses:   list = []
        self.psnr_history: list = []
        self.ssim_history: list = []
        self.edge_history: list = []
        self.uiqm_history: list = []
        self.best_psnr  = -float("inf")
        self.best_ssim  = -float("inf")
        self.best_edge  = -float("inf")
        self.best_uiqm  = -float("inf")
        self.best_loss  =  float("inf")
        self.best_epoch = 0

    def update(self, train_loss, val_loss, val_psnr, val_ssim, epoch,
               edge_score=None, uiqm_val=None) -> bool:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.psnr_history.append(val_psnr)
        self.ssim_history.append(val_ssim)
        if edge_score is not None: self.edge_history.append(edge_score)
        if uiqm_val   is not None: self.uiqm_history.append(uiqm_val)

        is_best = False
        if self.select_by == "psnr" and val_psnr > self.best_psnr:
            self.best_psnr = val_psnr; is_best = True
        elif self.select_by == "ssim" and val_ssim > self.best_ssim:
            self.best_ssim = val_ssim; is_best = True
        elif self.select_by == "loss" and val_loss < self.best_loss:
            self.best_loss = val_loss; is_best = True

        if is_best:
            self.best_epoch = epoch
            self.best_ssim  = max(self.best_ssim, val_ssim)
            self.best_psnr  = max(self.best_psnr, val_psnr)
            if edge_score is not None: self.best_edge = max(self.best_edge, edge_score)
            if uiqm_val   is not None: self.best_uiqm = max(self.best_uiqm, uiqm_val)
        return is_best

    def get_summary(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_psnr":  self.best_psnr,
            "best_ssim":  self.best_ssim,
            "best_edge":  self.best_edge,
            "best_uiqm":  self.best_uiqm,
        }


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.sum = self.count = self.avg = 0.0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


class Timer:
    def __init__(self): self._t = time.time()
    def elapsed(self) -> float: return time.time() - self._t
    def reset(self): self._t = time.time()
    def __str__(self):
        e = self.elapsed(); m, s = divmod(int(e), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"