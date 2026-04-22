import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import config

class SobelEdgeLoss(nn.Module):
    """
    Calculates the L1 loss between the edge maps of prediction and target.
    Crucial for Diver and Mine detection (Shape preservation).
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        device = pred.device
        b, c, h, w = pred.shape
        
        # Expand kernels to match channel count
        kx = self.kernel_x.expand(c, 1, 3, 3).to(device)
        ky = self.kernel_y.expand(c, 1, 3, 3).to(device)

        # Compute gradients
        pred_dx = F.conv2d(pred, kx, padding=1, groups=c)
        pred_dy = F.conv2d(pred, ky, padding=1, groups=c)
        target_dx = F.conv2d(target, kx, padding=1, groups=c)
        target_dy = F.conv2d(target, ky, padding=1, groups=c)

        # Magnitude
        pred_mag = torch.sqrt(pred_dx**2 + pred_dy**2 + 1e-6)
        target_mag = torch.sqrt(target_dx**2 + target_dy**2 + 1e-6)

        return F.l1_loss(pred_mag, target_mag)

class VGGPerceptualLoss(nn.Module):
    """
    Uses VGG16 features to measure 'Perceptual' difference.
    Prevents the 'blur' effect common in simple MSE loss.
    """
    def __init__(self):
        super().__init__()
        try:
            weights = VGG16_Weights.DEFAULT
            vgg = vgg16(weights=weights).features
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.active = True
        except Exception as e:
            print(f"Warning: VGG not available. Perceptual loss disabled. Error: {e}")
            self.active = False

    def forward(self, pred, target):
        if not self.active: return torch.tensor(0.0, device=pred.device)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for block in self.blocks:
            pred, target = block(pred), block(target)
            loss += F.l1_loss(pred, target)
        return loss

class SSIMLoss(nn.Module):
    """Structural Similarity Loss"""
    def __init__(self):
        super().__init__()

    def _ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        ssim_n = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
        ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        return (ssim_n / ssim_d).mean()

    def forward(self, pred, target):
        return 1.0 - self._ssim(pred, target)

class TotalVariationLoss(nn.Module):
    """
    Prevents digital hallucination and high-frequency 'crunchy' static.
    Forces spatial smoothness across the topological water column.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred):
        tv_h = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        return tv_h + tv_w

class PhysicsDCPLoss(nn.Module):
    """
    Physics-Informed Constraint based on the Dark Channel Prior (DCP).
    Forbids the network from violating global underwater light absorption laws 
    (stops the 'magenta bleed' hallucination).
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=15, stride=1, padding=7)

    def dark_channel(self, img):
        # The Dark Channel is the minimum pixel value across RGB, heavily pooled.
        # Computing Min Pool is geometrically identical to negative Max Pool of the negative image.
        dc = -self.pool(-torch.min(img, dim=1, keepdim=True)[0])
        return dc

    def forward(self, pred, target):
        return F.l1_loss(self.dark_channel(pred), self.dark_channel(target))


class ChromaPreservationLoss(nn.Module):
    """
    Enforces color channel fidelity in LAB colorspace.
    Prevents the network from learning to drain chroma (which causes the
    monochrome / grayscale 'cinematic overdark' failure mode seen in inference).
    Operates on the A and B channels (color axes) while ignoring L (luminance).
    """
    def __init__(self):
        super().__init__()
        # LAB conversion constants (sRGB D65)
        self.register_buffer("rgb_to_xyz", torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32))

    def _rgb_to_lab_approx(self, x):
        """Fast approximate RGB -> LAB via XYZ intermediate. Input in [0,1]."""
        # Approximate gamma linearization
        x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
        # XYZ
        r = self.rgb_to_xyz.to(x.device)
        b, c, h, w = x.shape
        x_flat = x.view(b, 3, -1)                        # (B, 3, HW)
        xyz = torch.einsum('ij,bjk->bik', r, x_flat)     # (B, 3, HW)
        xyz = xyz.view(b, 3, h, w)
        # Normalize by D65 white point
        xyz[:, 0] /= 0.95047
        xyz[:, 2] /= 1.08883
        # f(t) function
        delta = 6.0 / 29.0
        f = torch.where(xyz > delta**3,
                        xyz.clamp(min=1e-8) ** (1/3),
                        xyz / (3 * delta**2) + 4/29)
        L = 116 * f[:, 1] - 16
        A = 500 * (f[:, 0] - f[:, 1])
        B = 200 * (f[:, 1] - f[:, 2])
        return A, B

    def forward(self, pred, target):
        pred_a, pred_b = self._rgb_to_lab_approx(pred)
        tgt_a, tgt_b = self._rgb_to_lab_approx(target)
        return F.l1_loss(pred_a, tgt_a) + F.l1_loss(pred_b, tgt_b)

class SecurityLoss(nn.Module):
    """
    The True Physics-Informed Master Loss Function (PINN).
    Calculates: L1 + SSIM + Perceptual + Edge + TV + Physics + Chroma
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.vgg = VGGPerceptualLoss()
        self.edge = SobelEdgeLoss()
        self.tv = TotalVariationLoss()
        self.physics = PhysicsDCPLoss()
        self.chroma = ChromaPreservationLoss()  # NEW: prevents color drain
        
        self.weights = {
            'l1': config.LOSS_W_L1,
            'ssim': config.LOSS_W_SSIM,
            'perceptual': config.LOSS_W_PERC,
            'edge': config.LOSS_W_EDGE,
            'tv': getattr(config, 'LOSS_W_TV', 0.08),
            'physics': getattr(config, 'LOSS_W_PHYSICS', 0.12),
            'chroma': getattr(config, 'LOSS_W_CHROMA', 0.08),  # NEW
            'multiscale': [config.LOSS_SCALE_FULL, config.LOSS_SCALE_HALF, config.LOSS_SCALE_QUARTER]
        }

    def calculate_single_scale(self, pred, target):
        l_l1 = self.l1(pred, target)
        l_ssim = self.ssim(pred, target)
        l_perc = self.vgg(pred, target)
        l_edge = self.edge(pred, target)
        l_tv = self.tv(pred)
        l_phys = self.physics(pred, target)
        l_chroma = self.chroma(pred, target)  # NEW
        
        total = (self.weights['l1'] * l_l1 +
                 self.weights['ssim'] * l_ssim +
                 self.weights['perceptual'] * l_perc +
                 self.weights['edge'] * l_edge +
                 self.weights['tv'] * l_tv +
                 self.weights['physics'] * l_phys +
                 self.weights['chroma'] * l_chroma)  # NEW
        
        return total

    def forward(self, preds, target):
        # Handle Deep Supervision (Training Mode returns list)
        if isinstance(preds, list):
            scale_weights = self.weights['multiscale']
            loss = 0
            
            # Full scale (Original size)
            loss += scale_weights[0] * self.calculate_single_scale(preds[0], target)
            
            # Half scale (Downsampled target)
            target_half = F.interpolate(target, scale_factor=0.5, mode='bilinear')
            loss += scale_weights[1] * self.calculate_single_scale(preds[1], target_half)
            
            # Quarter scale (Downsampled target)
            target_quarter = F.interpolate(target, scale_factor=0.25, mode='bilinear')
            loss += scale_weights[2] * self.calculate_single_scale(preds[2], target_quarter)
            
            return loss, {}
            
        # Handle Validation/Inference (Single tensor)
        else:
            return self.calculate_single_scale(preds, target), {}