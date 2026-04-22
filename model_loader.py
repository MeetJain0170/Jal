from __future__ import annotations

import os
from typing import Optional

import torch

import config
from train.model import UNet

_MODEL: Optional[UNet] = None


def get_checkpoint_path() -> str:
    return os.path.join(config.CHECKPOINT_DIR, "best_model.pth")


def load_model() -> Optional[UNet]:

    global _MODEL

    if _MODEL is not None:
        return _MODEL

    path_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.ckpt")
    path_pth = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    path = path_ckpt if os.path.exists(path_ckpt) else path_pth

    if not os.path.exists(path):
        return None

    try:

        base = getattr(config, "UNET_BASE_FEATURES", 64)

        model = UNet(
            in_channels=3,
            out_channels=3,
            base=base
        ).to(config.DEVICE)

        ckpt = torch.load(path, map_location=config.DEVICE)

        # PyTorch Lightning format extraction
        if "state_dict" in ckpt:
            state = {k.replace('net.', ''): v for k, v in ckpt["state_dict"].items() if k.startswith('net.')}
        # Old Format Extract
        else:
            state = ckpt.get("model_state_dict", ckpt)

        model.load_state_dict(state)

        model.eval()

        _MODEL = model

        return _MODEL

    except Exception as exc:

        print(f"[model_loader] model load error: {exc}")

        return None