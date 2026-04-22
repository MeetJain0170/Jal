from __future__ import annotations
import numpy as np
import torch
import io
import base64
import os
from PIL import Image
import config

_midas_model = None
_midas_transform = None


def _load_model():
    global _midas_model, _midas_transform

    if _midas_model is not None:
        return _midas_model, _midas_transform

    repo_or_name = "intel-isl/MiDaS"
    hub_kwargs = {"trust_repo": True}
    local_repo = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    if os.path.isdir(local_repo):
        repo_or_name = local_repo
        hub_kwargs["source"] = "local"

    model = torch.hub.load(repo_or_name, config.MIDAS_MODEL_TYPE, **hub_kwargs)
    transforms = torch.hub.load(repo_or_name, "transforms", **hub_kwargs)

    transform = transforms.small_transform if "small" in config.MIDAS_MODEL_TYPE else transforms.dpt_transform

    model.eval()

    _midas_model = model
    _midas_transform = transform

    return _midas_model, _midas_transform


def estimate_depth_tensor(img_np):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform = _load_model()
    model = model.to(device)

    input_tensor = transform(img_np).to(device)

    with torch.no_grad():

        prediction = model(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    return depth


def estimate_depth(img_np):

    try:

        depth = estimate_depth_tensor(img_np)

        depth_u8 = (depth * 255).astype(np.uint8)

        depth_img = Image.fromarray(depth_u8)

        buf = io.BytesIO()
        depth_img.save(buf, format="PNG")

        depth_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "status": "ok",
            "depth_map": depth_b64,
            "average_depth": float(depth.mean())
        }

    except Exception as e:

        return {
            "status": "error",
            "error": str(e)
        }