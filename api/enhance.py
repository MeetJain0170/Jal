from http.server import BaseHTTPRequestHandler
import base64
import io
import json
import cgi

from PIL import Image

import config
from enhance import enhance_with_patches, calculate_metrics_full
from model_loader import load_model


# ─────────────────────────────────────────────
# HTTP Handler
# ─────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        model = load_model()
        if model is None:
            self._send_json(
                400,
                {"error": "No trained model found. Train first on your machine."},
            )
            return

        ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data":
            self._send_json(400, {"error": "Expected multipart/form-data upload"})
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
            keep_blank_values=True,
        )

        if "image" not in form:
            self._send_json(400, {"error": "No image field in form"})
            return

        file_item = form["image"]
        img_bytes = file_item.file.read()

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            self._send_json(400, {"error": "Failed to read image"})
            return

        original_w, original_h = img.size

        # ─────────────────────────────────────────────
        # True Full-Resolution Patch Inference
        # ─────────────────────────────────────────────
        out_img = enhance_with_patches(
            model,
            img,
            patch_size=config.PATCH_SIZE,
            overlap=config.PATCH_OVERLAP,
        )

        metrics = calculate_metrics_full(img, out_img)

        # Encode PNG (lossless)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        self._send_json(
            200,
            {
                "success": True,
                "enhanced_image": f"data:image/png;base64,{img_str}",
                "size": f"{original_w}x{original_h}",
                "device": "GPU" if config.DEVICE.type == "cuda" else "CPU",
                "psnr": metrics.get("psnr"),
                "ssim": metrics.get("ssim"),
                "eps": metrics.get("eps"),
                "uiqm": metrics.get("uiqm"),
                "uciqe": metrics.get("uciqe"),
            },
        )