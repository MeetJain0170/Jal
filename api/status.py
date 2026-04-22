#status.py
from http.server import BaseHTTPRequestHandler
import json

import config
from model_loader import load_model


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        model = load_model()
        body = json.dumps(
            {
                "model_loaded": model is not None,
                "device": str(config.DEVICE),
            }
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

