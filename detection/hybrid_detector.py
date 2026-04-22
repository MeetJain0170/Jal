from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _draw(img_np: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    if img_np.ndim == 2:
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.shape[2] == 4:
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    im = Image.fromarray(img_np)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    colors = {
        "Marine Life": (52, 211, 153),
        "Diver": (251, 191, 36),
        "Security Threat": (239, 68, 68),
        "Surface Vessel": (96, 165, 250),
        "Object": (156, 163, 175),
    }

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        c = colors.get(d.get("category", "Object"), (156, 163, 175))
        dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
        label = f"{d.get('display_class', d.get('class', 'Object'))} {int(float(d.get('confidence', 0))*100)}%"
        dr.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 7 + 6, y1], fill=c)
        dr.text((x1 + 3, max(0, y1 - 14)), label, fill=(0, 0, 0), font=font)

    return np.array(im)


class HybridDetector:
    """
    Combine:
    - custom marine model (fish/trash/etc. from the pipeline training)
    - pretrained diver model (COCO person → Diver) to keep legacy functionality
    """

    def __init__(
        self,
        marine_detector,
        diver_detector,
        shark_detector=None,
        fish_detector=None,
    ):
        self.marine_detector = marine_detector
        self.diver_detector = diver_detector
        self.shark_detector = shark_detector
        self.fish_detector = fish_detector

    def detect(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
        base_marine: List[Dict[str, Any]] = []

        try:
            base_marine = self.marine_detector.detect(img_np)
            for d in base_marine:
                d["source"] = "marine_base"
            dets.extend(base_marine)
        except Exception as e:
            logger.warning("Marine detector failed: %s", e)

        try:
            diver_dets = self.diver_detector.detect(img_np)
            # Keep diver model as a strict diver source to avoid class bleed
            # (e.g. fish -> shark/equipment labels from generic COCO mappings).
            diver_dets = [
                d for d in diver_dets
                if (d.get("display_class") == "Diver" or d.get("category") == "Diver")
                and float(d.get("confidence", 0.0)) >= 0.15
            ]
            for d in diver_dets:
                d["source"] = "diver"
            dets.extend(diver_dets)
        except Exception as e:
            logger.warning("Diver detector failed: %s", e)

        # Specialist shark-only pass
        if self.shark_detector is not None:
            try:
                shark_dets = self.shark_detector.detect(img_np)
                shark_dets = [
                    d for d in shark_dets
                    if (d.get("display_class") or "").lower() in {"shark", "reef shark", "hammerhead shark", "great white shark", "whale shark"}
                    and float(d.get("confidence", 0.0)) >= 0.15
                ]
                marine_boxes = [
                    m.get("bbox", [0, 0, 0, 0]) for m in base_marine
                    if m.get("category") == "Marine Life"
                ]
                for d in shark_dets:
                    conf = float(d.get("confidence", 0.0))
                    box = d.get("bbox", [0, 0, 0, 0])
                    overlaps_base = any(self._iou(box, mb) > 0.10 for mb in marine_boxes)
                    if marine_boxes and not overlaps_base and conf < 0.58:
                        continue
                    d["display_class"] = "Shark"
                    d["class"] = "shark"
                    d["category"] = "Marine Life"
                    d["source"] = "shark_specialist"
                dets.extend(shark_dets)
            except Exception as e:
                logger.warning("Shark detector failed: %s", e)

        # Specialist fish-only pass (normal fish-like objects)
        if self.fish_detector is not None:
            try:
                fish_dets = self.fish_detector.detect(img_np)
                fish_dets = [
                    d for d in fish_dets
                    if (d.get("display_class") or "").lower() in {"fish", "tropical fish", "marine animal", "animal other", "eel"}
                    and float(d.get("confidence", 0.0)) >= 0.15
                ]
                marine_boxes = [
                    m.get("bbox", [0, 0, 0, 0]) for m in base_marine
                    if m.get("category") == "Marine Life"
                ]
                for d in fish_dets:
                    conf = float(d.get("confidence", 0.0))
                    box = d.get("bbox", [0, 0, 0, 0])
                    overlaps_base = any(self._iou(box, mb) > 0.12 for mb in marine_boxes)
                    if marine_boxes and not overlaps_base and conf < 0.40:
                        continue
                    d["display_class"] = "Fish"
                    d["class"] = "fish"
                    d["category"] = "Marine Life"
                    d["source"] = "fish_specialist"
                dets.extend(fish_dets)
            except Exception as e:
                logger.warning("Fish detector failed: %s", e)

        dets = self._cleanup_diver_conflicts(dets)
        dets = self._rescue_additional_diver(dets)
        dets = self._quality_filter(dets)
        dets.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return dets

    @staticmethod
    def _iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter / (a1 + a2 - inter + 1e-6)

    def _cleanup_diver_conflicts(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        If we have a diver detection, suppress marine/object boxes that overlap
        heavily with the diver body to avoid "fish on diver suit" artifacts.
        """
        diver_boxes = [
            d for d in dets
            if d.get("display_class") == "Diver" or d.get("category") == "Diver"
        ]
        frame_w = max([float(d.get("bbox", [0, 0, 0, 0])[2]) for d in dets] + [1.0])
        frame_h = max([float(d.get("bbox", [0, 0, 0, 0])[3]) for d in dets] + [1.0])
        frame_area = max(1.0, frame_w * frame_h)
        if not diver_boxes:
            # Disabled fallback relabel to avoid fish scenes becoming "Diver".
            return dets

        cleaned: List[Dict[str, Any]] = []
        for d in dets:
            if d in diver_boxes:
                cleaned.append(d)
                continue

            drop = False
            for dv in diver_boxes:
                db = d.get("bbox", [0, 0, 0, 0])
                vb = dv.get("bbox", [0, 0, 0, 0])
                ov = self._iou(db, vb)
                # Drop marine/object fragments intersecting diver body.
                dx1, dy1, dx2, dy2 = db
                d_area = max(0.0, (dx2 - dx1) * (dy2 - dy1))
                d_area_ratio = d_area / frame_area
                cx, cy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                inside_diver = (vb[0] <= cx <= vb[2]) and (vb[1] <= cy <= vb[3])
                is_marineish = (d.get("category") in {"Marine Life", "Object"}) or (
                    (d.get("display_class") or "").lower() in {"fish", "eel", "other", "marine animal"}
                )
                marine_conf = float(d.get("confidence", 0.0))
                # Only suppress marine boxes that strongly overlap diver body and
                # are likely suit fragments; keep valid fish near a diver.
                if (ov > 0.55 and is_marineish and d_area_ratio < 0.08) or (
                    inside_diver and is_marineish and marine_conf < 0.22 and d_area_ratio < 0.03
                ):
                    drop = True
                    break
            if not drop:
                cleaned.append(d)
        return cleaned

    def _rescue_additional_diver(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        If only one diver is detected, promote one additional person-like box
        that is large enough and spatially separate.
        """
        # Disabled: this heuristic can relabel marine fauna as Diver and pollute
        # scenes where no person is present.
        return dets

    def _quality_filter(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep only high-confidence, non-noisy detections for cleaner overlays.
        """
        if not dets:
            return dets

        diver = [d for d in dets if d.get("display_class") == "Diver" or d.get("category") == "Diver"]
        others = [d for d in dets if d not in diver]
        frame_w = max([float(d.get("bbox", [0, 0, 0, 0])[2]) for d in dets] + [1.0])
        frame_h = max([float(d.get("bbox", [0, 0, 0, 0])[3]) for d in dets] + [1.0])
        frame_area = max(1.0, frame_w * frame_h)

        # Keep up to two strong, non-overlapping divers.
        if diver:
            diver.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
            keep: List[Dict[str, Any]] = []
            for d in diver:
                conf = float(d.get("confidence", 0.0))
                if conf < 0.50:
                    continue
                db = d.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = db
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                area_ratio = (bw * bh) / frame_area
                aspect = bw / max(bh, 1e-6)
                marine_overlap = 0.0
                for od in others:
                    if od.get("category") != "Marine Life":
                        continue
                    marine_overlap = max(marine_overlap, self._iou(db, od.get("bbox", [0, 0, 0, 0])))
                # Guard against statue/structure being mislabeled as a huge diver.
                if area_ratio > 0.14 and conf < 0.65:
                    continue
                # Reject tiny diver boxes and very wide fish-like boxes.
                if area_ratio < 0.012:
                    continue
                if aspect < 0.22 or aspect > 1.35:
                    continue
                if marine_overlap > 0.20 and area_ratio > 0.09 and conf < 0.78:
                    continue
                if any(self._iou(db, k.get("bbox", [0, 0, 0, 0])) > 0.35 for k in keep):
                    continue
                keep.append(d)
                if len(keep) >= 2:
                    break
            if not keep:
                keep = [diver[0]]

            # Keep more non-diver objects (especially marine life) while still
            # suppressing tiny noise and diver-overlapping fragments.
            marine_candidates = []
            misc_candidates = []
            for d in others:
                conf = float(d.get("confidence", 0.0))
                x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                area = max(0.0, (x2 - x1) * (y2 - y1))
                area_ratio = area / frame_area
                label = (d.get("display_class") or "").lower()
                is_marineish = (d.get("category") == "Marine Life") or (
                    label in {"fish", "eel", "ray", "shark", "marine animal", "animal other"}
                )
                min_conf = 0.15 if is_marineish else 0.15
                min_area = 450 if is_marineish else 1800
                if conf < min_conf:
                    continue
                if area < min_area:
                    continue
                if is_marineish and area_ratio > 0.30 and conf < 0.78:
                    continue
                if any(self._iou(d.get("bbox", [0, 0, 0, 0]), dv.get("bbox", [0, 0, 0, 0])) > 0.12 for dv in keep):
                    continue
                if is_marineish:
                    marine_candidates.append(d)
                else:
                    # Keep miscellaneous objects conservative to reduce clutter.
                    if conf >= 0.22:
                        misc_candidates.append(d)
            marine_candidates.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
            misc_candidates.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
            keep.extend(marine_candidates[:20])
            keep.extend(misc_candidates[:3])
            return keep

        # No diver found: keep more confident detections for dense fish scenes.
        strong = []
        for d in dets:
            conf = float(d.get("confidence", 0.0))
            label = (d.get("display_class") or "").lower()
            is_marineish = (d.get("category") == "Marine Life") or (
                label in {"fish", "eel", "ray", "shark", "marine animal", "animal other"}
            )
            min_conf = 0.15 if is_marineish else 0.15
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
            area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio = area / frame_area
            min_area = 400 if is_marineish else 1800
            if area < min_area:  # reject tiny corner noise boxes
                continue
            if is_marineish and area_ratio > 0.32 and conf < 0.80:
                continue
            strong.append(d)
        strong.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return strong[:30] if strong else dets[:1]

    def detect_and_annotate(
        self,
        inference_img: np.ndarray,
        original_img: Optional[np.ndarray] = None,
        context: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        dets = self.detect(inference_img)
        canvas = original_img if original_img is not None else inference_img
        ann = _draw(canvas, dets)
        pil = Image.fromarray(ann)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return dets, f"data:image/png;base64,{b64}"


_detector: Optional[HybridDetector] = None
_sig: Optional[tuple] = None


def get_detector(
    *,
    marine_weights: str,
    marine_conf: float,
    marine_iou: float,
    marine_imgsz: int,
    diver_weights: str,
    diver_conf: float,
    diver_iou: float,
    diver_imgsz: int,
    shark_weights: Optional[str] = None,
    shark_conf: float = 0.16,
    shark_iou: float = 0.45,
    shark_imgsz: int = 1280,
    fish_weights: Optional[str] = None,
    fish_conf: float = 0.10,
    fish_iou: float = 0.45,
    fish_imgsz: int = 1280,
) -> HybridDetector:
    global _detector, _sig
    sig = (
        marine_weights, marine_conf, marine_iou, marine_imgsz,
        diver_weights, diver_conf, diver_iou, diver_imgsz,
        shark_weights, shark_conf, shark_iou, shark_imgsz,
        fish_weights, fish_conf, fish_iou, fish_imgsz,
    )
    if _detector is None or _sig != sig:
        from detection.simple_detector import get_detector as get_marine
        from detection.yolo_detector import get_detector as get_diver

        marine = get_marine(
            weights=marine_weights,
            conf_thresh=marine_conf,
            iou_thresh=marine_iou,
            img_size=marine_imgsz,
            tta=False,
            multi_scale=False,
        )

        diver = get_diver(
            weights=diver_weights,
            conf_thresh=diver_conf,
            iou_thresh=diver_iou,
            img_size=diver_imgsz,
            enhance=False,
            tta=False,
            run_world_pass=False,
        )

        shark = None
        if shark_weights:
            shark = get_marine(
                weights=shark_weights,
                conf_thresh=shark_conf,
                iou_thresh=shark_iou,
                img_size=shark_imgsz,
                tta=False,
                multi_scale=False,
            )

        fish = None
        if fish_weights:
            fish = get_marine(
                weights=fish_weights,
                conf_thresh=fish_conf,
                iou_thresh=fish_iou,
                img_size=fish_imgsz,
                tta=False,
                multi_scale=False,
            )

        _detector = HybridDetector(
            marine_detector=marine,
            diver_detector=diver,
            shark_detector=shark,
            fish_detector=fish,
        )
        _sig = sig
    return _detector

