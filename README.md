# JalDrishti
### Underwater Enhancement & Maritime Scene Intelligence

> End-to-end computer vision system for underwater environments — image enhancement, marine object detection, depth estimation, and water-quality analytics in one integrated pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why Underwater Imaging is Hard](#why-underwater-imaging-is-hard)
- [System Architecture](#system-architecture)
- [Detection Dataset Pipeline](#detection-dataset-pipeline)
- [Unified Class Taxonomy](#unified-class-taxonomy)
- [Training Optimisations](#training-optimisations)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Technology Stack](#technology-stack)
- [Challenges & Future Work](#challenges--future-work)
- [Validation Checklist](#validation-checklist)

---

## Project Overview

JalDrishti is a full-stack computer vision system purpose-built for underwater environments. It addresses the core challenges of underwater imaging — colour cast, haze, backscatter, and low edge contrast — through a hybrid pipeline that combines deep learning, classical signal processing, multi-class object detection, monocular depth estimation, and water-quality analytics, all exposed through an interactive web interface.

The system was designed and engineered end-to-end: from dataset collection and label standardisation across three public underwater datasets, through a custom YOLOv8 training pipeline with per-step scripts, to deployment behind a Flask API with a JavaScript front-end.

| Module | Capability |
|---|---|
| **Enhancement** | U-Net deep model + classical OpenCV polish. Corrects colour cast, dehazes, sharpens, boosts contrast at full resolution. |
| **Detection** | YOLOv8 trained on 3 merged datasets (DeepFish, Fish4Knowledge, TrashCan). 17 marine classes: fish, 9 trash categories, crab, starfish, eel, echinus, holothurian, scallop, animal_other. |
| **Depth** | MiDaS monocular depth estimation. Produces per-pixel relative depth map and distance zone annotations (near / mid / far). |
| **Water Quality** | Visibility and turbidity KPI computation. Detects colour dominance, haze index, and transmission map statistics. |
| **Metrics** | PSNR, SSIM, UIQM (Underwater Image Quality Measure), UCIQE, and Edge Preservation Score. |
| **Web UI** | Interactive upload, before/after comparison, annotated results panel, analytics dashboard. |

---

## Why Underwater Imaging is Hard

Light behaves differently underwater than in air. Four physical effects consistently degrade image quality and make standard computer vision unreliable:

- **Colour cast** — water selectively absorbs red and yellow wavelengths, leaving images with a dominant green or blue shift that throws off colour-based detectors.
- **Haze and backscatter** — suspended particles scatter light back toward the camera, creating a veil of fog that reduces contrast and obscures edges.
- **Non-uniform illumination** — artificial dive lights produce bright centres and dark peripheries; natural light produces caustics and dappled patterns.
- **Low edge visibility** — fine texture, debris, and low-contrast boundaries make segmentation and detection substantially harder than in air.

Standard image enhancement and detection pipelines trained on surface imagery fail under these conditions. JalDrishti is designed specifically to handle them.

---

## System Architecture

```
Frontend (static HTML/CSS/JS)
        │
        ▼
Flask API Layer (api.py)
        │
   ┌────┴─────────────────────────────────┐
   │                                      │
Enhancement              Detection / Depth / Analytics
   │                                      │
U-Net checkpoint    YOLOv8 (marine_detector.pt)
   +                     +
Classical OpenCV    MiDaS Small
polish pipeline     Water Quality
                    Image Metrics
```

### End-to-End Request Flow

A single `POST /api/enhance` triggers this sequence:

1. Frontend encodes the uploaded image as base64 and posts to `/api/enhance`.
2. Flask decodes the payload and fans out to all backend modules in sequence.
3. **Enhancement** — image passes through the U-Net checkpoint, then the classical OpenCV pipeline for full-resolution sharpening and colour correction.
4. **Detection** — the enhanced frame goes to `MarineYOLODetector`. Bounding boxes, class labels, and confidence scores are returned as JSON.
5. **Depth** — MiDaS Small produces a normalised depth map and three zone overlays.
6. **Water Quality** — transmission map statistics, haze index, and colour dominance ratios are computed from raw and enhanced images.
7. **Metrics** — PSNR, SSIM, UIQM, UCIQE, and EPS are computed and validated as finite before being returned.
8. Flask assembles all outputs into a single JSON response. The frontend renders everything.

### Module Map

| Path | Module | Responsibility |
|---|---|---|
| `api.py` | Flask API | Route orchestration, request parsing, response assembly |
| `enhance.py` | Enhancement | Hybrid U-Net + OpenCV enhancement pipeline |
| `detection/yolo_detector.py` | Marine Detector | `MarineYOLODetector` class, 17-class inference, draw and filter utilities |
| `detection/simple_detector.py` | Fallback Detector | Lightweight rule-based fallback if YOLO weights are absent |
| `depth/depth_estimator.py` | Depth | MiDaS Small wrapper, zone annotation, depth map normalisation |
| `analysis/water_quality.py` | Water Quality | Turbidity, visibility, colour cast analysis |
| `analysis/image_quality.py` | Image Metrics | PSNR, SSIM, UIQM, UCIQE, EPS computation |
| `analysis/threat_analysis.py` | Threat Scoring | Scene-level threat summarisation from detection results |
| `model_loader.py` | Model Loader | Checkpoint discovery, U-Net weight loading, fallback resilience |
| `config.py` | Config | Central path and parameter configuration |
| `models/unet.py` | U-Net | U-Net architecture definition for enhancement |
| `train/` | Training Utils | Dataset loader, trainer, evaluator, inference script |
| `static/` | Frontend | HTML/CSS/JS: upload, before/after comparison, dashboard |
| `pipeline/` | Detection Pipeline | 14-step scripted training pipeline (see below) |

---

## Detection Dataset Pipeline

One of the core engineering contributions of JalDrishti is the multi-dataset fusion and training pipeline. Three publicly available underwater datasets were cleaned, class-remapped, merged, and used to train a custom YOLOv8 model. All 14 steps are scripted and fully reproducible.

### Source Datasets

| Dataset | Original Classes | Notes |
|---|---|---|
| DeepFish | 1 (unnamed) | Single unnamed fish class. Large number of paired images, high species and environment diversity. |
| Fish4Knowledge | 7 (0,1,2,5,6,7,8) | Seven fish-species classes — all mapped to a unified `fish` class. No class names in original `data.yaml`. |
| TrashCan | 28 | Richest label set. Includes marine animals, trash categories, plant, and ROV. Two classes dropped as irrelevant. |

### Pipeline Scripts

| Script | What it does |
|---|---|
| `step1_clean_datasets.py` | Removes corrupt images, empty labels, and mismatched image-label pairs across all dataset splits. |
| `step2_class_config.py` | Defines and exports the unified 17-class taxonomy and per-dataset mapping dictionaries. No disk changes — import this in other steps. |
| `step3_convert_labels.py` | Remaps all YOLO `.txt` label files in-place using the mapping config. Backs up originals to `labels_backup/` automatically. |
| `step4_merge_datasets.py` | Copies all images and labels into `data/merged/` with dataset prefix (`df_`, `f4k_`, `tc_`) to prevent filename collisions. |
| `step5_split_dataset.py` | Stratified 80/20 train-val split, proportional per dataset source. Seed-reproducible. |
| `step6_create_yaml.py` | Writes `data/marine_detection.yaml` with absolute paths and the 17-class name list. |
| `step7_train.py` | Trains YOLOv8s (or l/x) on the merged dataset with underwater-tuned augmentation config. |
| `step8_evaluate.py` | Full validation: mAP50, mAP50-95, per-class AP, precision, recall. Flags classes below the warning threshold. |
| `step9_improve_dataset.py` | Class distribution analysis, exact duplicate detection by MD5, YOLO label sanity check, writes auto-report. |
| `step10_retrain.py` | Optimised retrain: FP16 AMP, auto batch sizing, RAM image caching, rect training, `torch.compile`. Up to 10x faster than naive training. |
| `step11_export_model.py` | Copies `best.pt` to `models/detection/`. Optionally exports to ONNX (recommended) and TensorRT. |
| `step12_yolo_detector.py` | `MarineYOLODetector` class — drop-in replacement for `detection/yolo_detector.py`. Detect, batch-detect, draw, filter, summarise. |
| `step13_realworld_test.py` | CLI tester for image folders and video files. Saves annotated output and a per-detection failure log. |
| `step14_run_pipeline.py` | Orchestrator. Run all 14 steps, or `--from N` to resume, or `--steps N M` to run specific steps. |

```bash
# Run the full pipeline
python pipeline/step14_run_pipeline.py

# Resume from step 9 after improving labels
python pipeline/step14_run_pipeline.py --from 9

# Run specific steps only
python pipeline/step14_run_pipeline.py --steps 1 3 6
```

---

## Unified Class Taxonomy

All three datasets were remapped to 17 classes. Design decisions are noted for each.

| ID | Class | Mapping Notes |
|---|---|---|
| 0 | `fish` | DeepFish class 0; all 7 Fish4Knowledge species; TrashCan `animal_fish` |
| 1 | `trash_bag` | TrashCan `trash_bag` |
| 2 | `trash_bottle` | TrashCan `trash_bottle` |
| 3 | `trash_can` | TrashCan `trash_can` |
| 4 | `trash_cup` | TrashCan `trash_cup` |
| 5 | `trash_net` | TrashCan `trash_net` |
| 6 | `trash_rope` | TrashCan `trash_rope` |
| 7 | `trash_pipe` | TrashCan `trash_pipe` |
| 8 | `trash_wreckage` | TrashCan `trash_wreckage` |
| 9 | `trash_other` | Merged: `trash_branch`, `trash_clothing`, `trash_container`, `trash_snack_wrapper`, `trash_tarp`, `trash_unknown_instance` |
| 10 | `crab` | TrashCan `animal_crab` |
| 11 | `starfish` | TrashCan `starfish` + `animal_starfish` merged |
| 12 | `eel` | TrashCan `animal_eel` |
| 13 | `echinus` | TrashCan `echinus` + `seaurchin` merged |
| 14 | `holothurian` | TrashCan `holothurian` + `seacucumber` merged |
| 15 | `scallop` | TrashCan `scallop` |
| 16 | `animal_other` | TrashCan `animal_etc` + `animal_shells` — catch-all for unlabelled marine fauna |

> **Dropped:** TrashCan `plant` (class 0) and `rov` (class 10) — neither constitutes a detection target relevant to marine scene intelligence.

---

## Training Optimisations

`step10_retrain.py` was specifically engineered for fast iterative retraining after dataset improvements. Seven techniques are applied simultaneously:

| Technique | Config | Speedup |
|---|---|---|
| FP16 mixed precision | `amp=True` | 2–3x on modern NVIDIA GPU — halves memory bandwidth |
| Auto batch sizing | `batch=-1` | Fills available VRAM to maximise samples per forward pass |
| RAM image caching | `cache='ram'` | Images decoded once at epoch 0, served from memory thereafter — eliminates disk I/O |
| Rectangle training | `rect=True` | Batches images of similar aspect ratio — minimises padding waste per convolution |
| PyTorch op fusion | `torch.compile(mode='reduce-overhead')` | ~15% additional throughput on PyTorch 2.x |
| No mid-run checkpoints | `save_period=-1` | Removes per-epoch disk stalls; keeps only `best.pt` and `last.pt` |
| Aggressive early stop | `patience=10`, `close_mosaic=5` | Stops sooner on plateau; disables expensive mosaic augmentation in final epochs |

> If RAM is tight, switch `cache='ram'` to `cache='disk'` — still significantly faster than cold disk reads per epoch.

---

## Quick Start

### 1. Clone and pull model artifacts

```bash
git clone https://github.com/MeetJain0170/Jal-prac.git
cd Jal-prac
git lfs install
git lfs pull
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the API server

```bash
python api.py
# Open: http://localhost:5500
```

### 4. (Optional) Run the detection training pipeline

```bash
# Full pipeline from scratch
python pipeline/step14_run_pipeline.py

# Test on real underwater images or video
python pipeline/step13_realworld_test.py --source /path/to/images
python pipeline/step13_realworld_test.py --source /path/to/dive_footage.mp4
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Returns model and module readiness: YOLO weights, U-Net checkpoint, MiDaS, all analysis modules. |
| `POST` | `/api/enhance` | Main endpoint. Accepts base64 image. Returns enhanced image, detection JSON, depth map, water quality KPIs, and image metrics. |
| `POST` | `/api/detect` | Detection only. Returns bounding boxes, class labels, confidence scores for all 17 marine classes. |
| `POST` | `/api/depth` | Depth only. Returns normalised depth map and zone annotation overlay. |
| `POST` | `/api/analyze-water` | Water quality only. Returns turbidity index, visibility score, colour cast statistics. |
| `GET` | `/api/gallery` | Returns list of saved gallery items. |
| `POST` | `/api/gallery/save` | Saves an analysis result to the gallery. |
| `DELETE` | `/api/gallery/clear` | Clears all saved gallery items. |

---

## Technology Stack

| Layer | Libraries / Tools |
|---|---|
| Deep Learning | PyTorch, torchvision, timm, Ultralytics YOLOv8 & YOLO-World, MiDaS |
| Image Processing | OpenCV, Pillow, NumPy, SciPy, scikit-image |
| API Layer | Flask, Python 3.x |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Model Artifacts | Git LFS for checkpoint storage and versioning |
| Training Infra | Ultralytics training loop, FP16 AMP, `torch.compile`, RAM caching, ONNX export |

### Datasets

| Dataset | Purpose |
|---|---|
| UIEB | Paired underwater enhancement data for U-Net training |
| DeepFish | Underwater fish detection — single class |
| Fish4Knowledge | Underwater fish detection — 7 species classes |
| TrashCan | Marine animal + underwater trash detection — 28 classes |

---

## Challenges & Future Work

### Current Challenges

- **Confidence tuning** — a threshold that works in clear water generates false positives in heavy-backscatter scenes and vice versa. A scene-adaptive confidence schedule is needed.
- **Colour cast residual** — the U-Net enhancement reduces but does not fully eliminate green/blue dominance under strong ambient light. The classical post-processing pass partially compensates but the effect is inconsistent across depth ranges.
- **Class confusion at decision boundaries** — shark vs. diver ambiguity in cluttered frames; `echinus` vs. `holothurian` in degraded imagery. Both pairs share similar silhouettes at low resolution.
- **Metric stability** — UIQM and UCIQE can produce non-finite values under extreme transformations (overexposure, near-black frames). Finite-value validation guards are in place but the root cause is upstream in the metric formulations.
- **Class imbalance** — `fish` and `trash_bag` are over-represented in the merged dataset; `eel`, `scallop`, and `animal_other` are under-represented, leading to lower per-class mAP on rare categories.

### Future Improvements

- **Class-balanced sampling** — oversample rare classes (`eel`, `scallop`, `holothurian`) and use focal loss to down-weight the dominant `fish` class.
- **Class-specific fine-tuning** — dedicated fine-tuning for shark/diver distinction and `echinus`/`holothurian` discrimination with targeted hard-negative mining.
- **Video temporal consistency** — apply a Kalman filter or SORT tracker across frames to suppress detection flickering and maintain consistent track IDs.
- **Optimised inference deployment** — TensorRT FP16 engine for GPU deployment; ONNX + OpenVINO for CPU-only edge deployment.
- **More diverse data** — specifically targeting deeper, darker scenes and heavy particle backscatter environments where current detection fails most frequently.

---

## Validation Checklist

Before submitting for academic evaluation:

- [ ] Git LFS files are present (`git lfs ls-files` confirms all pointers are hydrated)
- [ ] `GET /api/status` returns healthy for YOLO weights, U-Net checkpoint, MiDaS, and all analysis modules
- [ ] `POST /api/enhance` returns a visibly improved image: reduced haze, corrected colour cast, sharpened edges
- [ ] Detection overlays are scene-consistent: fish in water, trash in debris zones, no obviously wrong class assignments
- [ ] Depth map shows plausible relative depth gradients (foreground vs. background)
- [ ] Water quality KPIs are populated and non-zero
- [ ] All five image metrics (PSNR, SSIM, UIQM, UCIQE, EPS) are finite and displayed correctly in the UI
- [ ] Gallery save and retrieve work without error

---

*JalDrishti — built for the ocean, tested in the deep.*
