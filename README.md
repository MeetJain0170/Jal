# 🌊 Jal – See Underwater, For Real

**AI-Based Underwater Image Enhancement System for Increased Maritime Security.**

Your murky underwater photos? We fix that.  
Jal is a deep-learning sidekick that takes sad, blurry, greenish underwater images and turns them into something you’d actually want to look at. Built for a uni project, kept readable on purpose (viva-friendly, promise).

---

## 🎯 What’s the vibe?

- **You give us**: One sad underwater image (turbid, low contrast, that classic “everything is green” vibe).
- **We give you**: A cleaner, punchier image with better contrast, nicer colours, and edges that don’t look like mush.
- **The magic**: A U-Net that learned from **paired data** — degraded → clean — so it actually knows what “good” looks like.

There’s also a **web UI** so you can drag-and-drop and slide before/after. No terminal required. 🖼️

---

## ⚡ Quick start

```bash
pip install -r requirements.txt
```

Folders you’ll care about (they pop up when needed):

- `data/raw/` – the blurry / green stuff  
- `data/enhanced/` – the “reference” clean versions  
- `outputs/` – checkpoints, pretty plots, and logs  

Using **UIEB**? Point `organize_uieb_data.py` at your Kaggle cache and run:

```bash
python organize_uieb_data.py
```

That’ll copy a fixed subset of paired images into `data/raw/` and `data/enhanced/`. Easy.

---

## Pretrained YOLO weights (repository root)

These Ultralytics checkpoints ship with the repo for detection / world-model experiments:

- **`yolov8n.pt`** – YOLOv8 nano  
- **`yolov8s.pt`** – YOLOv8 small  
- **`yolov8s-world.pt`** – YOLOv8 small, World (open-vocabulary)

---

## 🚀 Run everything

**Train the model** (grab coffee for long runs):

```bash
python train.py
```

This will split data, train the U-Net (L1 + SSIM, plus VGG perceptual when it can), save the best checkpoint to `outputs/checkpoints/best_model.pth`, and dump per-epoch comparison pics and training curves. All the good stuff.

**Evaluate on validation set:**

```bash
python evaluate.py
```

**Enhance one image from the CLI:**

```bash
python inference.py input.jpg output.jpg
```

**Fire up the web UI:**

```bash
python api.py
```

Then open **http://localhost:5500** and upload something. Drag, drop, compare. 🎨

---

## 🧠 Model & training (the short version)

- **Architecture**: U-Net, 4 down / 4 up stages, skip connections, BatchNorm + ReLU, Sigmoid at the end so outputs stay in \([0, 1]\).
- **Input size**: \(256 \times 256 \times 3\) (we resize for you).
- **Loss** (when VGG is available):  
  \(\text{loss} = 0.6 \cdot \text{L1} + 0.3 \cdot \text{SSIMLoss} + 0.1 \cdot \text{VGG perceptual}\).  
  No VGG (offline / SSL)? We fall back to L1 + SSIM. No drama.
- **Optimizer**: Adam, lr \(1 \times 10^{-4}\), batch size 8.
- **Epochs**: tweak in `config.py` (e.g. 15 for a quick sanity run, ~80 when you’re serious).
- **Early stopping**: we bail if validation loss gets boring for a while.

We log **PSNR** (dB) and **SSIM** (0–1) so you can flex in the report.

---

## 📁 What’s in the box?

| File / folder      | What it does |
|--------------------|--------------|
| `config.py`        | Paths, image size, hyperparameters. Your control panel. |
| `dataset.py`       | Paired dataset, resize + light augmentation. |
| `model.py`         | U-Net: encoder–decoder + skip connections. |
| `losses.py`        | SSIM loss + optional VGG16 perceptual, one combined class. |
| `train.py`         | Training loop, early stopping, checkpointing, per-epoch grids. |
| `evaluate.py`      | Run on validation split, report PSNR/SSIM. |
| `inference.py`     | CLI: one image in, one enhanced image out. |
| `utils.py`         | Metrics, training curves, comparison image saving. |
| `api.py`           | Flask backend: loads `best_model.pth`, serves `/api/enhance`. |
| `static/`          | The Jal web UI (HTML/CSS/JS). |

---

## 🐠 Honest fine print

- Trained on a **limited UIEB subset**, so it’s good but not magic on every ocean on Earth.
- Default training is CPU; long runs can be slow. Drop `NUM_EPOCHS` in `config.py` for quick experiments.
- Goal here is **clarity and explainability** for viva/review, not to outdo every SOTA paper. Still looks great on a slide. 📊

---

**Jal** – because the sea is cool and your photos should show it. 🌊✨
