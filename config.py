"""
config.py — Central configuration for JalDrishti Maritime Security Vision System.
All modules import from here. Change values once, effects propagate everywhere.
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(BASE_DIR, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
ENHANCED_DIR = os.path.join(DATA_DIR, 'enhanced')
SAMPLE_DIR   = os.path.join(DATA_DIR, 'sample')

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR     = os.path.join(BASE_DIR, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR    = os.path.join(OUTPUT_DIR, 'results')
LOGS_DIR       = os.path.join(OUTPUT_DIR, 'logs')
GALLERY_DIR    = os.path.join(OUTPUT_DIR, 'gallery')
GALLERY_JSON   = os.path.join(GALLERY_DIR, 'gallery.json')

for _d in [OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, GALLERY_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Model architecture ───────────────────────────────────────────────────────
IMAGE_HEIGHT       = 256
IMAGE_WIDTH        = 256
IMAGE_CHANNELS     = 3
UNET_BASE_FEATURES = 64
UNET_INIT_FEATURES = UNET_BASE_FEATURES   # alias kept for train.py

# ── Patch inference ───────────────────────────────────────────────────────────
PATCH_SIZE    = 256
PATCH_OVERLAP = 64
BLEND_ALPHA   = 0.65   # weight for model output; 1.0=pure model, 0.0=pure original
                        # 0.65 = natural restoration balance; 0.85 was causing over-darkening

# ── Training hyper-parameters ────────────────────────────────────────────────
BATCH_SIZE     = 8
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 80
TRAIN_SPLIT    = 0.8
RANDOM_SEED    = 42
GRAD_CLIP_NORM = 1.0

# ── Loss weights ──────────────────────────────────────────────────────────────
# Loss weights — tuned for NATURAL color fidelity over cinematic contrast
# Old sum: 1.35 (over-biased to edge/structure). New sum: ~1.0 (balanced).
LOSS_W_L1      = 0.40   # Pixel-level fidelity (primary anchor)
LOSS_W_SSIM    = 0.25   # Structural similarity (perceptual quality)
LOSS_W_PERC    = 0.12   # Perceptual VGG features (reduced to avoid over-sharpening)
LOSS_W_EDGE    = 0.08   # Edge preservation (reduced — was pushing cinematic look)
LOSS_W_TV      = 0.08   # Total Variation — smoothness, prevent static hallucination
LOSS_W_PHYSICS = 0.12   # Dark channel prior — prevents magenta bleeds
LOSS_W_CHROMA  = 0.08   # NEW: LAB chroma (A+B) preservation — prevents color drain/monochrome look

LOSS_SCALE_FULL    = 1.0
LOSS_SCALE_HALF    = 0.4
LOSS_SCALE_QUARTER = 0.2

# ── LR Scheduler ─────────────────────────────────────────────────────────────
USE_LR_SCHEDULER = True
LR_MIN           = 1e-6

# ── Early stopping ────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 10
EARLY_STOP_METRIC   = "psnr"
SAVE_FREQ           = 10

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0
PIN_MEMORY  = torch.cuda.is_available()

# ── Evaluation & visualisation ───────────────────────────────────────────────
NUM_VISUALIZATION_SAMPLES = 5
SAVE_ALL_PREDICTIONS      = False

# ── Detection (max-quality profile) ──────────────────────────────────────────
# High-capacity model + larger inference size + TTA/multi-scale.
def _pick_trained_yolo_weights() -> str:
    """
    Prefer Step 7 training outputs when available (best > last), else fallback.

    We check a few common locations because training/output cwd can vary and
    users may copy weights in manually.
    """
    candidates = [
        # Expected Ultralytics output from Step 7
        os.path.join(BASE_DIR, "runs", "marine", "v1", "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "marine", "v1", "weights", "last.pt"),
        # Local downloaded artifacts (e.g. from Hugging Face dataset mirror)
        os.path.join(BASE_DIR, "data", "JalDrishti", "best.pt"),
        # If user copied only the file into runs/
        os.path.join(BASE_DIR, "runs", "best.pt"),
        os.path.join(BASE_DIR, "runs", "last.pt"),
        # Fallback
        "yolov8l-worldv2.pt",
    ]
    for p in candidates:
        if p.endswith(".pt") and os.path.isabs(p) and os.path.exists(p):
            return p
        if not os.path.isabs(p):
            return p  # fallback model name
    return "yolov8l-worldv2.pt"


YOLO_WEIGHTS = _pick_trained_yolo_weights()
_USING_TRAINED_WEIGHTS = isinstance(YOLO_WEIGHTS, str) and (os.path.isabs(YOLO_WEIGHTS) and os.path.exists(YOLO_WEIGHTS))
# Early epochs produce low-confidence boxes; use a softer threshold while training checkpoints are used.
YOLO_CONF_THRESH = 0.08 if _USING_TRAINED_WEIGHTS else 0.12
YOLO_IOU_THRESH  = 0.45
YOLO_IMG_SIZE    = 1280
YOLO_TTA         = False
YOLO_MULTI_SCALE = False

# ── Diver detection (pretrained, COCO person → Diver) ─────────────────────────
# Your custom-trained marine model does not include a "diver" class. To keep
# diver detection (as the project had before the pipeline), we run a small
# pretrained model alongside the marine model and merge results.
ENABLE_DIVER_DETECTOR = True
DIVER_WEIGHTS         = "yolov8x.pt"
DIVER_CONF_THRESH     = 0.30
DIVER_IOU_THRESH      = 0.45
DIVER_IMG_SIZE        = 1280

# ── Specialist detectors (optional, togglable) ───────────────────────────────
# These run alongside the marine + diver detectors when enabled and keep only
# their target classes for stronger class-specific recall.
ENABLE_SHARK_DETECTOR  = True
SHARK_WEIGHTS          = "yolov8l.pt"
SHARK_CONF_THRESH      = 0.15
SHARK_IOU_THRESH       = 0.45
SHARK_IMG_SIZE         = 1280

ENABLE_FISH_DETECTOR   = True
FISH_WEIGHTS           = "yolov8m.pt"
FISH_CONF_THRESH       = 0.15
FISH_IOU_THRESH        = 0.45
FISH_IMG_SIZE          = 1280

# Hard minimum confidence filter applied before final response.
DETECTION_MIN_CONFIDENCE = 0.15
STRICT_MARINE_LABELS_ONLY = True
ENABLE_WORLD_MARINE_AUGMENT = False
ENABLE_STRUCTURE_RESCUE = False
# Runtime behavior mode for /api/detect:
# - marine_clean: strict diver/fish/shark outputs, low noise
# - full_debug:   keep wider labels/augmentations for debugging
DETECTION_MODE = "marine_clean"

# Active detector profile for /api/detect when request does not provide one.
# Allowed: "base", "marine_only", "shark_focus", "fish_focus", "full"
DETECTION_PROFILE      = "full"

# ── Detection input preprocessing ─────────────────────────────────────────────
# If True, `/api/detect` will run inference on the "classical OpenCV polish"
# image (same resolution) instead of the raw uploaded image.
DETECT_ON_OPENCV_POLISH = False

# ── Depth estimation ─────────────────────────────────────────────────────────
MIDAS_MODEL_TYPE = "MiDaS_small"

# ── Water quality ────────────────────────────────────────────────────────────
TURBIDITY_DARK_PATCH_PCT = 0.001

# ── Video processing ─────────────────────────────────────────────────────────
VIDEO_MAX_FRAMES  = 300
VIDEO_FPS_DEFAULT = 25
VIDEO_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, 'videos')
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# ── Data pipeline ────────────────────────────────────────────────────────────
DATASET_TYPE      = 'paired'
USE_AUGMENTATION  = False
AUGMENTATION_PROB = 0.5

# ── Logging ───────────────────────────────────────────────────────────────────
PRINT_FREQ = 10
VERBOSE    = True