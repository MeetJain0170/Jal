from __future__ import annotations
import numpy as np
import cv2

# ------------------------------------------------------------------------------
# DARK CHANNEL (FAST)
# ------------------------------------------------------------------------------

def _fast_dark_channel(img, patch=15):
    # Dark channel should consider all RGB channels; using only G/B
    # inflates haze in many clear/coastal scenes.
    min_c = img.min(axis=2)
    try:
        from scipy.ndimage import minimum_filter
        return minimum_filter(min_c, size=patch)
    except:
        kernel = np.ones((patch, patch), np.uint8)
        return cv2.erode(min_c, kernel)


# ------------------------------------------------------------------------------
# TURBIDITY
# ------------------------------------------------------------------------------

def estimate_turbidity(img):
    dark = _fast_dark_channel(img)
    haze = float(dark.mean())

    if haze < 0.1:
        level = "Clear"
    elif haze < 0.22:
        level = "Moderate"
    elif haze < 0.4:
        level = "High"
    else:
        level = "Severe"

    return {
        "turbidity_index": round(haze, 4),
        "turbidity_level": level,
        "haze_density": round(haze, 4),
    }


# ------------------------------------------------------------------------------
# VISIBILITY
# ------------------------------------------------------------------------------

def estimate_visibility(img):
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

    h, w = gray.shape
    bh, bw = max(1, h//8), max(1, w//8)

    contrasts = []
    for i in range(8):
        for j in range(8):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            mn, mx = block.min(), block.max()
            if mn + mx > 1e-5:
                contrasts.append((mx - mn) / (mx + mn))

    mean_contrast = float(np.mean(contrasts)) if contrasts else 0
    contrast_loss = max(0.0, 1.0 - mean_contrast)

    attenuation = 0.12 * (1 + contrast_loss * 4)

    visibility = int((mean_contrast / (attenuation + 1e-6)) * 5)
    visibility = int(np.clip(visibility, 1, 40))

    return {
        "visibility_range_meters": visibility,
        "contrast_loss": round(contrast_loss, 4),
        "mean_contrast": round(mean_contrast, 4),
    }


# ------------------------------------------------------------------------------
# COLOR ATTENUATION
# ------------------------------------------------------------------------------

def estimate_color_attenuation(img):
    r, g, b = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
    ref = max(r, g, b, 1e-5)

    return {
        "red": round(1 - r/ref, 4),
        "green": round(1 - g/ref, 4),
        "blue": round(1 - b/ref, 4),
    }


# ------------------------------------------------------------------------------
# SEABED DETECTION
# ------------------------------------------------------------------------------

def detect_seabed(img):
    gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])

    texture = np.std(gray)
    edges = cv2.Canny((gray*255).astype(np.uint8), 50, 150)
    edge_density = edges.mean() / 255
    horizontal_gradient = np.abs(np.diff(gray, axis=0)).mean()
    vertical_gradient   = np.abs(np.diff(gray, axis=1)).mean()

    # More conservative: require low texture AND very flat gradients AND low edge density
    flatness = horizontal_gradient < 0.025 and vertical_gradient < 0.025
    if texture < 0.06 and edge_density < 0.04 and flatness:
        return True
    return False


# ------------------------------------------------------------------------------
# DEPTH HEURISTIC
# ------------------------------------------------------------------------------

def estimate_depth_heuristic(img, att):
    r, g, b = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()

    brightness = img.mean()
    # Clamp blue_ratio to [0, 4] to avoid extreme values on clear-blue sky / surface shots
    blue_ratio = min(b / (r + 1e-6), 4.0)

    depth_score = (
        0.5 * blue_ratio +
        0.3 * att["red"] +
        0.2 * (1.0 - min(brightness, 1.0))
    )

    return depth_score


# ------------------------------------------------------------------------------
# ENVIRONMENT CLASSIFIER
# ------------------------------------------------------------------------------

def classify_environment(img, vis, turb, att):
    is_seabed   = detect_seabed(img)
    depth_score = estimate_depth_heuristic(img, att)

    visibility = vis["visibility_range_meters"]
    turbidity  = turb["turbidity_index"]

    # Priority 1: Seabed texture signature
    if is_seabed:
        return "Seabed"

    # Guardrail: bright/clear scenes should not be forced into "deep sea".
    mean_brightness = float(img.mean())
    is_clear_scene = visibility >= 14 and turbidity < 0.20

    # Priority 2: True deep sea — require strong depth cues + low visibility.
    if (
        depth_score > 1.95
        and att["red"] > 0.70
        and visibility < 11
        and turbidity < 0.45
        and mean_brightness < 0.45
        and not is_clear_scene
    ):
        return "Deep Sea"

    # Priority 3: Highly turbid — require both high turbidity and poor visibility.
    if turbidity > 0.58 and visibility < 12:
        return "Highly Turbid"

    # Priority 4: Clear water / good visibility
    if visibility > 18:
        return "Clear Surface Water"

    # Priority 5: Moderate coastal / open water
    if visibility > 8:
        return "Moderate Coastal"

    # Priority 6: Shallow but some turbidity
    if turbidity < 0.20:
        return "Shallow Coastal"

    return "Low Visibility Coastal"


# ------------------------------------------------------------------------------
# FINAL PIPELINE
# ------------------------------------------------------------------------------

def analyze_water_quality(img_np):
    img = img_np.astype(np.float32) / 255.0

    turb = estimate_turbidity(img)
    vis  = estimate_visibility(img)
    att  = estimate_color_attenuation(img)

    env = classify_environment(img, vis, turb, att)

    return {
        "visibility_range_meters": vis["visibility_range_meters"],
        "contrast_loss":  vis["contrast_loss"],
        "mean_contrast":  vis["mean_contrast"],
        "turbidity_level": turb["turbidity_level"],
        "turbidity_index": turb["turbidity_index"],
        "haze_density":    turb["haze_density"],
        "attenuation":     att,
        "environment_type": env,
    }


