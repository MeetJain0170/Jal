/**
 * api.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Thin wrappers around every Flask endpoint.
 * All functions return Promises.  Callers handle UI — no DOM touches here.
 * ─────────────────────────────────────────────────────────────────────────────
 */

const BASE = '';  // Same-origin Flask server.

function fd(file) {
  const f = new FormData();
  f.append('image', file);
  return f;
}

async function dataUrlToFile(dataUrl, filename = 'enhanced.png') {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || 'image/png' });
}

/**
 * GET /api/status
 */
export async function fetchStatus() {
  const r = await fetch(`${BASE}/api/status`);
  return r.json();
}

/**
 * POST /api/enhance
 * Returns {success, enhanced_image_hybrid, enhanced_image_opencv, psnr, ssim,
 *          uiqm, uciqe, eps, size, processing_time, heatmap}
 */
export async function enhance(file) {
  const r = await fetch(`${BASE}/api/enhance`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/detect
 * @param {File} inferenceFile  Image used for YOLO inference and annotation
 */
export async function detect(inferenceFile) {
  const f = new FormData();
  f.append('image', inferenceFile);
  const r = await fetch(`${BASE}/api/detect`, { method: 'POST', body: f });
  return r.json();
}

/**
 * POST /api/depth
 */
export async function depth(file) {
  const r = await fetch(`${BASE}/api/depth`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/analyze-water
 */
export async function analyzeWater(file) {
  const r = await fetch(`${BASE}/api/analyze-water`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/process-video
 */
export async function processVideo(file) {
  const f = new FormData();
  f.append('video', file);
  const r = await fetch(`${BASE}/api/process-video`, { method: 'POST', body: f });
  return r.json();
}

// ── Gallery persistence ───────────────────────────────────────────────────────

/**
 * GET /api/gallery
 */
export async function fetchGallery() {
  const r = await fetch(`${BASE}/api/gallery`);
  return r.json();
}

/**
 * POST /api/gallery/save
 */
export async function saveToGalleryAPI(enhanced_b64, originalName, metrics = {}) {
  const r = await fetch(`${BASE}/api/gallery/save`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enhanced_b64, filename: originalName, ...metrics }),
  });
  return r.json();
}

/**
 * DELETE /api/gallery/clear
 */
export async function clearGalleryAPI() {
  const r = await fetch(`${BASE}/api/gallery/clear`, { method: 'DELETE' });
  return r.json();
}

/**
 * Run all four analysis APIs.
 * Enhancement runs first, then detect/depth/water fire in parallel immediately
 * using the original file (no blob conversion delay).
 *
 * @param {File}   file       The original uploaded file
 * @param {Object} callbacks  { onEnhance, onDetect, onDepth, onWater, onError }
 */
export async function analyzeAll(file, { onEnhance, onDetect, onDepth, onWater, onError } = {}) {
  try {
    const wrap = (promise, cb) =>
      promise.then(data => cb && cb(data)).catch(err => onError && onError(err));

    // 1. Enhancement FIRST
    const enhanceData = await enhance(file);
    try { if (onEnhance) onEnhance(enhanceData); } catch (e) { console.warn('[ui] onEnhance render error', e); }


    // 2. Run detection fully on enhanced output (inference + visualization).
    let inferenceFile = file;
    if (enhanceData?.enhanced_image_hybrid) {
      try {
        inferenceFile = await dataUrlToFile(
          enhanceData.enhanced_image_hybrid,
          `enhanced_${file.name || 'input'}.png`
        );
      } catch (e) {
        console.warn('[api] enhanced image conversion failed, fallback to original', e);
      }
    }

    await Promise.all([
      wrap(detect(inferenceFile), onDetect),
      wrap(depth(file),        onDepth),
      wrap(analyzeWater(file), onWater),
    ]);
  } catch (err) {
    if (onError) onError(err);
  }
}