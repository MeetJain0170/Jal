/**
 * ui.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Pure UI layer — no raw fetch() calls.  All API calls go through api.js.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { analyzeAll, enhance, fetchStatus,
         fetchGallery, saveToGalleryAPI, clearGalleryAPI } from './api.js';

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
const $ = id  => document.getElementById(id);
const setText = (id, val) => { const e = $(id); if (e) e.textContent = val ?? '—'; };
const setHTML = (id, val) => { const e = $(id); if (e) e.innerHTML  = val; };
const show    = (id, d='block') => { const e = $(id); if (e) e.style.display = d; };
const hide    = id => { const e = $(id); if (e) e.style.display = 'none'; };

/* ═══════════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════════ */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      $(`${btn.dataset.tab}-tab`).classList.add('active');
    });
  });
}

/* ═══════════════════════════════════════════════════════════
   RIPPLE
═══════════════════════════════════════════════════════════ */
function initRipple() {
  document.querySelectorAll('.ripple-container').forEach(el => {
    el.addEventListener('click', e => {
      const r   = el.getBoundingClientRect();
      const w   = Math.max(r.width, r.height);
      const rip = document.createElement('span');
      rip.className = 'ripple-wave';
      Object.assign(rip.style, {
        width:  w + 'px', height: w + 'px',
        left:   (e.clientX - r.left - w / 2) + 'px',
        top:    (e.clientY - r.top  - w / 2) + 'px',
      });
      el.appendChild(rip);
      setTimeout(() => rip.remove(), 800);
    });
  });
}

/* ═══════════════════════════════════════════════════════════
   PIPELINE STATUS STRIP
═══════════════════════════════════════════════════════════ */
const setPipe = (id, state) => {
  const e = $(`pipe-${id}`);
  if (e) e.className = 'pipeline-step' + (state ? ` ${state}` : '');
};

/* ═══════════════════════════════════════════════════════════
   SESSION STATS
═══════════════════════════════════════════════════════════ */
const session = { count: 0, psnr: 0, ssim: 0, uiqm: 0, time: 0 };

function updateSessionStats(data) {
  session.count++;
  if (data.psnr)            session.psnr += data.psnr;
  if (data.ssim)            session.ssim += data.ssim;
  if (data.uiqm)            session.uiqm += data.uiqm;
  if (data.processing_time) session.time += data.processing_time;
  const n = session.count;
  setText('totalProcessed', n);
  setText('avgPSNR',        (session.psnr / n).toFixed(2));
  setText('statsProcessed', n);
  setText('statsAvgPSNR',   (session.psnr / n).toFixed(2));
  setText('statsAvgSSIM',   (session.ssim / n).toFixed(4));
  setText('statsAvgUIQM',   session.uiqm > 0 ? (session.uiqm / n).toFixed(3) : '—');
  setText('statsTotalTime', session.time.toFixed(1) + 's');
}

/* ═══════════════════════════════════════════════════════════
   DASHBOARD PANEL UPDATERS
═══════════════════════════════════════════════════════════ */

function applyEnhance(file, data) {
  setPipe('enhance', data.success ? 'done' : 'error');
  // Null-safe src setters — elements may not exist if layout changed
  const setSrc = (id, src) => { const e = $(id); if (e && src) e.src = src; };
  const originalSrc = URL.createObjectURL(file);

  // Always show input image as baseline so panel is never blank.
  setSrc('imgOriginal', originalSrc);
  if (!data.success) {
    setSrc('imgEnhancedHybrid', originalSrc);
    setSrc('imgEnhancedOpencv', originalSrc);
    setSrc('imgOpencvFull', originalSrc);
    setText('procTime', 'Enhancement unavailable');
    return;
  }

  setSrc('imgEnhancedHybrid', data.enhanced_image_hybrid || originalSrc);
  setSrc('imgEnhancedOpencv', data.enhanced_image_opencv || originalSrc);  // may be absent (layout v2)
  setSrc('imgOpencvFull', data.enhanced_image_opencv || originalSrc);  // full-res panel in Row 2

  const fmt = (v, d) => (Number.isFinite(v) ? Number(v).toFixed(d) : '—');
  setText('imgSize',    data.size || '—');
  setText('psnrScore',  fmt(data.psnr, 2));
  setText('ssimScore',  fmt(data.ssim, 4));
  setText('uiqmScore',  fmt(data.uiqm, 3));
  setText('uciqeScore', fmt(data.uciqe, 3));
  setText('epsScore',   fmt(data.eps, 3));
  setText('procTime',   Number.isFinite(data.processing_time) ? Number(data.processing_time).toFixed(2) + 's' : '—');

  window._lastEnhanced = { ...data, _originalFile: file };
  updateSessionStats(data);
}


/* ─── Threat & Confidence colour palette ─── */
const THREAT_COLORS = {
  'Security Threat': '#ff4444',
  'Diver':           '#ff9955',
  'Surface Vessel':  '#f0c040',
  'Marine Life':     '#55cc88',
  'Object':          '#4fc3f7',
};

/**
 * Render the merged Threat + Confidence section.
 * Each detection gets a visible confidence bar terminating in an exact %.
 * We do not guess fine-grained species/genus here; we show only model labels.
 */
function renderThreatPanel(data) {
  const detections = data.detections || [];

  // ── Score ring + alert badge ──────────────────────────────────────────────
  const score = data.threat_score || 0;
  setText('threatScoreVal', Math.round(score));
  const ring  = $('threatRing');
  const badge = $('alertBadge');
  ring.className  = 'threat-score-ring';
  badge.className = 'alert-badge';
  if (data.alert_level === 'Red') {
    ring.classList.add('red');    badge.classList.add('red');
    badge.textContent = '⚠ RED ALERT';
  } else if (data.alert_level === 'Yellow') {
    ring.classList.add('yellow'); badge.classList.add('yellow');
    badge.textContent = '⚡ YELLOW ALERT';
  } else {
    badge.textContent = '✔ GREEN – CLEAR';
  }

  // ── Merged detection list + confidence bars ───────────────────────────────
  const dl = $('detectList');
  dl.innerHTML = '';

  if (!detections.length) {
    dl.innerHTML = '<div style="opacity:.4;font-size:12px;padding:10px 0;">No objects detected</div>';
  } else {
    // Sort: threats first, then by confidence
    const sorted = [...detections].sort((a, b) => {
      const pri = { 'Security Threat': 0, 'Diver': 1, 'Surface Vessel': 2, 'Marine Life': 3, 'Object': 4 };
      const ap = pri[a.category] ?? 5, bp = pri[b.category] ?? 5;
      return ap !== bp ? ap - bp : b.confidence - a.confidence;
    });

    sorted.forEach(d => {
      const pct   = (d.confidence * 100).toFixed(1);
      const color = THREAT_COLORS[d.category] || '#4fc3f7';
      const isThreat = d.category === 'Security Threat';
      const confTip = `Confidence (${pct}%): the model's certainty that this detected object class is correct. This percentage is detection reliability, not threat severity.`;

      const item = document.createElement('div');
      item.className = 'detect-item' + (isThreat ? ' di-threat' : '');
      item.style.cssText = 'margin-bottom:10px;padding:10px 12px;border-radius:10px;' +
        `background:${color}11;border:1px solid ${color}33;`;

      item.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <div style="display:flex;align-items:center;gap:7px;">
            <span style="font-size:13px;font-weight:600;color:${color};">${d.display_class || d.class}</span>
            <span style="font-size:10px;opacity:.45;padding:2px 6px;border-radius:20px;
                         background:${color}18;">${d.category}</span>
          </div>
          <div class="has-tip"
               data-tip="${confTip.replace(/"/g, '&quot;')}"
               style="display:flex;align-items:center;gap:4px;cursor:help;">
            <span style="font-size:8px;letter-spacing:.07em;opacity:.4;text-transform:uppercase;">Conf</span>
            <span style="font-size:12px;font-weight:700;color:${color};">${pct}%</span>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="flex:1;background:rgba(255,255,255,.08);border-radius:4px;height:5px;overflow:hidden;">
            <div style="height:100%;width:${pct}%;background:${color};border-radius:4px;
                        box-shadow:0 0 6px ${color}88;transition:width .6s;"></div>
          </div>
        </div>`;

      dl.appendChild(item);
    });
  }

  // ── Recommendations ───────────────────────────────────────────────────────
  const rl = $('recommendList');
  rl.innerHTML = '';
  (data.recommendations || []).forEach(r => {
    const e = document.createElement('div');
    e.className = 'recommend-item'; e.textContent = r;
    rl.appendChild(e);
  });
}

function applyDetect(data) {
  setPipe('detect', data.success ? 'done' : 'error');
  hide('detectPending');

  // Annotated image: boxes are already drawn on the original by the backend
  if (data.annotated_image) {
    const img = $('annotatedImg');
    img.src = data.annotated_image; img.style.display = 'block';
  }

  hide('threatPending');
  show('threatPanel', 'flex');   // flex container — must use 'flex' not 'block'
  renderThreatPanel(data);
}


function applyDepth(data) {
  setPipe('depth', data.success ? 'done' : 'error');
  hide('depthPending');
  if (data.depth_map) {
    const img = $('depthMapImg');
    img.src = data.depth_map; img.style.display = 'block';
  }
  setText('avgDepth', data.average_depth != null
    ? (data.average_depth * 100).toFixed(1) + '%' : '—');
  const zones = data.object_distances || [];
  const total = zones.reduce((s, z) => s + z.pixels, 0) || 1;
  zones.forEach(z => {
    const id = z.zone === 'Near' ? 'dzNear' : z.zone === 'Mid' ? 'dzMid' : 'dzFar';
    const e = $(id); if (e) e.style.flex = String(z.pixels / total * 10);
  });
}

function applyWater(data) {
  setPipe('water', data.success ? 'done' : 'error');
  hide('waterPending'); show('waterPanel');

  // Visibility KPI — with tooltip
  const visEl = $('visibilityVal');
  if (visEl) {
    visEl.innerHTML = (data.visibility_range_meters || '—') + '<small>m</small>';
    visEl.title = 'Estimated horizontal visibility range in metres. Derived from local contrast analysis across image blocks. < 5m = very poor, > 15m = clear conditions.';
    visEl.style.cursor = 'help';
  }
  setText('visibilityMetric', (data.visibility_range_meters || '—') + 'm');

  // Turbidity KPI — with tooltip
  const turbEl = $('turbidityLevel');
  if (turbEl) {
    turbEl.textContent = data.turbidity_level || '—';
    turbEl.title = 'Turbidity measures how much suspended particles (sediment, algae, organic matter) reduce water transparency. Clear < 0.1, Moderate 0.1–0.22, High 0.22–0.55, Severe > 0.55. Estimated via Dark Channel Prior analysis.';
    turbEl.style.cursor = 'help';
  }

  // Environment KPI — with tooltip
  const envEl = $('environmentType');
  if (envEl) {
    envEl.textContent = data.environment_type || '—';
    envEl.title = 'Scene environment classification based on colour attenuation depth score, turbidity index, and seabed detection. Categories: Clear Surface Water, Moderate Coastal, Shallow Coastal, Low Visibility Coastal, Highly Turbid, Deep Sea, Seabed.';
    envEl.style.cursor = 'help';
  }

  setText('visibilityExplain',
    'Visibility estimates how far features remain distinguishable in this frame.');
  setText('turbidityExplain',
    'Turbidity reflects suspended particles (sediment/algae) reducing clarity.');
  setText('environmentExplain',
    'Environment is the scene type inferred from visibility, turbidity and color attenuation.');

  // Turbidity gauge
  const gauge = $('turbidityGauge');
  if (gauge) gauge.style.setProperty('--gauge-pos', (data.turbidity_index * 100).toFixed(0) + '%');

  // Attenuation bars
  const att = data.attenuation || {};
  ['Red', 'Green', 'Blue'].forEach(c => {
    const v   = att[c.toLowerCase()] || 0;
    const f   = $('att' + c); if (f) f.style.width = (v * 100).toFixed(0) + '%';
    const lbl = $('att' + c + 'Val'); if (lbl) lbl.textContent = (v * 100).toFixed(0) + '%';
  });
}

/* ═══════════════════════════════════════════════════════════
   MAIN UPLOAD HANDLER
═══════════════════════════════════════════════════════════ */
function handleFile(file) {
  hide('dropZone');
  show('resultsSection');

  ['enhance', 'detect', 'depth', 'water'].forEach(p => setPipe(p, 'active'));
  ['detectPending', 'depthPending', 'waterPending', 'threatPending'].forEach(id => show(id, 'flex'));
  ['annotatedImg', 'depthMapImg'].forEach(id => hide(id));
  ['waterPanel', 'threatPanel'].forEach(id => hide(id));

  analyzeAll(file, {
    onEnhance: data => applyEnhance(file, data),
    onDetect:  data => applyDetect(data),
    onDepth:   data => applyDepth(data),
    onWater:   data => applyWater(data),
    onError:   _err => { /* pipes marked error per handler */ },
  });
}

/* ═══════════════════════════════════════════════════════════
   DROP ZONE
═══════════════════════════════════════════════════════════ */
function initDropZone() {
  const dropZone  = $('dropZone');
  const fileInput = $('fileInput');
  if (!dropZone) return;
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragging'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
}

/* ═══════════════════════════════════════════════════════════
   ACTION BUTTONS
═══════════════════════════════════════════════════════════ */
function initActions() {
  const downloadDataUrl = (href, filename) => {
    const a = document.createElement('a');
    a.href = href;
    a.download = filename;
    a.click();
  };

  const baseName = (name = 'image') => {
    const idx = name.lastIndexOf('.');
    return idx > 0 ? name.slice(0, idx) : name;
  };

  window.downloadImage = () => {
    if (!window._lastEnhanced) return;
    const d = window._lastEnhanced;
    const stem = baseName(d._originalFile?.name || 'enhanced');
    // Download hybrid full-resolution PNG
    if (d.enhanced_image_hybrid) {
      downloadDataUrl(d.enhanced_image_hybrid, `${stem}_hybrid_fullres.png`);
    }
    // Download OpenCV polished full-resolution PNG
    if (d.enhanced_image_opencv) {
      setTimeout(() => downloadDataUrl(d.enhanced_image_opencv, `${stem}_opencv_fullres.png`), 250);
    }
  };

  window.resetUpload = () => {
    hide('resultsSection'); show('dropZone');
    $('fileInput').value = ''; window._lastEnhanced = null;
  };

  window.saveToGallery = async () => {
    if (!window._lastEnhanced) return;
    const d = window._lastEnhanced;
    const stem = baseName(d._originalFile?.name || 'image');
    // Save hybrid to gallery
    try {
      await saveToGalleryAPI(
        d.enhanced_image_hybrid,
        `${stem}_hybrid_fullres`,
        { psnr: d.psnr, ssim: d.ssim, eps: d.eps,
          uiqm: d.uiqm, uciqe: d.uciqe,
          size: d.size, processing_time: d.processing_time }
      );
      // Save opencv to gallery too
      if (d.enhanced_image_opencv) {
        await saveToGalleryAPI(
          d.enhanced_image_opencv,
          `${stem}_opencv_fullres`,
          { size: d.size }
        );
      }
    } catch (e) { console.warn('[gallery] backend save failed', e); }
    // Add hybrid to local grid
    addGalleryItem({
      enhanced_b64: d.enhanced_image_hybrid,
      saved_at:     new Date().toLocaleTimeString(),
      label:        'Hybrid AI',
      psnr:         d.psnr, ssim: d.ssim,
    });
    // Add opencv to local grid
    if (d.enhanced_image_opencv) {
      addGalleryItem({
        enhanced_b64: d.enhanced_image_opencv,
        saved_at:     new Date().toLocaleTimeString(),
        label:        'OpenCV',
        psnr:         null, ssim: null,
      });
    }
  };

  window.clearGallery = async () => {
    try { await clearGalleryAPI(); } catch (e) { console.warn('[gallery] clear failed', e); }
    setHTML('galleryGrid', `<div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" width="56" height="56">
        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg><p>No enhanced images yet.</p></div>`);
  };
}

/* ═══════════════════════════════════════════════════════════
   GALLERY
═══════════════════════════════════════════════════════════ */
function addGalleryItem(entry) {
  const grid  = $('galleryGrid');
  const empty = grid.querySelector('.empty-state');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = 'gallery-item';
  item.innerHTML = `
    <img src="${entry.enhanced_b64}" alt="Enhanced">
    <div class="gallery-meta">
      <div class="gallery-time">${entry.saved_at || '—'} ${ entry.label ? `<span style="opacity:.5;font-size:9px;">• ${entry.label}</span>` : '' }</div>
      <div class="gallery-scores">
        <span>PSNR ${entry.psnr ? Number(entry.psnr).toFixed(2) : '—'}</span>
        <span>SSIM ${entry.ssim ? Number(entry.ssim).toFixed(4) : '—'}</span>
      </div>
    </div>`;
  item.style.cursor = 'pointer';
  item.addEventListener('click', async () => {
    // Switch to Analysis tab
    document.querySelectorAll('.tab-btn').forEach(b => {
      if(b.dataset.tab === 'single') b.click();
    });
    // Create a File object from the base64 image and trigger analysis
    try {
      const res = await fetch(entry.enhanced_b64);
      const blob = await res.blob();
      const file = new File([blob], "gallery_image.png", { type: blob.type || "image/png" });
      handleFile(file);
    } catch(e) {
      console.warn('Could not analyze gallery item:', e);
    }
  });
  grid.prepend(item);
}

async function initGallery() {
  try {
    const result = await fetchGallery();
    if (result.success && result.entries && result.entries.length)
      result.entries.forEach(e => addGalleryItem(e));
  } catch (err) { console.warn('[gallery] failed to load', err); }
}

/* ═══════════════════════════════════════════════════════════
   BATCH — sequential processing
═══════════════════════════════════════════════════════════ */
function initBatch() {
  const batchInput = $('batchFileInput');
  if (!batchInput) return;

  const processBatchFiles = async (files) => {
    if (!files || !files.length) return;
    const fill = $('progressFill'), progressT = $('progressText'), results = $('batchResults');
    results.innerHTML = ''; show('batchProgress'); hide('batchActions');
    let done = 0; window._batchResults = [];

    for (const file of files) {
      try {
        const data = await enhance(file);
        done++;
        const pct = Math.round((done / files.length) * 100);
        fill.style.width = pct + '%'; progressT.textContent = `${done}/${files.length}`;
        if (!data.error && data.success) {
          window._batchResults.push({ data, file });
          const item = document.createElement('div');
          item.className = 'batch-item';
          item.innerHTML = `
            <img src="${data.enhanced_image_hybrid}" alt="">
            <div class="batch-meta">
              <div class="name">${file.name}</div>
              <div class="scores">
                <span>PSNR ${data.psnr ? data.psnr.toFixed(1) : '—'}</span>
                <span>SSIM ${data.ssim ? data.ssim.toFixed(3) : '—'}</span>
              </div>
            </div>`;
          item.style.cursor = 'pointer';
          item.addEventListener('click', () => {
            // Switch to Analysis tab
            document.querySelectorAll('.tab-btn').forEach(b => {
              if(b.dataset.tab === 'single') b.click();
            });
            // Analyze original file
            handleFile(file);
          });
          results.appendChild(item);
        }
      } catch (err) {
        done++;
        fill.style.width = Math.round((done / files.length) * 100) + '%';
        progressT.textContent = `${done}/${files.length}`;
      }
    }
    show('batchActions', 'flex');
  };

  batchInput.addEventListener('change', () => {
    processBatchFiles(Array.from(batchInput.files));
  });

  const batchZone = document.querySelector('.batch-upload-area');
  if (batchZone) {
    batchZone.addEventListener('dragover', e => { 
      e.preventDefault(); 
      batchZone.style.borderColor = 'var(--cyan)'; 
      batchZone.style.backgroundColor = 'rgba(0,200,255,0.05)';
    });
    batchZone.addEventListener('dragleave', () => { 
      batchZone.style.borderColor = 'rgba(0,200,255,0.15)'; 
      batchZone.style.backgroundColor = 'transparent';
    });
    batchZone.addEventListener('drop', e => {
      e.preventDefault();
      batchZone.style.borderColor = 'rgba(0,200,255,0.15)';
      batchZone.style.backgroundColor = 'transparent';
      if (e.dataTransfer && e.dataTransfer.files) {
        processBatchFiles(Array.from(e.dataTransfer.files));
      }
    });
  }

  window.downloadAllBatch = () => {
    (window._batchResults || []).forEach(({ data, file }, i) => {
      setTimeout(() => {
        const a = document.createElement('a');
        a.href = data.enhanced_image_hybrid; a.download = `enhanced_${file.name}`; a.click();
      }, i * 300);
    });
  };

  window.clearBatch = () => {
    $('batchResults').innerHTML = ''; hide('batchProgress'); hide('batchActions');
    window._batchResults = []; $('batchFileInput').value = '';
  };
}

/* ═══════════════════════════════════════════════════════════
   MODEL STATUS
═══════════════════════════════════════════════════════════ */
function initStatus() {
  fetchStatus()
    .then(d => {
      const pill = $('status');
      if (d.model_loaded) {
        pill.classList.add('ready');
        pill.querySelector('.status-text').textContent = 'Model Ready';
        setText('deviceInfo', d.device);
        if (d.parameters) setText('paramCount', Number(d.parameters).toLocaleString());
      } else {
        pill.classList.add('error');
        pill.querySelector('.status-text').textContent = 'Model Not Loaded';
      }
      if (d.modules) {
        const mp = $('modulePills');
        if (mp) {
          mp.innerHTML = '';
          [{ key:'yolo', label:'YOLOv8 Detection' },
           { key:'midas', label:'MiDaS Depth' },
           { key:'opencv', label:'OpenCV Video' }].forEach(m => {
            const e = document.createElement('div');
            e.className = 'module-pill ' + (d.modules[m.key] ? 'ok' : 'warn');
            e.innerHTML = `<span class="module-pill-dot"></span>${m.label}`;
            mp.appendChild(e);
          });
        }
      }
    })
    .catch(() => {
      const pill = $('status');
      pill.classList.add('error');
      pill.querySelector('.status-text').textContent = 'Offline';
    });
}

/* ═══════════════════════════════════════════════════════════
   BOOT
═══════════════════════════════════════════════════════════ */
export function initUI() {
  initTabs();
  initRipple();
  initDropZone();
  initActions();
  initBatch();
  initStatus();
  initGallery();
}