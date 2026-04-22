/**
 * animation.js
 * ─────────────────────────────────────────────────────────────────────────────
 * ONE requestAnimationFrame loop drives everything:
 *   • cursor (3-layer custom cursor)
 *   • drift streaks canvas
 *   • caustic blobs + cursor ripples canvas
 *   • bubble pool canvas
 *   • creature canvas (fish, jellyfish, manta, shark, school, octopus)
 *
 * All canvases share a single rAF tick.  No setInterval anywhere.
 * ─────────────────────────────────────────────────────────────────────────────
 */

export function initAnimation() {

  /* ═══════════════════════════════════════════════════════════
     SHARED STATE
  ═══════════════════════════════════════════════════════════ */
  const cursor = { x: innerWidth / 2, y: innerHeight / 2, vx: 0, vy: 0 };
  let   r1x = cursor.x, r1y = cursor.y;
  let   r2x = cursor.x, r2y = cursor.y;

  document.addEventListener('mousemove', e => {
    cursor.vx = e.clientX - cursor.x;
    cursor.vy = e.clientY - cursor.y;
    cursor.x  = e.clientX;
    cursor.y  = e.clientY;
  });

  // Expose for creature canvas fish-flee physics
  window._cursor = cursor;

  /* ═══════════════════════════════════════════════════════════
     CANVAS SETUP HELPER
  ═══════════════════════════════════════════════════════════ */
  function setupCanvas(id) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    let W = 0, H = 0;
    function resize() {
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();
    return { canvas, ctx, get W() { return W; }, get H() { return H; } };
  }

  /* ═══════════════════════════════════════════════════════════
     CURSOR
  ═══════════════════════════════════════════════════════════ */
  const curDot   = document.getElementById('cur-dot');
  const curRing  = document.getElementById('cur-ring');
  const curTrail = document.getElementById('cur-trail');

  function tickCursor() {
    if (!curDot) return;
    r1x += (cursor.x - r1x) * 0.14;
    r1y += (cursor.y - r1y) * 0.14;
    r2x += (cursor.x - r2x) * 0.07;
    r2y += (cursor.y - r2y) * 0.07;
    curDot.style.transform   = `translate(${cursor.x}px,${cursor.y}px) translate(-50%,-50%)`;
    curRing.style.transform  = `translate(${r1x}px,${r1y}px) translate(-50%,-50%)`;
    curTrail.style.transform = `translate(${r2x}px,${r2y}px) translate(-50%,-50%)`;
  }

  // Hover expand
  document.querySelectorAll('a,button,label,.gallery-item,.tab-btn,.file-btn,.btn,.batch-item').forEach(el => {
    el.addEventListener('mouseenter', () => document.body.classList.add('cur-hover'));
    el.addEventListener('mouseleave', () => document.body.classList.remove('cur-hover'));
  });

  /* ═══════════════════════════════════════════════════════════
     DRIFT CANVAS  (light streaks / water current)
  ═══════════════════════════════════════════════════════════ */
  const drift = setupCanvas('drift-canvas');
  const streaks = drift ? Array.from({ length: 18 }, () => ({
    x:     Math.random() * 1920,
    y:     Math.random() * 900,
    len:   60  + Math.random() * 140,
    angle: Math.PI * 0.08 + (Math.random() - 0.5) * 0.12,
    speed: 0.12 + Math.random() * 0.18,
    alpha: 0.04 + Math.random() * 0.08,
    width: 0.5  + Math.random() * 1.5,
  })) : [];

  function tickDrift() {
    if (!drift) return;
    const { ctx, W, H } = drift;
    ctx.clearRect(0, 0, W, H);
    streaks.forEach(s => {
      s.x += Math.cos(s.angle) * s.speed;
      s.y += Math.sin(s.angle) * s.speed + 0.05;
      if (s.x > W + s.len || s.y > H + 20) { s.x = -s.len; s.y = Math.random() * H; }
      const ex = s.x + Math.cos(s.angle) * s.len;
      const ey = s.y + Math.sin(s.angle) * s.len;
      const g  = ctx.createLinearGradient(s.x, s.y, ex, ey);
      g.addColorStop(0, 'transparent');
      g.addColorStop(0.5, `rgba(0,200,255,${s.alpha})`);
      g.addColorStop(1, 'transparent');
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(ex, ey);
      ctx.strokeStyle = g; ctx.lineWidth = s.width; ctx.stroke();
    });
  }

  /* ═══════════════════════════════════════════════════════════
     CAUSTIC CANVAS  (blobs + cursor ripples)
  ═══════════════════════════════════════════════════════════ */
  const caustic = setupCanvas('caustic-canvas');
  const blobs = caustic ? Array.from({ length: 14 }, () => ({
    x:  Math.random() * 1920,
    y:  Math.random() * 540,
    r:  20 + Math.random() * 60,
    ox: Math.random() * Math.PI * 2,
    oy: Math.random() * Math.PI * 2,
    sx: 0.00018 + Math.random() * 0.00015,
    sy: 0.00022 + Math.random() * 0.00018,
    ax: 30 + Math.random() * 80,
    ay: 20 + Math.random() * 60,
  })) : [];

  const ripples = [];
  let lastRippleTime = 0;
  document.addEventListener('mousemove', () => {
    const now   = performance.now();
    const speed = Math.hypot(cursor.vx, cursor.vy);
    if (speed > 8 && now - lastRippleTime > 200 && ripples.length < 6) {
      lastRippleTime = now;
      ripples.push({ x: cursor.x, y: cursor.y, r: 0, maxR: 80 + speed * 1.5, born: now, life: 1200 });
    }
  });

  function tickCaustic(now) {
    if (!caustic) return;
    const { ctx, W, H } = caustic;
    ctx.clearRect(0, 0, W, H);

    blobs.forEach(b => {
      const bx = b.x + Math.sin(now * b.sx + b.ox) * b.ax;
      const by = b.y + Math.sin(now * b.sy + b.oy) * b.ay;
      const depthFade = Math.max(0, 1 - by / (H * 0.55));
      if (depthFade <= 0) return;
      const g = ctx.createRadialGradient(bx, by, 0, bx, by, b.r);
      g.addColorStop(0, `rgba(0,210,255,${0.1 * depthFade})`);
      g.addColorStop(0.5, `rgba(0,180,220,${0.04 * depthFade})`);
      g.addColorStop(1, 'transparent');
      ctx.beginPath(); ctx.arc(bx, by, b.r, 0, Math.PI * 2);
      ctx.fillStyle = g; ctx.fill();
    });

    for (let i = ripples.length - 1; i >= 0; i--) {
      const rp = ripples[i];
      const t  = (now - rp.born) / rp.life;
      if (t >= 1) { ripples.splice(i, 1); continue; }
      rp.r = rp.maxR * t;
      const alpha = (1 - t) * 0.18;
      ctx.beginPath(); ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0,200,255,${alpha})`; ctx.lineWidth = 1.5; ctx.stroke();
      if (rp.r > 20) {
        ctx.beginPath(); ctx.arc(rp.x, rp.y, rp.r * 0.55, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0,220,255,${alpha * 0.5})`; ctx.lineWidth = 0.8; ctx.stroke();
      }
    }
  }

  /* ═══════════════════════════════════════════════════════════
     BUBBLE CANVAS  (object pool — 120 bubbles max)
  ═══════════════════════════════════════════════════════════ */
  const bubbleC = setupCanvas('bubble-canvas');
  const POOL_SIZE = 120;

  function makeBubble(ambient, H) {
    return {
      active: true, ambient,
      x:     Math.random() * 1920,
      y:     ambient ? Math.random() * (H || 900) : 0,
      r:     ambient ? 1 + Math.random() * 3 : 0.5 + Math.random() * 1.8,
      speed: ambient ? 0.15 + Math.random() * 0.35 : 0.3 + Math.random() * 0.5,
      drift: (Math.random() - 0.5) * 0.2,
      alpha: ambient ? 0.06 + Math.random() * 0.16 : 0.15 + Math.random() * 0.2,
      life:  ambient ? Infinity : 1500 + Math.random() * 1000,
      born:  performance.now(),
      wobble: Math.random() * Math.PI * 2,
      wobbleSpeed: 0.01 + Math.random() * 0.018,
    };
  }

  const pool = Array.from({ length: POOL_SIZE }, () => makeBubble(true, window.innerHeight));

  // Fish trail bubble injection (called by creature canvas)
  window._spawnFishBubble = (x, y, size) => {
    const b = pool.find(b => !b.active || (!b.ambient && (performance.now() - b.born) > b.life));
    if (!b) return;
    Object.assign(b, {
      active: true, ambient: false,
      x, y,
      r:     0.4 + size * 0.025,
      speed: 0.25 + Math.random() * 0.4,
      drift: (Math.random() - 0.5) * 0.3,
      alpha: 0.12 + Math.random() * 0.18,
      life:  1200 + Math.random() * 800,
      born:  performance.now(),
      wobble: Math.random() * Math.PI * 2,
      wobbleSpeed: 0.015 + Math.random() * 0.02,
    });
  };

  function tickBubbles(now) {
    if (!bubbleC) return;
    const { ctx, W, H } = bubbleC;
    ctx.clearRect(0, 0, W, H);
    pool.forEach(b => {
      if (!b.active) return;
      if (!b.ambient) {
        const age = (now - b.born) / b.life;
        if (age >= 1) { b.active = false; return; }
        b.alpha = 0.18 * (1 - age);
        b.r    *= 0.9985;
      }
      b.y       -= b.speed;
      b.x       += b.drift + Math.sin(b.wobble) * 0.25;
      b.wobble  += b.wobbleSpeed;
      if (b.ambient && b.y < -10) { b.y = H + 10; b.x = Math.random() * W; }
      const a = Math.max(0, b.alpha);
      ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(120,220,255,${a})`; ctx.lineWidth = 0.5; ctx.stroke();
      ctx.beginPath(); ctx.arc(b.x - b.r * 0.28, b.y - b.r * 0.28, b.r * 0.32, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(220,245,255,${a * 0.55})`; ctx.fill();
    });
  }

  /* ═══════════════════════════════════════════════════════════
     CREATURE CANVAS
  ═══════════════════════════════════════════════════════════ */
  const creatureC = setupCanvas('creature-canvas');

  const PALETTES = {
    fish:      ['#1e8aa8','#137896','#22a0a0','#0e8890','#1a7a9a','#2a9eae'],
    jellyfish: ['#1a5a88','#1a6a9a','#0e5a80','#183070'],
    manta:     ['#0f4e6a','#136080','#0d4460'],
    shark:     ['#1e6680','#1a4e66','#0e5070'],
    school:    ['#1aa8c0','#0e8aaa','#22a0b8'],
    octopus:   ['#3a4e9a','#2a4888','#1a3880'],
  };

  const LAYER_PROPS = [
    { alphaScale: 0.55, speedScale: 0.65, sizeScale: 0.65 },
    { alphaScale: 0.80, speedScale: 0.85, sizeScale: 0.85 },
    { alphaScale: 1.0,  speedScale: 1.0,  sizeScale: 1.0  },
  ];

  function lerp(a, b, t) { return a + (b - a) * t; }
  function lighten(hex, amt) {
    const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b2 = parseInt(hex.slice(5, 7), 16);
    return `rgba(${Math.min(255, r + amt)},${Math.min(255, g + amt)},${Math.min(255, b2 + amt)},1)`;
  }

  // ── Draw primitives (unchanged from original) ──────────────────────────────
  function drawFish(ctx, x, y, size, color, facingLeft, alpha, phase, tilt) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.translate(x, y);
    if (facingLeft) ctx.scale(-1, 1);
    ctx.rotate(tilt * (facingLeft ? -1 : 1));
    const sw = Math.sin(phase) * size * 0.25;
    const bg = ctx.createRadialGradient(size * 0.1, 0, size * 0.05, 0, 0, size * 0.65);
    bg.addColorStop(0, lighten(color, 45)); bg.addColorStop(1, color);
    ctx.beginPath(); ctx.ellipse(0, 0, size * 0.55, size * 0.23, 0, 0, Math.PI * 2);
    ctx.fillStyle = bg; ctx.fill();
    ctx.beginPath(); ctx.moveTo(-size * 0.48, 0);
    ctx.lineTo(-size * 0.9, -size * 0.25 + sw); ctx.lineTo(-size * 0.82, 0);
    ctx.lineTo(-size * 0.9, size * 0.25 + sw); ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
    ctx.beginPath(); ctx.moveTo(-size * 0.05, -size * 0.22);
    ctx.quadraticCurveTo(size * 0.12, -size * 0.42, size * 0.25, -size * 0.22); ctx.closePath();
    ctx.fillStyle = lighten(color, 22); ctx.fill();
    ctx.beginPath(); ctx.moveTo(size * 0.05, size * 0.15);
    ctx.quadraticCurveTo(size * 0.2, size * 0.38, size * 0.3, size * 0.18); ctx.closePath();
    ctx.fillStyle = lighten(color, 16); ctx.fill();
    ctx.beginPath(); ctx.ellipse(size * 0.05, size * 0.05, size * 0.3, size * 0.1, 0, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.09)'; ctx.fill();
    ctx.beginPath(); ctx.arc(size * 0.3, -size * 0.04, size * 0.065, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,15,30,0.88)'; ctx.fill();
    ctx.beginPath(); ctx.arc(size * 0.32, -size * 0.06, size * 0.022, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.65)'; ctx.fill();
    ctx.restore();
  }

  function drawJellyfish(ctx, x, y, size, color, alpha, phase, currentX) {
    ctx.save(); ctx.globalAlpha = alpha;
    const bob = Math.sin(phase * 0.7) * size * 0.06;
    ctx.translate(x, y + bob);
    const bg = ctx.createRadialGradient(0, -size * 0.15, 0, 0, 0, size * 0.6);
    bg.addColorStop(0, lighten(color, 55)); bg.addColorStop(0.6, color); bg.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath(); ctx.ellipse(0, 0, size * 0.52, size * 0.38, 0, Math.PI, Math.PI * 2);
    ctx.fillStyle = bg; ctx.fill();
    ctx.beginPath(); ctx.ellipse(0, 0, size * 0.52, size * 0.06, 0, 0, Math.PI);
    ctx.fillStyle = lighten(color, 32); ctx.fill();
    for (let i = 0; i < 7; i++) {
      const tx = (i / 6 - 0.5) * size * 0.9;
      const swayX = Math.sin(phase + i * 0.9) * size * 0.14 + currentX * size * 0.08;
      const length = size * (0.55 + Math.sin(phase + i) * 0.2);
      ctx.beginPath(); ctx.moveTo(tx, size * 0.02);
      ctx.bezierCurveTo(tx + swayX * 0.5, size * 0.25, tx + swayX, size * 0.5, tx + swayX * 1.2, length);
      ctx.strokeStyle = lighten(color, 22); ctx.lineWidth = 1.3;
      ctx.globalAlpha = alpha * 0.55; ctx.stroke(); ctx.globalAlpha = alpha;
    }
    ctx.restore();
  }

  function drawManta(ctx, x, y, size, color, facingLeft, alpha, phase) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.translate(x, y + Math.sin(phase * 0.6) * size * 0.04);
    if (facingLeft) ctx.scale(-1, 1);
    const flap = Math.sin(phase) * size * 0.14;
    const bg = ctx.createRadialGradient(0, 0, 0, 0, 0, size * 0.8);
    bg.addColorStop(0, lighten(color, 38)); bg.addColorStop(1, color);
    ctx.beginPath();
    ctx.moveTo(size * 0.65, 0);
    ctx.bezierCurveTo(size * 0.3, -size * 0.15, -size * 0.1, -size * 0.55 + flap, -size * 0.65, -size * 0.06);
    ctx.bezierCurveTo(-size * 0.4, 0, -size * 0.1, size * 0.02, size * 0.65, 0);
    ctx.moveTo(size * 0.65, 0);
    ctx.bezierCurveTo(size * 0.3, size * 0.15, -size * 0.1, size * 0.55 - flap, -size * 0.65, size * 0.06);
    ctx.bezierCurveTo(-size * 0.4, 0, -size * 0.1, -size * 0.02, size * 0.65, 0);
    ctx.fillStyle = bg; ctx.fill();
    ctx.restore();
  }

  function drawShark(ctx, x, y, size, color, facingLeft, alpha, phase) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.translate(x, y + Math.sin(phase * 0.5) * size * 0.02);
    if (facingLeft) ctx.scale(-1, 1);
    const sw = Math.sin(phase * 1.4) * size * 0.14;
    const bg = ctx.createLinearGradient(-size * 0.8, -size * 0.25, size * 0.8, size * 0.25);
    bg.addColorStop(0, lighten(color, 28)); bg.addColorStop(0.5, color); bg.addColorStop(1, lighten(color, 12));
    ctx.beginPath();
    ctx.moveTo(size * 0.75, 0);
    ctx.bezierCurveTo(size * 0.5, -size * 0.22, -size * 0.4, -size * 0.22, -size * 0.7, 0);
    ctx.bezierCurveTo(-size * 0.4, size * 0.15, size * 0.5, size * 0.15, size * 0.75, 0);
    ctx.fillStyle = bg; ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-size * 0.65, 0); ctx.lineTo(-size * 1.02, -size * 0.28 + sw);
    ctx.lineTo(-size * 0.88, 0); ctx.lineTo(-size * 1.02, size * 0.2 + sw); ctx.closePath();
    ctx.fillStyle = lighten(color, 16); ctx.fill();
    ctx.beginPath(); ctx.moveTo(size * 0.05, -size * 0.2);
    ctx.bezierCurveTo(size * 0.12, -size * 0.52, size * 0.32, -size * 0.52, size * 0.38, -size * 0.2); ctx.closePath();
    ctx.fillStyle = lighten(color, 11); ctx.fill();
    ctx.beginPath(); ctx.arc(size * 0.52, -size * 0.06, size * 0.04, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,10,20,0.9)'; ctx.fill();
    ctx.restore();
  }

  function drawSchool(ctx, cx, cy, color, facingLeft, alpha, phase, tilt) {
    [[0, 0, 1], [55, -18, 0.9], [55, 20, 0.9], [110, 0, 0.85], [110, -35, 0.8], [110, 38, 0.8], [165, -18, 0.75], [165, 18, 0.75]]
      .forEach(([ox, oy, sc], i) => {
        const waveY = Math.sin(phase + i * 0.4) * 5;
        drawFish(ctx, cx + (facingLeft ? ox : -ox), cy + oy + waveY, 16 * sc, color, facingLeft, alpha * 0.85, phase + i * 0.25, tilt * 0.5);
      });
  }

  function drawOctopus(ctx, x, y, size, color, alpha, phase, currentX) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.translate(x, y + Math.sin(phase * 0.4) * size * 0.06);
    const mg = ctx.createRadialGradient(0, -size * 0.35, size * 0.05, 0, -size * 0.2, size * 0.6);
    mg.addColorStop(0, lighten(color, 40)); mg.addColorStop(0.4, lighten(color, 18)); mg.addColorStop(1, color);
    ctx.beginPath();
    ctx.moveTo(0, -size * 0.65);
    ctx.bezierCurveTo(size * 0.55, -size * 0.55, size * 0.45, -size * 0.05, 0, size * 0.05);
    ctx.bezierCurveTo(-size * 0.45, -size * 0.05, -size * 0.55, -size * 0.55, 0, -size * 0.65);
    ctx.fillStyle = mg; ctx.fill();
    for (let i = 0; i < 8; i++) {
      const spread = (i - 3.5) * 0.22;
      const sway = Math.sin(phase + i * 0.6) * size * 0.12 + currentX * size * 0.06;
      const startX = spread * size * 0.25, startY = size * 0.05;
      const endX = startX + sway * 1.3, endY = size * 0.9 + Math.sin(phase + i) * size * 0.12;
      ctx.beginPath(); ctx.moveTo(startX, startY);
      ctx.quadraticCurveTo(startX + sway, size * 0.55 + Math.sin(phase * 0.7 + i) * size * 0.08, endX, endY);
      ctx.strokeStyle = lighten(color, 18); ctx.lineWidth = size * 0.12 * Math.max(0, 1 - i * 0.05);
      ctx.lineCap = 'round'; ctx.globalAlpha = alpha; ctx.stroke();
    }
    ctx.restore();
  }

  // ── Creature pool  (18 creatures, replaced one-for-one as they exit) ────────
  function spawnCreature(W, H) {
    const types = ['fish','fish','fish','fish','fish','jellyfish','jellyfish','manta','shark','school','school','octopus'];
    const type  = types[Math.floor(Math.random() * types.length)];
    const fromLeft = Math.random() > 0.5;
    let layer = (type === 'shark' || type === 'manta') ? 2
              : (type === 'jellyfish' || type === 'octopus') ? Math.floor(Math.random() * 2) + 1
              : Math.floor(Math.random() * 3);
    const lp = LAYER_PROPS[layer];
    const baseSize = type === 'shark' ? 60 + Math.random() * 35
                   : type === 'manta' ? 50 + Math.random() * 35
                   : type === 'octopus' ? 38 + Math.random() * 28
                   : type === 'jellyfish' ? 30 + Math.random() * 28
                   : type === 'school' ? 22 + Math.random() * 10
                   : 24 + Math.random() * 32;
    const size  = baseSize * lp.sizeScale;
    const y     = H * 0.08 + Math.random() * (H * 0.84);
    const speedPx = (22 + Math.random() * 28) * lp.speedScale;
    const distance = W + size * 5;
    const duration = (distance / speedPx) * 1000;
    const pal   = PALETTES[type] || PALETTES.fish;
    const color = pal[Math.floor(Math.random() * pal.length)];
    return {
      type, fromLeft, layer, y, duration, size, color,
      alpha: (0.6 + Math.random() * 0.3) * lp.alphaScale,
      startX: fromLeft ? -size * 3 : W + size * 3,
      endX:   fromLeft ? W + size * 3 : -size * 3,
      startTime: performance.now(),
      phase: Math.random() * Math.PI * 2,
      phaseSpeed: 0.012 + Math.random() * 0.022,
      waveAmp:  8  + Math.random() * 22,
      waveFreq: 0.00015 + Math.random() * 0.00025,
      baseSpeed: speedPx,
      px: fromLeft ? -size * 3 : (W || 1400) + size * 3,
      py: y,
      vx: (fromLeft ? 1 : -1) * speedPx / 60,
      vy: 0,
      targetVx: (fromLeft ? 1 : -1) * speedPx / 60,
      targetVy: 0,
      tilt: 0,
      fleePhase: 0,
      bubbleTimer: 0,
    };
  }

  let parallaxX = 0, parallaxY = 0;
  const creatures = [];
  const initW = window.innerWidth, initH = window.innerHeight;
  for (let i = 0; i < 18; i++) {
    const c = spawnCreature(initW, initH);
    const frac = Math.random();
    c.px = c.startX + (c.endX - c.startX) * frac;
    c.py = c.y;
    c.startTime = performance.now() - frac * c.duration;
    creatures.push(c);
  }

  function tickCreatures(now) {
    if (!creatureC) return;
    const { ctx, W, H } = creatureC;
    ctx.clearRect(0, 0, W, H);

    parallaxX = lerp(parallaxX, (cursor.x / W - 0.5) * 18, 0.04);
    parallaxY = lerp(parallaxY, (cursor.y / H - 0.5) * 10, 0.04);

    creatures.sort((a, b) => a.layer - b.layer);

    for (let i = creatures.length - 1; i >= 0; i--) {
      const c = creatures[i];
      const t = (now - c.startTime) / c.duration;
      if (t >= 1) { creatures.splice(i, 1); creatures.push(spawnCreature(W, H)); continue; }

      const baseX = c.startX + (c.endX - c.startX) * t;
      const baseY = c.y + Math.sin(now * c.waveFreq) * c.waveAmp;

      const dx = c.px - cursor.x, dy = c.py - cursor.y;
      const dist = Math.hypot(dx, dy) || 1;
      const detectR = c.size * (c.type === 'shark' ? 6 : c.type === 'manta' ? 8 : 9);

      if (dist < detectR) {
        const strength = Math.pow(1 - dist / detectR, 1.8) * 3.5;
        const nx = dx / dist, ny = dy / dist;
        c.targetVx = lerp(c.targetVx, (c.fromLeft ? 1 : -1) * c.baseSpeed / 60 + nx * c.baseSpeed * 0.06 * strength, 0.12);
        c.targetVy = lerp(c.targetVy, ny * c.baseSpeed * 0.04 * strength, 0.10);
        c.fleePhase = Math.min(1, c.fleePhase + 0.08);
      } else {
        c.targetVx = lerp(c.targetVx, (c.fromLeft ? 1 : -1) * c.baseSpeed / 60, 0.03);
        c.targetVy = lerp(c.targetVy, 0, 0.025);
        c.fleePhase = Math.max(0, c.fleePhase - 0.02);
      }

      c.vx = lerp(c.vx, c.targetVx, 0.08);
      c.vy = lerp(c.vy, c.targetVy, 0.06);
      const pw = Math.min(1, c.fleePhase * 2);
      c.px = lerp(baseX, c.px + c.vx, pw * 0.3);
      c.py = lerp(baseY, Math.max(H * 0.05, Math.min(H * 0.95, c.py + c.vy)), pw * 0.3);
      c.tilt = lerp(c.tilt, c.vy * 0.018, 0.1);
      c.phase += c.phaseSpeed * (1 + c.fleePhase * 1.5);

      // Bubble trails
      if (window._spawnFishBubble && (c.type === 'fish' || c.type === 'school')) {
        c.bubbleTimer++;
        const rate = c.fleePhase > 0.3 ? 3 : c.size > 30 ? 8 : 18;
        if (c.bubbleTimer % rate === 0 && Math.random() < 0.6) {
          const tailOff = c.fromLeft ? -c.size * 0.55 : c.size * 0.55;
          window._spawnFishBubble(c.px + tailOff, c.py, c.size);
        }
      }

      let edgeFade = 1;
      if (t < 0.04) edgeFade = t / 0.04;
      if (t > 0.96) edgeFade = (1 - t) / 0.04;

      const pScale = [0.8, 0.4, 0.15][c.layer];
      const rx = c.px + parallaxX * pScale;
      const ry = c.py + parallaxY * pScale;

      ctx.save(); ctx.globalAlpha = edgeFade;
      switch (c.type) {
        case 'fish':      drawFish(ctx, rx, ry, c.size, c.color, !c.fromLeft, c.alpha, c.phase, c.tilt); break;
        case 'jellyfish': drawJellyfish(ctx, rx, ry, c.size, c.color, c.alpha, c.phase, cursor.vx * 0.001); break;
        case 'manta':     drawManta(ctx, rx, ry, c.size, c.color, !c.fromLeft, c.alpha, c.phase); break;
        case 'shark':     drawShark(ctx, rx, ry, c.size, c.color, !c.fromLeft, c.alpha, c.phase); break;
        case 'school':    drawSchool(ctx, rx, ry, c.color, !c.fromLeft, c.alpha, c.phase, c.tilt); break;
        case 'octopus':   drawOctopus(ctx, rx, ry, c.size, c.color, c.alpha, c.phase, cursor.vx * 0.0008); break;
      }
      ctx.restore();
    }
  }

  /* ═══════════════════════════════════════════════════════════
     MASTER rAF LOOP  — single tick drives everything
  ═══════════════════════════════════════════════════════════ */
  function tick(now) {
    tickCursor();
    tickDrift();
    tickCaustic(now);
    tickBubbles(now);
    tickCreatures(now);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}