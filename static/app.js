'use strict';

/* ── Color data ─────────────────────────────────────────────────────────────── */
const HEX = {
  black:  '#1a1a1a', brown:  '#8B4513', red:    '#CC2200',
  orange: '#FF7700', yellow: '#FFD700', green:  '#2A7A2A',
  blue:   '#1560B8', violet: '#7B1FA2', gray:   '#757575',
  white:  '#F2F2F2', gold:   '#D4A017', silver: '#A8A9AD',
  none:   'transparent',
};

const DIGIT_MAP    = {black:0,brown:1,red:2,orange:3,yellow:4,green:5,blue:6,violet:7,gray:8,white:9};
const MULT_MAP     = {black:1,brown:10,red:100,orange:1e3,yellow:1e4,green:1e5,blue:1e6,violet:1e7,gray:1e8,white:1e9,gold:0.1,silver:0.01};
const TOL_MAP      = {brown:'±1%',red:'±2%',green:'±0.5%',blue:'±0.25%',violet:'±0.1%',gray:'±0.05%',gold:'±5%',silver:'±10%',none:'±20%'};
const TEMPCO_MAP   = {black:'250 ppm/K',brown:'100 ppm/K',red:'50 ppm/K',orange:'15 ppm/K',yellow:'25 ppm/K',green:'20 ppm/K',blue:'10 ppm/K',violet:'5 ppm/K',gray:'1 ppm/K'};

/* Colors available per band type */
const COLORS_BY_TYPE = {
  digit:      ['black','brown','red','orange','yellow','green','blue','violet','gray','white'],
  multiplier: ['black','brown','red','orange','yellow','green','blue','violet','gray','white','gold','silver'],
  tolerance:  ['brown','red','green','blue','violet','gray','gold','silver','none'],
  tempco:     ['black','brown','red','orange','yellow','green','blue','violet','gray'],
};

/* Band configs per band count */
const BAND_CONFIGS = {
  3: [
    {label:'1st Digit',   type:'digit'},
    {label:'2nd Digit',   type:'digit'},
    {label:'Multiplier',  type:'multiplier'},
  ],
  4: [
    {label:'1st Digit',   type:'digit'},
    {label:'2nd Digit',   type:'digit'},
    {label:'Multiplier',  type:'multiplier'},
    {label:'Tolerance',   type:'tolerance'},
  ],
  5: [
    {label:'1st Digit',   type:'digit'},
    {label:'2nd Digit',   type:'digit'},
    {label:'3rd Digit',   type:'digit'},
    {label:'Multiplier',  type:'multiplier'},
    {label:'Tolerance',   type:'tolerance'},
  ],
  6: [
    {label:'1st Digit',   type:'digit'},
    {label:'2nd Digit',   type:'digit'},
    {label:'3rd Digit',   type:'digit'},
    {label:'Multiplier',  type:'multiplier'},
    {label:'Tolerance',   type:'tolerance'},
    {label:'Temp Coeff.', type:'tempco'},
  ],
};

/* Default color selections */
const DEFAULTS = {
  digit:       'brown',
  multiplier:  'red',
  tolerance:   'gold',
  tempco:      'brown',
};

/* SVG band x-positions [x, width] per band count */
const SVG_BANDS = {
  3: [[90,20],[128,20],[166,20]],
  4: [[82,16],[110,16],[152,16],[218,16]],
  5: [[76,14],[100,14],[124,14],[158,14],[220,14]],
  6: [[72,12],[92,12],[112,12],[146,12],[202,12],[224,12]],
};

/* ── Formatting ──────────────────────────────────────────────────────────────── */
function formatOhms(v) {
  if (v === 0) return {val:'0', unit:'Ω'};
  const abs = Math.abs(v);
  if (abs < 1000)  return {val: +v.toPrecision(3) + '',         unit:'Ω'};
  if (abs < 1e6)   return {val: +(v / 1e3).toPrecision(3) + '', unit:'kΩ'};
  if (abs < 1e9)   return {val: +(v / 1e6).toPrecision(3) + '', unit:'MΩ'};
  return               {val: +(v / 1e9).toPrecision(3) + '',    unit:'GΩ'};
}

/* ── DOM refs ────────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const tabs        = document.querySelectorAll('.tab');
const tabPanels   = document.querySelectorAll('.tab-panel');
const dropzone    = $('dropzone');
const fileInput   = $('fileInput');
const selectBtn   = $('selectBtn');
const cameraBtn   = $('cameraBtn');
const previewArea = $('previewArea');
const previewCanvas = $('previewCanvas');
const videoEl     = $('videoEl');
const clearBtn    = $('clearBtn');
const cameraHint  = $('cameraHint');
const analyzeBtn  = $('analyzeBtn');
const analyzeTxt  = $('analyzeTxt');
const analyzeSpinner = $('analyzeSpinner');
const resultPanel = $('resultPanel');
const rpValue     = $('rpValue');
const rpChips     = $('rpChips');
const rpBands     = $('rpBands');
const rpConfidence = $('rpConfidence');
const errorPanel  = $('errorPanel');
const errorMsg    = $('errorMsg');

/* Manual */
const bandCountSeg   = $('bandCountSeg');
const bandPickers    = $('bandPickers');
const manualResult   = $('manualResult');
const mrValue        = $('mrValue');
const mrMeta         = $('mrMeta');
const svgBands       = $('svgBands');

/* Gemini toggle */
const geminiToggle   = $('geminiToggle');
const useGeminiCb    = $('useGemini');

/* ── State ───────────────────────────────────────────────────────────────────── */
let cameraStream    = null;
let currentBlob     = null;   // blob to send to /analyze
let imageState      = 'empty'; // 'empty' | 'file' | 'camera' | 'analyzing'
let manualBandCount = 4;
let manualSelections = {};    // role index → color name

// Preview zoom/pan state (manual wheel zoom)
const ZOOM_MIN = 1;
const ZOOM_MAX = 5;
const ZOOM_FACTOR = 1.12;
let previewZoom = 1;
let previewPanX = 0;
let previewPanY = 0;

/* ── Tab switching ───────────────────────────────────────────────────────────── */
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    tabs.forEach(t => { t.classList.toggle('active', t.dataset.tab === target); t.setAttribute('aria-selected', t.dataset.tab === target); });
    tabPanels.forEach(p => p.classList.add('hidden'));
    $(`tab-${target}`).classList.remove('hidden');
    hideResult();
    hideError();
  });
});

/* ── Image mode: state transitions ─────────────────────────────────────────── */
function setState(s) {
  imageState = s;
  const isEmpty    = s === 'empty';
  const hasContent = s === 'file' || s === 'camera';
  const analyzing  = s === 'analyzing';

  // Dropzone vs preview
  dropzone.classList.toggle('hidden', !isEmpty);
  previewArea.classList.toggle('hidden', isEmpty);

  // Video vs canvas
  if (s === 'camera') {
    videoEl.classList.remove('hidden');
    previewCanvas.classList.add('hidden');
    cameraHint.classList.remove('hidden');
  } else {
    videoEl.classList.add('hidden');
    previewCanvas.classList.remove('hidden');
    cameraHint.classList.add('hidden');
  }

  // Analyze button
  analyzeBtn.classList.toggle('hidden', isEmpty);
  analyzeBtn.disabled = !hasContent && !analyzing;

  // Gemini toggle (show when there's content)
  geminiToggle.classList.toggle('hidden', isEmpty);

  if (analyzing) {
    analyzeBtn.classList.add('loading');
    analyzeTxt.textContent = 'Analyzing…';
    analyzeSpinner.classList.remove('hidden');
  } else {
    analyzeBtn.classList.remove('loading');
    analyzeTxt.textContent = s === 'camera' ? 'Capture & Analyze' : 'Analyze Bands';
    analyzeSpinner.classList.add('hidden');
    analyzeBtn.disabled = !hasContent;
  }
}

function showResult(data) {
  const {val, unit} = formatOhms(data.ohms);
  rpValue.textContent = `${val} ${unit}`;

  rpChips.innerHTML = '';
  if (data.tolerance) rpChips.innerHTML += `<span class="chip tol">${data.tolerance}</span>`;
  if (data.tempco)    rpChips.innerHTML += `<span class="chip">${data.tempco}</span>`;
  if (data.band_count) rpChips.innerHTML += `<span class="chip">${data.band_count}-band</span>`;

  rpBands.innerHTML = data.bands.map(c =>
    `<div class="rp-band">
      <div class="rp-swatch" style="background:${HEX[c] ?? '#555'};border-color:${c==='white'?'rgba(255,255,255,0.3)':'rgba(255,255,255,0.08)'}"></div>
      <div class="rp-name">${c}</div>
    </div>`
  ).join('');

  const conf = data.confidence || 'medium';
  rpConfidence.innerHTML = `Model confidence: <span class="conf-${conf}">${conf}</span>`;

  resultPanel.classList.remove('hidden');
  hideError();
}

function hideResult() {
  resultPanel.classList.add('hidden');
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorPanel.classList.remove('hidden');
  hideResult();
}

function hideError() {
  errorPanel.classList.add('hidden');
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function getActivePreviewEl() {
  return imageState === 'camera' ? videoEl : previewCanvas;
}

function clampPreviewPan() {
  const mediaEl = getActivePreviewEl();
  const maxX = (mediaEl.offsetWidth * (previewZoom - 1)) / 2;
  const maxY = (mediaEl.offsetHeight * (previewZoom - 1)) / 2;
  previewPanX = clamp(previewPanX, -maxX, maxX);
  previewPanY = clamp(previewPanY, -maxY, maxY);
}

function applyPreviewTransform() {
  const transform = `translate(${previewPanX}px, ${previewPanY}px) scale(${previewZoom})`;
  previewCanvas.style.transform = transform;
  videoEl.style.transform = transform;
}

function resetPreviewZoom() {
  previewZoom = 1;
  previewPanX = 0;
  previewPanY = 0;
  applyPreviewTransform();
}

previewArea.addEventListener('wheel', e => {
  if (imageState !== 'file' && imageState !== 'camera') return;
  e.preventDefault();

  const rect = previewArea.getBoundingClientRect();
  const cursorX = e.clientX - rect.left - rect.width / 2;
  const cursorY = e.clientY - rect.top - rect.height / 2;

  const prevZoom = previewZoom;
  const nextZoom = e.deltaY < 0 ? prevZoom * ZOOM_FACTOR : prevZoom / ZOOM_FACTOR;
  previewZoom = clamp(nextZoom, ZOOM_MIN, ZOOM_MAX);

  if (previewZoom !== prevZoom) {
    const ratio = previewZoom / prevZoom;
    previewPanX = cursorX - ratio * (cursorX - previewPanX);
    previewPanY = cursorY - ratio * (cursorY - previewPanY);
    clampPreviewPan();
    applyPreviewTransform();
  }
}, {passive: false});

previewArea.addEventListener('dblclick', () => {
  if (imageState === 'file' || imageState === 'camera') resetPreviewZoom();
});

window.addEventListener('resize', () => {
  if (imageState !== 'file' && imageState !== 'camera') return;
  clampPreviewPan();
  applyPreviewTransform();
});

function resetImageMode() {
  stopCamera();
  currentBlob = null;
  hideResult();
  hideError();
  resetPreviewZoom();
  const ctx = previewCanvas.getContext('2d');
  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  fileInput.value = '';
  setState('empty');
}

/* ── File selection ─────────────────────────────────────────────────────────── */
function loadFile(file) {
  if (!file) return;
  currentBlob = file;
  const img = new Image();
  img.onload = () => {
    const dpr = window.devicePixelRatio || 1;
    const maxW = Math.min(900, previewArea.parentElement.clientWidth - 2);
    const scale = Math.min(1, maxW / img.width);
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    previewCanvas.style.width  = w + 'px';
    previewCanvas.style.height = h + 'px';
    previewCanvas.width  = Math.round(w * dpr);
    previewCanvas.height = Math.round(h * dpr);
    const ctx = previewCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.drawImage(img, 0, 0, w, h);
    URL.revokeObjectURL(img.src);
    resetPreviewZoom();
    setState('file');
    hideResult();
    hideError();
  };
  img.src = URL.createObjectURL(file);
}

selectBtn.addEventListener('click', () => {
  stopCamera();
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files?.[0]) loadFile(fileInput.files[0]);
});

/* Drag and drop */
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', ()  => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) loadFile(file);
});
dropzone.addEventListener('click', e => { if (e.target === dropzone || e.target.closest('#dropzone') && !e.target.closest('button')) fileInput.click(); });
dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });

/* ── Camera ──────────────────────────────────────────────────────────────────── */
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: {ideal: 'environment'}, width: {ideal: 1280}, height: {ideal: 720} }
    });
    videoEl.srcObject = cameraStream;
    await videoEl.play();
    resetPreviewZoom();
    setState('camera');
    hideResult();
    hideError();
  } catch (err) {
    showError(`Camera unavailable: ${err.message}`);
  }
}

function stopCamera() {
  if (!cameraStream) return;
  cameraStream.getTracks().forEach(t => t.stop());
  cameraStream = null;
  videoEl.srcObject = null;
}

cameraBtn.addEventListener('click', async () => {
  if (cameraStream) {
    stopCamera();
    resetImageMode();
    return;
  }
  await startCamera();
});

clearBtn.addEventListener('click', resetImageMode);

/* ── Capture frame from live video ─────────────────────────────────────────── */
function captureVideoFrame() {
  return new Promise(resolve => {
    const w = videoEl.videoWidth;
    const h = videoEl.videoHeight;
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    off.getContext('2d').drawImage(videoEl, 0, 0, w, h);
    off.toBlob(resolve, 'image/jpeg', 0.9);
  });
}

/* ── Analyze ──────────────────────────────────────────────────────────────────── */
analyzeBtn.addEventListener('click', async () => {
  let blob = currentBlob;

  // Camera mode: capture current frame first
  if (imageState === 'camera') {
    blob = await captureVideoFrame();
    // Show snapshot
    const img = new Image();
    img.onload = () => {
      const dpr = window.devicePixelRatio || 1;
      const maxW = Math.min(900, previewArea.parentElement.clientWidth - 2);
      const scale = Math.min(1, maxW / img.width);
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      previewCanvas.style.width  = w + 'px';
      previewCanvas.style.height = h + 'px';
      previewCanvas.width  = Math.round(w * dpr);
      previewCanvas.height = Math.round(h * dpr);
      const ctx = previewCanvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.drawImage(img, 0, 0, w, h);
      resetPreviewZoom();
    };
    img.src = URL.createObjectURL(blob);
    stopCamera();
    // Show canvas instead of video (setState handles this)
  }

  if (!blob) return;
  setState('analyzing');
  hideResult();
  hideError();

  try {
    const fd = new FormData();
    fd.append('image', blob, 'resistor.jpg');
    if (useGeminiCb.checked) fd.append('use_gemini', 'true');
    const res = await fetch('/analyze', {method: 'POST', body: fd});
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    setState('file');
    showResult(data);
  } catch (err) {
    setState('file');
    showError(err.message);
  }
});

/* ── Manual mode ─────────────────────────────────────────────────────────────── */
function buildPickers(n) {
  manualBandCount = n;
  const config = BAND_CONFIGS[n];

  // Initialize selections with defaults
  manualSelections = {};
  config.forEach((band, i) => {
    manualSelections[i] = DEFAULTS[band.type];
  });

  // Render picker rows
  bandPickers.innerHTML = '';
  config.forEach((band, i) => {
    const colors = COLORS_BY_TYPE[band.type];
    const row = document.createElement('div');
    row.className = 'band-row';
    row.innerHTML = `<div class="band-row-label">${band.label}</div><div class="swatches" data-band="${i}"></div>`;
    const swatchContainer = row.querySelector('.swatches');

    colors.forEach(color => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'swatch-btn' + (color === 'none' ? ' swatch-none' : '');
      btn.dataset.c = color;
      btn.dataset.band = i;
      btn.title = color;
      btn.setAttribute('aria-label', color);
      if (color !== 'none') {
        btn.style.background = HEX[color] || '#555';
      } else {
        btn.textContent = '—';
      }
      if (manualSelections[i] === color) btn.classList.add('sel');
      btn.addEventListener('click', () => {
        manualSelections[i] = color;
        // Update selection UI
        swatchContainer.querySelectorAll('.swatch-btn').forEach(b => b.classList.remove('sel'));
        btn.classList.add('sel');
        updateSvgBands();
        updateManualResult();
      });
      swatchContainer.appendChild(btn);
    });

    bandPickers.appendChild(row);
  });

  updateSvgBands();
  updateManualResult();
}

function updateSvgBands() {
  const n = manualBandCount;
  const config = BAND_CONFIGS[n];
  const positions = SVG_BANDS[n];

  svgBands.innerHTML = '';
  config.forEach((band, i) => {
    const color = manualSelections[i] ?? DEFAULTS[band.type];
    if (color === 'none') return;
    const [x, w] = positions[i];
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x);
    rect.setAttribute('y', '18');
    rect.setAttribute('width', w);
    rect.setAttribute('height', '44');
    rect.setAttribute('fill', HEX[color] || '#555');
    if (color === 'white') rect.setAttribute('fill-opacity', '0.85');
    svgBands.appendChild(rect);
  });
}

function updateManualResult() {
  const n = manualBandCount;
  const config = BAND_CONFIGS[n];
  const bands = config.map((_, i) => manualSelections[i] ?? DEFAULTS[config[i].type]);

  // Client-side resistance calculation (matches server logic)
  try {
    let ohms;
    let tol = null;
    let tempco = null;
    const d = s => DIGIT_MAP[s] ?? 0;
    const m = s => MULT_MAP[s] ?? 1;
    const t = s => TOL_MAP[s];
    const tc = s => TEMPCO_MAP[s];

    if (n === 3) {
      ohms = (d(bands[0])*10 + d(bands[1])) * m(bands[2]);
    } else if (n === 4) {
      ohms = (d(bands[0])*10 + d(bands[1])) * m(bands[2]);
      tol = t(bands[3]);
    } else if (n === 5) {
      ohms = (d(bands[0])*100 + d(bands[1])*10 + d(bands[2])) * m(bands[3]);
      tol = t(bands[4]);
    } else if (n === 6) {
      ohms = (d(bands[0])*100 + d(bands[1])*10 + d(bands[2])) * m(bands[3]);
      tol = t(bands[4]);
      tempco = tc(bands[5]);
    }

    const {val, unit} = formatOhms(ohms);
    mrValue.textContent = `${val} ${unit}`;
    const meta = [tol, tempco].filter(Boolean).join(' · ');
    mrMeta.textContent = meta;
    manualResult.classList.remove('hidden');
  } catch (e) {
    manualResult.classList.add('hidden');
  }
}

/* Band count switcher */
bandCountSeg.querySelectorAll('.seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    bandCountSeg.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    buildPickers(parseInt(btn.dataset.n, 10));
  });
});

/* ── Init ─────────────────────────────────────────────────────────────────────── */
setState('empty');
buildPickers(4);
