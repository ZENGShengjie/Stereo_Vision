// Stereo Vision — Performance Panel
// Polls /api/stream/stats every 500ms and renders gauges + sparkline + calibration.

const USB_FPS = 10;   // matches config USB_FPS default
const TARGET_MS = (1000 / USB_FPS).toFixed(1);
const POLL_INTERVAL_MS = 500;

// ── DOM refs ────────────────────────────────────────────────────────────────
const elFps        = document.getElementById("fpsVal");
const elP50        = document.getElementById("latP50");
const elP95        = document.getElementById("latP95");
const elMax        = document.getElementById("latMax");
const elStageRows  = document.getElementById("stageRows");
const elBudgetRows = document.getElementById("budgetRows");
const elChart      = document.getElementById("latChart");
const elDepthVal   = document.getElementById("depthVal");
const elDepthBadge = document.getElementById("depthBadge");
const elDMedianVal = document.getElementById("dMedianVal");
const elDMedianBadge = document.getElementById("dMedianBadge");
const elScaleVal   = document.getElementById("scaleVal");
const elScaleBadge = document.getElementById("scaleBadge");
const elConfVal    = document.getElementById("confVal");
const elConfBadge  = document.getElementById("confBadge");
const elWinInfo    = document.getElementById("winInfo");
const elTargetInfo = document.getElementById("targetInfo");
const elUpdInfo    = document.getElementById("updInfo");
const elDot        = document.getElementById("refreshDot");
const elRealCmInput = document.getElementById("realCmInput");
const elCalibrateBtn = document.getElementById("calibrateBtn");
const elCalibResult = document.getElementById("calibResult");
const elScaleTable   = document.getElementById("scaleTable");
const elScaleTbody   = document.getElementById("scaleTableBody");

// ── Sparkline canvas ────────────────────────────────────────────────────────
const ctx = elChart.getContext("2d");

function resizeChart() {
  const dpr = window.devicePixelRatio || 1;
  const w   = elChart.offsetWidth;
  const h   = elChart.offsetHeight;
  elChart.width  = w * dpr;
  elChart.height = h * dpr;
  ctx.scale(dpr, dpr);
  elChart.style.width  = w + "px";
  elChart.style.height = h + "px";
}
window.addEventListener("resize", resizeChart);
resizeChart();

// ── Colour helpers ───────────────────────────────────────────────────────────
function stageColour(ms, budget) {
  const ratio = ms / budget;
  if (ratio < 0.50) return "#4f8";
  if (ratio < 0.75) return "#f80";
  return "#f44";
}

function latencyColour(ms) {
  if (ms < 30)  return "good";
  if (ms < 60)  return "warn";
  return "bad";
}

function fpsColour(fps) {
  if (fps >= 20) return "good";
  if (fps >= 10) return "warn";
  return "bad";
}

// ── Rendering helpers ───────────────────────────────────────────────────────
function renderStageBar(name, ms, totalBudget) {
  const pct    = Math.min(100, (ms / totalBudget) * 100);
  const colour = stageColour(ms, totalBudget);
  const row = document.createElement("div");
  row.className = "stage-bar-row";
  row.innerHTML = `
    <span class="stage-name">${name}</span>
    <div class="stage-bar-bg">
      <div class="stage-bar-fill" style="width:${pct}%;background:${colour}"></div>
    </div>
    <span class="stage-ms" style="color:${colour}">${ms.toFixed(1)}</span>
  `;
  return row;
}

function renderBudgetBar(name, ms, totalBudget) {
  const usedPct = Math.min(100, (ms / totalBudget) * 100);
  const colour  = usedPct < 70 ? "#4f8" : usedPct < 90 ? "#f80" : "#f44";
  const row = document.createElement("div");
  row.className = "stage-bar-row";
  row.innerHTML = `
    <span class="stage-name">${name}</span>
    <div class="stage-bar-bg">
      <div class="stage-bar-fill" style="width:${usedPct}%;background:${colour}"></div>
    </div>
    <span class="stage-ms" style="color:${colour}">${ms.toFixed(1)} / ${totalBudget}</span>
  `;
  return row;
}

function drawSparkline(history) {
  const W = elChart.offsetWidth;
  const H = elChart.offsetHeight;
  ctx.clearRect(0, 0, W, H);
  if (!history.length) return;

  const maxVal = Math.max(...history, 1);
  const pad    = 2;
  const range  = maxVal;  // minVal=0

  // Grid lines
  ctx.strokeStyle = "#2a2a2a";
  ctx.lineWidth   = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad + (g / 4) * (H - 2 * pad);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Target line
  const targetY = pad + ((maxVal - TARGET_MS) / range) * (H - 2 * pad);
  if (targetY > pad && targetY < H - pad) {
    ctx.strokeStyle = "#6cf4"; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, targetY); ctx.lineTo(W, targetY); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Data line
  const xStep = W / Math.max(history.length - 1, 1);
  ctx.beginPath();
  ctx.strokeStyle = "#6cf"; ctx.lineWidth = 1.5;
  history.forEach((v, i) => {
    const x = i * xStep;
    const y = pad + ((maxVal - v) / range) * (H - 2 * pad);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Area fill
  ctx.lineTo((history.length - 1) * xStep, H - pad);
  ctx.lineTo(0, H - pad);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "rgba(102,204,255,0.15)");
  grad.addColorStop(1, "rgba(102,204,255,0.0)");
  ctx.fillStyle = grad;
  ctx.fill();
}

// ── Calibration ───────────────────────────────────────────────────────────────
async function runCalibration() {
  const realCm = parseFloat(elRealCmInput.value);
  if (!realCm || realCm < 0.5 || realCm > 500) {
    elCalibResult.className  = "calib-result err";
    elCalibResult.textContent = "Please enter a real distance between 0.5 and 500 cm";
    elCalibResult.hidden = false;
    return;
  }

  elCalibrateBtn.disabled = true;
  elCalibrateBtn.textContent = "Calibrating...";

  try {
    const res = await fetch("/api/calibrate/disp_scale", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ real_cm: realCm }),
    });
    const data = await res.json();

    if (!res.ok) {
      elCalibResult.className  = "calib-result err";
      elCalibResult.textContent = data.error + (data.hint ? ` (${data.hint})` : "");
      elCalibResult.hidden = false;
      return;
    }

    elScaleVal.textContent = data.new_disp_scale.toFixed(4);
    elScaleBadge.className = "depth-badge has-value";
    elCalibResult.className  = "calib-result ok";
    elCalibResult.textContent =
      `DISP_SCALE updated: ${data.old_disp_scale} → ${data.new_disp_scale} ` +
      `(real ${data.real_cm}cm → measured ${data.z_measured_cm}cm). ` +
      `Formula: ${data.formula}. Restart to persist.`;
    elCalibResult.hidden = false;
  } catch (err) {
    elCalibResult.className  = "calib-result err";
    elCalibResult.textContent = "Network error: " + err.message;
    elCalibResult.hidden = false;
  } finally {
    elCalibrateBtn.disabled = false;
    elCalibrateBtn.textContent = "Apply Calibration";
  }
}

elCalibrateBtn.addEventListener("click", runCalibration);
elRealCmInput.addEventListener("keydown", e => {
  if (e.key === "Enter") runCalibration();
});

// ── Stats polling ────────────────────────────────────────────────────────────
let lastHistory = [];

async function poll() {
  try {
    const res  = await fetch("/api/stream/stats");
    if (!res.ok) return;
    const data = await res.json();
    update(data);
  } catch (_) { /* silent */ }
}

function update(data) {
  const fps       = data.fps                || 0;
  const p50       = data.latency_p50_ms     || 0;
  const p95       = data.latency_p95_ms     || 0;
  const maxMs     = data.latency_max_ms     || 0;
  const avgMs     = data.latency_avg_ms     || 0;
  const stages    = data.stages             || {};
  const winFrames = data.frames_in_window   || 0;
  const total     = data.frames_served      || 0;
  const depthCm   = data.depth_cm;
  const dMedian   = data.d_median_px;
  const zMeasured = data.z_measured_cm;
  const scales    = data.suggested_disp_scales || {};

  // ── Top-level metrics ──────────────────────────────────────────────────
  elFps.textContent = fps.toFixed(1);
  elFps.className   = "metric-value " + fpsColour(fps);

  elP50.innerHTML = `${p50.toFixed(1)}<span class="metric-unit"> ms</span>`;
  elP50.className = "metric-value " + latencyColour(p50);

  elP95.innerHTML = `${p95.toFixed(1)}<span class="metric-unit"> ms</span>`;
  elP95.className = "metric-value " + latencyColour(p95);

  elMax.innerHTML = `${maxMs.toFixed(1)}<span class="metric-unit"> ms</span>`;
  elMax.className = "metric-value " + latencyColour(maxMs);

  // ── Stage bars ─────────────────────────────────────────────────────────
  const stageOrder = [
    { key: "yolo_l_ms",  label: "YOLO Left"  },
    { key: "yolo_r_ms",  label: "YOLO Right" },
    { key: "sgbm_ms",    label: "SGBM"       },
    { key: "depth_ms",   label: "Depth"      },
    { key: "render_ms",  label: "Render"     },
  ];
  const totalBudget = TARGET_MS;

  elStageRows.innerHTML = "";
  let pipelineMs = 0;
  for (const s of stageOrder) {
    const ms = stages[s.key] || 0;
    pipelineMs += ms;
    elStageRows.appendChild(renderStageBar(s.label, ms, totalBudget));
  }

  // ── Budget bars ─────────────────────────────────────────────────────────
  elBudgetRows.innerHTML = "";
  elBudgetRows.appendChild(renderBudgetBar("Pipeline", pipelineMs, totalBudget));
  elBudgetRows.appendChild(renderBudgetBar("Budget",   parseFloat(TARGET_MS), totalBudget));
  elBudgetRows.appendChild(renderBudgetBar("Margin",
    Math.max(0, parseFloat(TARGET_MS) - avgMs), parseFloat(TARGET_MS)));

  // ── Depth ───────────────────────────────────────────────────────────────
  if (depthCm != null) {
    elDepthVal.textContent   = depthCm.toFixed(1);
    elDepthBadge.className   = "depth-badge has-value";
  } else {
    elDepthVal.textContent   = "--";
    elDepthBadge.className   = "depth-badge no-value";
  }

  if (dMedian != null) {
    elDMedianVal.textContent   = dMedian.toFixed(1);
    elDMedianBadge.className   = "depth-badge has-value";
  } else {
    elDMedianVal.textContent   = "--";
    elDMedianBadge.className   = "depth-badge no-value";
  }

  // ── Match confidence ───────────────────────────────────────────────────
  const conf = data.match_confidence;
  if (conf != null) {
    elConfVal.textContent = (conf * 100).toFixed(0) + "%";
    elConfBadge.className = "depth-badge has-value";
    if (conf >= 0.4) {
      elConfBadge.style.color = "#4f8";
      elConfBadge.style.borderColor = "#2a6";
    } else if (conf >= 0.2) {
      elConfBadge.style.color = "#f80";
      elConfBadge.style.borderColor = "#a60";
    } else {
      elConfBadge.style.color = "#f44";
      elConfBadge.style.borderColor = "#a22";
    }
  } else {
    elConfVal.textContent = "--";
    elConfBadge.className = "depth-badge no-value";
    elConfBadge.style.color = "";
    elConfBadge.style.borderColor = "";
  }

  // ── Suggested SCALE table ─────────────────────────────────────────────
  if (Object.keys(scales).length > 0) {
    elScaleTbody.innerHTML = "";
    for (const [cm, scale] of Object.entries(scales)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${cm} cm</td><td>${scale.toFixed(3)}</td>`;
      elScaleTbody.appendChild(tr);
    }
    elScaleTable.hidden = false;
  }

  // ── Info row ───────────────────────────────────────────────────────────
  elWinInfo.textContent    = `window: ${winFrames} frames`;
  elTargetInfo.textContent = `target: ${TARGET_MS} ms/frame`;
  elUpdInfo.textContent    = `updated: ${new Date().toLocaleTimeString()}`;

  // ── Sparkline ───────────────────────────────────────────────────────────
  if (avgMs > 0) {
    lastHistory.push(avgMs);
    if (lastHistory.length > 120) lastHistory.shift();
    drawSparkline(lastHistory.slice(-60));
  }

  // ── Refresh indicator ───────────────────────────────────────────────────
  elDot.classList.add("active");
  setTimeout(() => elDot.classList.remove("active"), 300);
}

// ── Start ────────────────────────────────────────────────────────────────────
setInterval(poll, POLL_INTERVAL_MS);
poll();
