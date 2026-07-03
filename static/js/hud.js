/**
 * HUD view client (2026-07-04).
 *
 * - 拉一次 `/api/hud/state` 同时拿到 stats + hardware overrides
 * - 把 pipeline 实时读数更新到顶部面板
 * - 5 个参数输入框 + SET 按钮单独走 `/api/calibrate/hardware`
 * - 真距校准按钮走 `/api/calibrate/disp_scale?real_cm=...`
 * - 右上角 [ - ] 按钮切显示/隐藏, 走 `/api/hud/toggle`
 *
 * 设计原则:
 * - HUD 只读最新一次 pull, 不储存 history (避免内存泄漏)
 * - 错误/成功提示 3 秒自动消失
 * - 防抖: 输入框失焦时才推, 不要每个字符都 POST
 */

(function () {
  "use strict";

  const POLL_MS = 700;             // 后端轮询周期
  const FEEDBACK_MS = 2200;         // 提示条停留时间

  // ---------- DOM ----------
  const bar       = document.getElementById("hud-bar");
  const toggleBtn = document.getElementById("hud-toggle-btn");
  const foot      = document.getElementById("hud-foot");
  const img       = document.getElementById("stereo-feed");

  const fpsEl     = document.getElementById("hud-fps");
  const depthEl   = document.getElementById("hud-depth");
  const dispEl    = document.getElementById("hud-disp");
  const confEl    = document.getElementById("hud-conf");
  const qualityEl = document.getElementById("hud-quality");
  const statusEl  = document.getElementById("hud-status");
  const pathEl    = document.getElementById("hud-path");

  const realCmBtn  = document.getElementById("btn-real-cm");
  const realCmIn   = document.getElementById("real-cm-input");
  const resetBtn   = document.getElementById("btn-reset");

  // ---------- 反馈条 ----------
  let footTimer = null;
  function feedback(msg, kind) {
    if (!msg) return;
    foot.textContent = msg;
    foot.className = "hud-foot show " + (kind || "");
    if (footTimer) clearTimeout(footTimer);
    footTimer = setTimeout(() => {
      foot.className = "hud-foot";
      foot.textContent = "";
    }, FEEDBACK_MS);
  }

  // ---------- pull HUD state ----------
  async function pullState() {
    try {
      const r = await fetch("/api/hud/state", { cache: "no-store" });
      const data = await r.json();
      renderStats(data);
      renderHardware(data.hardware_overrides || {});
    } catch (err) {
      statusEl.textContent = "unreachable";
      statusEl.style.color = "#f55";
    }
  }

  function renderStats(d) {
    if (!d || d.error) {
      statusEl.textContent = d && d.error ? "err" : "no pipeline";
      return;
    }
    fpsEl.textContent     = (d.fps ?? 0).toFixed(1);
    depthEl.textContent   = (d.depth_cm == null) ? "—" : d.depth_cm.toFixed(1);
    dispEl.textContent    = (d.d_median_px == null) ? "—" : d.d_median_px.toFixed(1);
    confEl.textContent    = ((d.match_confidence ?? 0) * 100).toFixed(0) + "%";
    qualityEl.textContent = (d.quality || "--").toUpperCase();
    qualityEl.className   = "badge " + (d.quality || "low");
    statusEl.textContent  = d.depth_status || "—";
    statusEl.style.color  = statusColor(d.depth_status);
    const counts = d.depth_status_counts || {};
    pathEl.textContent    = Object.keys(d.sgbm_path_counts || {}).map(k =>
      `${k}=${d.sgbm_path_counts[k]}`).join(" ") || "—";
    void counts; // reserved for future UI
  }

  function statusColor(s) {
    if (!s) return "#888";
    if (s === "ok") return "#4f8";
    if (s === "sgbm_error" || s === "no_target") return "#f55";
    return "#fc4";
  }

  function renderHardware(map) {
    // 5 个硬件键, 用 data-key 关联, 把 current/default 值写到 input
    Object.keys(map).forEach(k => {
      const inp = document.getElementById("hw-" + k);
      if (inp && document.activeElement !== inp) {
        inp.value = map[k].current;
      }
    });
  }

  // ---------- 参数下发 ----------
  async function setHwKey(key, value) {
    try {
      const r = await fetch("/api/calibrate/hardware", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: value }),
      });
      const out = await r.json();
      if (!r.ok) throw new Error(out.error || `HTTP ${r.status}`);
      flashBtn(key, true);
      feedback(`set ${key}=${value} → OK`, "success");
      pullState();
    } catch (err) {
      flashBtn(key, false);
      feedback(`set ${key}: ${err.message}`, "error");
    }
  }

  function flashBtn(key, ok) {
    document.querySelectorAll(".btn-set[data-key='" + key + "']").forEach(b => {
      b.classList.add(ok ? "success" : "error");
      setTimeout(() => b.classList.remove("success", "error"), 1200);
    });
  }

  // 监听所有 SET 按钮 + 输入框失焦/回车
  document.querySelectorAll(".btn-set").forEach(btn => {
    const key = btn.dataset.key;
    const inp = document.getElementById("hw-" + key);
    btn.addEventListener("click", () => {
      const v = parseFloat(inp.value);
      if (isFinite(v)) setHwKey(key, v);
    });
    inp.addEventListener("change", () => {
      const v = parseFloat(inp.value);
      if (isFinite(v)) setHwKey(key, v);
    });
    inp.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        const v = parseFloat(inp.value);
        if (isFinite(v)) setHwKey(key, v);
      }
    });
  });

  // ---------- 真距校准 ----------
  realCmBtn.addEventListener("click", async () => {
    const cm = parseFloat(realCmIn.value);
    if (!isFinite(cm) || cm <= 0) {
      feedback("真距必须 > 0", "error");
      return;
    }
    try {
      const r = await fetch("/api/calibrate/disp_scale", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ real_cm: cm }),
      });
      const out = await r.json();
      if (!r.ok) throw new Error(out.error || `HTTP ${r.status}`);
      feedback(`真距 ${cm}cm → DISP_SCALE=${out.new_disp_scale}`, "success");
      pullState();
    } catch (err) {
      feedback(`真距校准失败: ${err.message}`, "error");
    }
  });

  // ---------- 恢复默认 ----------
  resetBtn.addEventListener("click", async () => {
    if (!window.confirm("恢复所有可调参数到默认值?")) return;
    // 通过把所有 override 清空 (POST 把已知 default, 用 api/calibrate/hardware 不能直接 "reset")
    // 折中: 调一个未注册的 key 会 404; 简化: 调 hud/toggle + reload
    // 这里用更直接的方案: 调一个 reset 端点(但当前没实现, 简化为 reload)
    feedback("重置请改 data/hardware_overrides.json 然后重启", "success");
  });

  // ---------- 右上角开关 ----------
  toggleBtn.addEventListener("click", async () => {
    const newHidden = !bar.classList.contains("hidden");
    bar.classList.toggle("hidden", newHidden);
    toggleBtn.textContent = newHidden ? "[ + ]" : "[ - ]";
    try {
      await fetch("/api/hud/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visible: !newHidden }),
      });
    } catch (_) { /* ignore */ }
  });

  // ---------- 视频流 ----------
  img.src = "/video_feed";

  // ---------- 启动轮询 ----------
  pullState();
  setInterval(pullState, POLL_MS);
})();
