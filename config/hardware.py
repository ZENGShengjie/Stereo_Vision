"""项目统一硬件常量 + 工具。

本文件是 Stereo_Vision 项目的"硬件真理源":所有摄像头/光学/标定相关的数值
统一从这里读,严禁在其它文件中硬编码 3mm / 6cm / 1920 / 1080 / 3840 / 80°。

术语约定:全代码用 ``target_depth_cm`` 表示"相机到目标绝对深度",单位厘米。
本任务的"目标"是 COCO ID=41 cup,作为头皮替代参照物。
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dynamic hardware parameter overrides (live-tunable via API + persisted to disk)
# ---------------------------------------------------------------------------
# Architecture (2026-07-03):
# - The build-time defaults below are the "source of truth" for code that runs
#   during import (e.g. assert in stereo_depth.py).
# - Runtime mutations (POST /api/calibrate/disp_scale) are stored in
#   ``_OVERRIDES``, a thread-safe dict.
# - :func:`get_disp_scale` (and the analogous getters added later) is the ONLY
#   safe way for code to read these values at runtime — it falls back to the
#   module constant if no override is set.
# - :func:`set_disp_scale` writes to both the live dict and a JSON file at
#   ``data/hardware_overrides.json`` so values survive a server restart.
# - Every mutation is appended to ``data/hardware_override_log.jsonl`` (one
#   line per change) for audit / tuning forensics.
# ---------------------------------------------------------------------------
_OVERRIDES: dict[str, float] = {}
_OVERRIDES_LOCK = RLock()
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_OVERRIDES_FILE = _DATA_DIR / "hardware_overrides.json"
_OVERRIDES_LOG = _DATA_DIR / "hardware_override_log.jsonl"


def _load_overrides() -> None:
    """Read persisted overrides from disk into ``_OVERRIDES`` (called at import).

    Silent on missing/corrupt files — build-time defaults remain authoritative
    until the operator explicitly calls :func:`set_disp_scale`.
    """
    if not _OVERRIDES_FILE.exists():
        return
    try:
        import json
        with _OVERRIDES_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            with _OVERRIDES_LOCK:
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        _OVERRIDES[k] = float(v)
    except Exception:
        pass


def _persist_overrides() -> None:
    """Write ``_OVERRIDES`` to ``hardware_overrides.json`` (best-effort)."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        import json
        with _OVERRIDES_LOCK:
            snapshot = dict(_OVERRIDES)
        with _OVERRIDES_FILE.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _log_override(key: str, old: float, new: float, source: str, extra: dict | None = None) -> None:
    """Append one NDJSON line to the audit log (best-effort)."""
    try:
        import json
        import time
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts_ms": int(time.time() * 1000),
            "key": key,
            "old": old,
            "new": new,
            "source": source,
        }
        if extra:
            payload.update(extra)
        with _OVERRIDES_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Load persisted overrides eagerly so the first frame after restart already uses them.
_load_overrides()

# ---------------------------------------------------------------------------
# 1. 摄像头光学
# ---------------------------------------------------------------------------
FOCAL_LENGTH_MM: float = 3.0
"""镜头焦距(物理),3 mm。"""

BASELINE_CM: float = 6.0
"""双目基线,6 cm。"""

HFOV_DEG: float = 80.0
"""镜头水平视场角,80°(用来反推像素焦距,见 compute_focal_px)。"""

# ---------------------------------------------------------------------------
# 2. 分辨率(默认 1920x1080,可被环境变量 STEREO_MONO_HEIGHT / STEREO_USB_HEIGHT 覆盖)
# ---------------------------------------------------------------------------
# 注意:某些廉价 USB 摄像头在 1920x1080 模式下 read() 要 ~1000ms,实际只跑得动
# 1fps。把 STEREO_USB_HEIGHT 调小(如 540)能让摄像头切换到快读模式
# (实测 ~180ms,即 5.5fps)。
# MONO_HEIGHT 优先级:STEREO_MONO_HEIGHT > STEREO_USB_HEIGHT > 1080 默认
# 例(Windows PowerShell):
#   $env:STEREO_USB_HEIGHT = 540        # 让摄像头跑快读模式
#   $env:STEREO_MONO_HEIGHT = 1080      # 让 pipeline 按 1080 处理
#   .venv\Scripts\python.exe main.py
def _resolve_mono_height() -> int:
    """优先级 STEREO_MONO_HEIGHT > STEREO_USB_HEIGHT > 默认 1080"""
    for env_name in ("STEREO_MONO_HEIGHT", "STEREO_USB_HEIGHT"):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            v = int(raw)
        except ValueError:
            continue
        if v > 0:
            return v
    return 1080


MONO_WIDTH: int = 1920
"""单目物理宽度,像素。"""

MONO_HEIGHT: int = _resolve_mono_height()
"""单目物理高度,像素。优先级:STEREO_MONO_HEIGHT > STEREO_USB_HEIGHT > 1080。"""

SBS_WIDTH: int = MONO_WIDTH * 2  # 跟随
"""双目拼接横向分辨率,像素。"""

SBS_HEIGHT: int = MONO_HEIGHT  # 跟随
"""双目拼接高度,像素。"""

# ---------------------------------------------------------------------------
# 3. 标定文件路径
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
"""仓库根目录。"""

CALIB_NPZ_PATH: Path = BASE_DIR / "config" / "calibration" / "calib.npz"
"""外部产出的双目标定 npz 文件路径。

约定坐标系 = 单眼 1920x1080(因为单设备 SBS 模式下,``USBCamera._read_raw``
先做 ``frame[:, :eye_w]`` 拆出单眼图,后续 remap 都在这个单眼坐标系上做)。
"""

# ---------------------------------------------------------------------------
# 4. 目标检测
# ---------------------------------------------------------------------------
TARGET_CLS_ID: int = 41
"""COCO 类别 ID 41 = cup,作为本任务的目标检测类别(替代头皮)。"""

TARGET_CLS_NAME: str = "cup"
"""类别名,只用于日志/UI,不影响逻辑。"""

DEFAULT_CONF_THRESHOLD: float = 0.5
"""YOLO 推理的最低置信度阈值。"""

# ---------------------------------------------------------------------------
# 5. SGBM 参数
# ---------------------------------------------------------------------------
SGBM_WIDTH: int = 320
"""SGBM 内部计算用的宽度,像素。"""

SGBM_HEIGHT: int = 240
"""SGBM 内部计算用的高度,像素。"""

SGBM_NUM_DISPARITIES: int = 192
"""SGBM 视差范围(必须是 16 的倍数)。

Bug fix(2026-06-17):原值 64 太小。SGBM 算出的视差(d)上限 ≈ numDisparities ×
(mono_w / sgbm_w) = 64 × (1920/320) = 384 px。物理 5cm 处实际 d ≈ 1373 px,
超过这个上限,5cm 处 SGBM 算不出来,显示"无有效深度"。

放宽到 192 后上限 ≈ 192 × 6 = 1152 px,覆盖物理 5-6cm 起的近距离。
如果想覆盖更近(< 5cm),继续往上加(必须是 16 倍数)。
"""

SGBM_BLOCK_SIZE: int = 5
"""SGBM 匹配块大小(奇数)。"""

SGBM_P1_MULT: int = 8
"""SGBM P1 = P1_MULT * 3 * blockSize**2(OpenCV 官方推荐公式)。"""

SGBM_P2_MULT: int = 32
"""SGBM P2 = P2_MULT * 3 * blockSize**2。"""

SGBM_MEDIAN_KSIZE: int = 5
"""视差图后处理中值滤波核大小(奇数)。"""

# ---------------------------------------------------------------------------
# 6. 深度解算
# ---------------------------------------------------------------------------
DEPTH_SMOOTH_WINDOW: int = 5
"""深度值滑动均值窗口长度(防抖)。"""

DISP_SCALE: float = 1.0
"""SGBM 视差 → 真实物理 cm 的整体校准系数(2026-06-17 引入)。

**这是当前 fix 的核心**。实测发现 SGBM 输出视差 d 比"按像素焦距 + 基线"
公式推出的 d **小约 2-4 倍** —— 主要因为:
1. 没标定 npz(UNRECTIFIED 模式),左右相机未做立体校正,SGBM 算出的视差
   包含"未校正几何误差",导致视差偏小
2. SGBM 的 ``uniquenessRatio / P1 / P2`` 等参数在 1080p 上对弱纹理区域
   会输出保守的小视差
3. 镜头真实焦距可能和规约字面值 ``FOCAL_LENGTH_MM=3.0`` 有偏差(USB 摄像头
   镜头组实际焦距常在 2.4-3.6mm 之间)

校准公式:
- 真实深度: Z_real = (f_px * BASELINE_CM) / (d * DISP_SCALE)
- 当前显示: Z_meas = (f_px * BASELINE_CM) / (d * 1.0)  (DISP_SCALE=1.0 时)
- 校准后:   DISP_SCALE = Z_real / Z_meas

**实测校准流程**:
1. 把 cup 放在尺子量出的已知距离 L_cm 处
2. 运行程序,记录显示的 z_cm
3. 计算: ``DISP_SCALE = L_cm / z_cm``
   (例如 L=25cm, 显示 z=13cm → SCALE≈1.92)
4. 改本常量,重启程序(或用 /api/calibrate/disp_scale 动态更新),
   验证显示 ≈ L_cm

**用性能面板校准** (无需重启):
1. 打开 https://0.0.0.0:9000/perf
2. 确认左侧视频流里能看到 cup 并显示深度值
3. 等待 perf 页面更新(d_median_px 和 z_measured_cm 有数字)
4. 用尺子量出真实距离,算出 DISP_SCALE = real_cm / z_measured_cm

默认 1.0 是未校准状态;已知正确值范围在 1.5 - 5.0 之间。
"""

DEPTH_MIN_CM: float = 0.5
"""深度显示下限,cm。

Bug fix(2026-06-17):从 0.01 改回 0.5。引入 :data:`DISP_SCALE` 后公式输出
对齐真实物理 cm,cup 30cm 显示 30cm(不是 0.13)。0.5cm 下限防止 SGBM
近处偶尔算出错误的极大值(> 真实 100cm)。
"""

DEPTH_MAX_CM: float = 500.0
"""深度显示上限,cm。

放宽到 500 覆盖远距离工作场景(2-3m 内)。超过 500cm 视为 SGBM 失真。
"""

# ---------------------------------------------------------------------------
# 7. 预警阈值
# ---------------------------------------------------------------------------
SAFE_CM: float = 5.0
"""> SAFE_CM = 安全接近区(绿色)。"""

DANGER_CM: float = 2.0
"""<= DANGER_CM 且 > 0 = 危险接触区(红色闪烁);<= SAFE_CM 且 > DANGER_CM = 标准工作区(黄色)。"""

DANGER_FLASH_HZ: float = 1.0
"""危险区闪烁频率,Hz。"""

# ---------------------------------------------------------------------------
# 8. 工具函数
# ---------------------------------------------------------------------------
def compute_focal_px() -> float:
    """把硬件焦距 + HFOV 推算成单目像素焦距。

    公式(从水平视场角反推):
        focal_px = MONO_WIDTH / (2 * tan(HFOV_DEG/2))

    这样不需要在 npz 里假定 Q 矩阵,也能给出一致的像素焦距。

    Returns:
        像素焦距(浮点)。
    """
    hfov_rad = math.radians(HFOV_DEG)
    return MONO_WIDTH / (2.0 * math.tan(hfov_rad / 2.0))


def z_cm_from_disparity(disp: float) -> float:
    """把视差(像素)转成相机到目标绝对深度(cm)。

    公式采用"真实双目几何 + DISP_SCALE 校准"(2026-06-17 修复):
        Z_cm = (f_px * BASELINE_CM) / (d * DISP_SCALE)

    其中:
    - ``f_px = compute_focal_px()`` —— 用 HFOV + 物理焦距反推的像素焦距
      (3mm + 80° → ~1144 px),这是真实双目三角测距公式
    - ``d`` —— SGBM 输出的像素视差
    - ``DISP_SCALE`` —— 校准系数,见 :data:`DISP_SCALE` 说明,默认 4.0
      实际值通过 :func:`get_disp_scale` 实时读取(支持热更新)

    **历史**:原公式 ``Z_cm = (FOCAL_LENGTH_MM * BASELINE_CM * 10) / d / 10``
    = ``18 / d`` 仅用物理焦距 3mm 不带像素焦距,与真实物理 cm 有 ~4 倍尺度差。
    现在改为真实双目三角测距公式 + DISP_SCALE 校准。

    Args:
        disp: SGBM 输出的视差(像素),``d > 0``,调用方需自己保证。

    Returns:
        深度,cm(对齐真实物理,经 DISP_SCALE 校准后)。

    Raises:
        ValueError: ``disp <= 0`` 时。
    """
    if disp <= 0:
        raise ValueError(f"disp must be > 0, got {disp}")
    f_px = compute_focal_px()
    return (f_px * BASELINE_CM) / (disp * get_disp_scale())


# ---------------------------------------------------------------------------
# Public API for runtime-tunable hardware parameters (2026-07-03)
# ---------------------------------------------------------------------------
def get_disp_scale() -> float:
    """Read the live DISP_SCALE (override → build-time default).

    Always call this instead of importing the constant directly when computing
    depth. Reading the constant via ``from config.hardware import DISP_SCALE``
    captures the import-time value, which means a runtime ``POST /api/calibrate
    /disp_scale`` would not be visible to the depth formula.
    """
    with _OVERRIDES_LOCK:
        v = _OVERRIDES.get("DISP_SCALE")
    if v is not None and v > 0:
        return float(v)
    return float(DISP_SCALE)


def set_disp_scale(new_value: float, *, source: str = "api", extra: dict | None = None) -> dict:
    """Update DISP_SCALE in memory, persist to disk, and append an audit line.

    Args:
        new_value: New scale (must be > 0 and <= 100; sanity bound).
        source: Short tag for the audit log (e.g. ``"api"`` or ``"cli"``).
        extra: Optional metadata to attach to the audit log entry.

    Returns:
        ``{"old": float, "new": float, "persisted": bool}``

    Raises:
        ValueError: ``new_value`` out of range.
    """
    new_value = float(new_value)
    if not (0.001 <= new_value <= 100.0):
        raise ValueError(f"DISP_SCALE must be in [0.001, 100], got {new_value}")
    with _OVERRIDES_LOCK:
        old_value = _OVERRIDES.get("DISP_SCALE", float(DISP_SCALE))
        _OVERRIDES["DISP_SCALE"] = new_value
    _persist_overrides()
    _log_override("DISP_SCALE", old_value, new_value, source, extra)
    return {"old": old_value, "new": new_value, "persisted": True}


def list_overrides() -> dict[str, dict]:
    """Return a snapshot of all live overrides (audit-friendly view)."""
    with _OVERRIDES_LOCK:
        return {
            k: {"current": v, "default": float(DISP_SCALE) if k == "DISP_SCALE" else None}
            for k, v in _OVERRIDES.items()
        }


def resize_for_sgbm(img: np.ndarray) -> tuple[np.ndarray, float]:
    """等比例缩放到 SGBM 内部计算分辨率(见 §0 resize 硬约束)。

    - 始终等比例,``fx == fy``。
    - 选 ``min(sx, sy)`` 保证不超框、不裁切、不留黑边。
    - 返回 ``scale`` 用于把 SGBM 视差图反推回 mono 分辨率时使用同一比例,
      保证 ``(cx, cy)`` 采样点不偏移。

    Args:
        img: 输入图像(HxW 或 HxWx3)。

    Returns:
        (resized_img, scale)
    """
    h, w = img.shape[:2]
    if (w, h) == (SGBM_WIDTH, SGBM_HEIGHT):
        return img, 1.0
    sx = SGBM_WIDTH / w
    sy = SGBM_HEIGHT / h
    scale = min(sx, sy)  # 等比例 + 不超框
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if (new_w, new_h) == (w, h):
        return img, 1.0
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def resize_to_mono(img: np.ndarray, scale: float) -> np.ndarray:
    """把 SGBM 分辨率的图等比例反推回 mono 分辨率。

    与 :func:`resize_for_sgbm` 配对使用:``resize_for_sgbm`` 给出 ``scale``,
    本函数使用同一 ``scale`` 反推。这样视差图和原图在 ``(cx, cy)`` 采样点
    1:1 对应,无偏移。

    Args:
        img: SGBM 分辨率的图(HxW 或 HxWxC)。
        scale: :func:`resize_for_sgbm` 返回的缩放比。

    Returns:
        mono 分辨率(MONO_WIDTH x MONO_HEIGHT)图。
    """
    if scale == 1.0 and img.shape[1] == MONO_WIDTH and img.shape[0] == MONO_HEIGHT:
        return img
    if scale == 1.0:
        return cv2.resize(img, (MONO_WIDTH, MONO_HEIGHT), interpolation=cv2.INTER_LINEAR)
    new_w = int(round(img.shape[1] / scale))
    new_h = int(round(img.shape[0] / scale))
    if (new_w, new_h) != (MONO_WIDTH, MONO_HEIGHT):
        return cv2.resize(
            img,
            (MONO_WIDTH, MONO_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
    return cv2.resize(
        img,
        (MONO_WIDTH, MONO_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


__all__ = [
    "FOCAL_LENGTH_MM",
    "BASELINE_CM",
    "HFOV_DEG",
    "MONO_WIDTH",
    "MONO_HEIGHT",
    "SBS_WIDTH",
    "SBS_HEIGHT",
    "BASE_DIR",
    "CALIB_NPZ_PATH",
    "TARGET_CLS_ID",
    "TARGET_CLS_NAME",
    "DEFAULT_CONF_THRESHOLD",
    "SGBM_WIDTH",
    "SGBM_HEIGHT",
    "SGBM_NUM_DISPARITIES",
    "SGBM_BLOCK_SIZE",
    "SGBM_P1_MULT",
    "SGBM_P2_MULT",
    "SGBM_MEDIAN_KSIZE",
    "DEPTH_SMOOTH_WINDOW",
    "DISP_SCALE",
    "DEPTH_MIN_CM",
    "DEPTH_MAX_CM",
    "SAFE_CM",
    "DANGER_CM",
    "DANGER_FLASH_HZ",
    "compute_focal_px",
    "z_cm_from_disparity",
    "resize_for_sgbm",
    "resize_to_mono",
    "get_disp_scale",
    "set_disp_scale",
    "list_overrides",
]
