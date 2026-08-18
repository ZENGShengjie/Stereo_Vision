"""双目视差 + 深度解算。

任务规约约束:
- 公式:``Z_mm = (FOCAL_LENGTH_MM * BASELINE_CM * 10) / d``,``Z_cm = Z_mm / 10``。
- 防抖:维护 ``DEPTH_SMOOTH_WINDOW`` 长的滑动均值窗口。
- 钳位:``DEPTH_MIN_CM`` 到 ``DEPTH_MAX_CM``(对**有效**深度值才生效)。
- SGBM 内部计算用 :func:`config.hardware.resize_for_sgbm` 等比例缩放;
  输出视差图用同一 ``scale`` 反推回 mono 分辨率,保证 ``(cx, cy)`` 采样点不偏移。
- 输入必须是已校正图(由 :meth:`USBCamera.read_rectified_pair` 给出)。

Bug fix(2026-06-17):
- 新增 :meth:`StereoDepthSolver.read_depth_in_box`,在 box 内做网格多点采样取中位数,
  解决 cup 中心点落在杯柄/反光区导致单点 SGBM 视差无效的问题。

Refactor(2026-07-03):
- ``read_depth_at`` 单点采样已删除:单点采样在杯柄/反光区噪点极大,
  精度不可控;统一改用框内多点采样 + 直方图峰值 + 2D 高斯空间加权 +
  trimmed mean(见 :meth:`read_depth_in_box`)。
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Optional

import cv2
import numpy as np

# Debug logging (2026-06-17, session=c7ffa8)
_DEBUG_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "debug-c7ffa8.log",
)


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "?") -> None:
    """写一行 NDJSON 到 debug-c7ffa8.log (不依赖服务器, 直接落盘)."""
    try:
        payload = {
            "id": f"log_{int(time.time()*1000)}_{os.getpid()}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

from config.hardware import (
    BASELINE_CM,
    DEPTH_MAX_CM,
    DEPTH_MIN_CM,
    DEPTH_SMOOTH_WINDOW,
    FOCAL_LENGTH_MM,
    MONO_HEIGHT,
    MONO_WIDTH,
    SGBM_HEIGHT,
    SGBM_MEDIAN_KSIZE,
    SGBM_P1_MULT,
    SGBM_P2_MULT,
    SGBM_WIDTH,
    compute_focal_px,
    get_hw,
    resize_for_sgbm,
    z_cm_from_disparity,
)

# Read DISP_SCALE dynamically so runtime calibration can change it without restart
def _disp_scale() -> float:
    from config.hardware import DISP_SCALE as _ds
    return _ds

logger = logging.getLogger(__name__)


# 硬约束哨兵:如果硬件常量被误改,启动期直接报错
assert FOCAL_LENGTH_MM == 3.0, (
    f"FOCAL_LENGTH_MM must be 3.0 (task spec), got {FOCAL_LENGTH_MM}"
)
assert BASELINE_CM == 6.0, (
    f"BASELINE_CM must be 6.0 (task spec), got {BASELINE_CM}"
)


class StereoDepthSolver:
    """SGBM 视差 + 公式深度解算 + 滑动均值防抖。

    Args:
        window_size: 滑动均值窗口长度,默认 :data:`DEPTH_SMOOTH_WINDOW` = 5。
    """

    def __init__(self, window_size: int = DEPTH_SMOOTH_WINDOW) -> None:
        # 初始 matcher 用当前 SGBM_* 动态值
        self._matcher_num_disp = -1
        self._matcher_block_size = -1
        self._matcher = self._build_matcher()
        self._focal_px = float(compute_focal_px())
        self._baseline_m = float(get_hw("BASELINE_CM")) / 100.0
        self._window_size = max(1, int(window_size))
        self._last_disp_at_mono: Optional[np.ndarray] = None
        self._last_d_median: Optional[float] = None
        # 光照鲁棒追踪:匹配置信度(0-1, 低光照/剧烈光影变化时下降)
        self._match_confidence: float = 1.0
        # 置信度自适应 EMA 平滑(替代简单滑动均值):
        # - ema_alpha 由置信度动态决定: conf 越高 → alpha 越大 → 新帧权重越高
        # - 避免低质量帧污染长期均值，从根本上消除静止时的数值闪烁
        self._ema_value: Optional[float] = None
        self._ema_confidence: float = 1.0
        # 自适应离群值追踪: 连续被拒绝计数 → 触发阈值放宽
        self._consecutive_rejects = 0
        # 分段线性校准表: sorted list of (sensor_reading_cm, true_depth_cm) pairs
        # 由 add_calibration_point() 累积, reset_smoothing() 清空
        self._calib_points: list[tuple[float, float]] = [
            # 用户实测 4 点: (传感器读数, 真实距离)
            # 11.2→11.5, 12.3→13.5, 13.9→15.5, 15.6→17.5
            (11.2, 11.5),
            (12.4, 13.5),
            (13.9, 15.5),
            (15.6, 17.5),
        ]
        # 最近一次未校正的深度原始值 (用于校准时获取真实传感器读数)
        self._last_raw_depth_cm: Optional[float] = None

        # 状态锁:保护 EMA/校准状态,防止并发写入
        # 当前 pipeline 只有主线程访问,但 MJPEG handler 在 aiohttp 线程池里
        # 读 stats_summary() 时可能与 process_one_frame() 并发
        self._state_lock = Lock()

        # #region agent log — H4: 验证 self._focal_px vs compute_focal_px() 实时值
        _debug_log(
            "processing/stereo_depth.py:__init__",
            "Solver init focal_px + DISP_SCALE",
            {
                "self_focal_px": self._focal_px,
                "compute_focal_px_now": float(compute_focal_px()),
                "FOCAL_LENGTH_MM": FOCAL_LENGTH_MM,
                "BASELINE_CM": float(get_hw("BASELINE_CM")),
                "DISP_SCALE": _disp_scale(),
                "HFOV_DEG": float(get_hw("HFOV_DEG")),
                "opencv_build_info": self._get_opencv_backend_info(),
            },
            "H4",
        )
        # #endregion

        logger.info(
            "[StereoDepthSolver] SGBM: numDisp=%d blockSize=%d p1=%d p2=%d, "
            "smoothing window=%d, focal_px=%.2f",
            self._matcher_num_disp, self._matcher_block_size,
            self._p1, self._p2,
            self._window_size, self._focal_px,
        )

    def _build_matcher(self) -> cv2.StereoSGBM:
        """Build an SGBM matcher from current dynamic SGBM_NUM_DISPARITIES / SGBM_BLOCK_SIZE."""
        num_disp = int(get_hw("SGBM_NUM_DISPARITIES"))
        block = int(get_hw("SGBM_BLOCK_SIZE"))
        self._p1 = SGBM_P1_MULT * 3 * block * block
        self._p2 = SGBM_P2_MULT * 3 * block * block
        m = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block,
            P1=self._p1,
            P2=self._p2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=0,
            speckleRange=0,
            mode=cv2.STEREO_SGBM_MODE_SGBM,
        )
        self._matcher_num_disp = num_disp
        self._matcher_block_size = block
        return m

    def _maybe_rebuild_matcher(self) -> None:
        """Re-build SGBM matcher if dynamic SGBM_* keys changed since last build."""
        cur_num = int(get_hw("SGBM_NUM_DISPARITIES"))
        cur_block = int(get_hw("SGBM_BLOCK_SIZE"))
        if cur_num != self._matcher_num_disp or cur_block != self._matcher_block_size:
            _debug_log(
                "processing/stereo_depth.py:_maybe_rebuild_matcher",
                "sgbm_matcher_rebuilt",
                {
                    "old_num_disp": self._matcher_num_disp,
                    "new_num_disp": cur_num,
                    "old_block_size": self._matcher_block_size,
                    "new_block_size": cur_block,
                },
                "PERF",
            )
            self._matcher = self._build_matcher()

    def _get_opencv_backend_info(self) -> dict:
        """收集 OpenCV 推理后端信息."""
        info = {}
        try:
            info["cv_version"] = cv2.__version__
        except Exception:
            pass
        try:
            backends = []
            if hasattr(cv2, "cuda"):
                backends.append("cuda")
            try:
                if hasattr(cv2, "ocl"):
                    info["has_ocl"] = True
                    try:
                        ctx = cv2.ocl.Context.getProp()
                        info["ocl_device"] = str(ctx)
                    except Exception:
                        pass
            except Exception:
                pass
            info["detected_backends"] = backends
        except Exception:
            pass
        return info

    def compute_roi(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        box_l: tuple[int, int, int, int] | None = None,
        box_r: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """ROI-bounded SGBM (leader recommendation #6).

        After YOLO detects the cup in both eyes, the cup occupies a small
        fraction of the 1920×1080 image (typically <25%). Running SGBM on the
        full image is wasteful — both the SGBM cost and the cost of cropping
        to SGBM resolution scale with pixel count.

        This method:
        1. Computes a "stereo ROI" from the union of ``box_l`` and ``box_r``
           with margin; the right-eye ROI extends LEFT by
           ``SGBM_NUM_DISPARITIES`` pixels to allow disparity search.
        2. Crops left_rect/right_rect to the stereo ROI (much smaller area).
        3. Resizes the cropped region to SGBM internal size.
        4. Runs SGBM on the smaller, content-focused region.
        5. Places the disparity result back into a full-size map at the
           ROI location (zero elsewhere) so existing ``read_depth_in_box``
           keeps working unchanged.

        When ``box_l`` and ``box_r`` are both ``None``, falls back to
        :meth:`compute` (full-image SGBM, backward-compatible).

        Args:
            left_rect, right_rect: 已校正 BGR 图 (MONO_HEIGHT, MONO_WIDTH, 3).
            box_l, box_r: (x1, y1, x2, y2) cup boxes from YOLO, in MONO coords.

        Returns:
            disp_at_mono: full-size disparity map (MONO_HEIGHT, MONO_WIDTH).
            Pixels outside the ROI are 0 (invalid).
        """
        if box_l is None and box_r is None:
            return self.compute(left_rect, right_rect)

        h, w = left_rect.shape[:2]
        # ── 1. Compute stereo ROI ────────────────────────────────────
        # Y range: union of both boxes (cup appears at similar y in both eyes
        # since the cameras are nearly co-located).
        # X range: union of both boxes.
        # NOTE: We do NOT add a separate "disparity padding" column for the
        # right eye. cv2.StereoSGBM requires left.size() == right.size()
        # (asserts it at runtime). The internal numDisparities (192) already
        # limits how far left SGBM searches in the right image; if YOLO saw
        # the cup in both eyes, the right-eye cup position is inside the
        # same x-range (modulo disparity ≤ numDisp) — the cup appears
        # SHIFTED-LEFT in the right eye, but the right-eye cup is still
        # within [min(box_l.x1, box_r.x1), max(box_l.x2, box_r.x2)].
        margin_x = 40
        margin_y = 40

        if box_l is not None and box_r is not None:
            x1_l, y1_l, x2_l, y2_l = box_l
            x1_r, y1_r, x2_r, y2_r = box_r
            roi_y1 = max(0, min(y1_l, y1_r) - margin_y)
            roi_y2 = min(h, max(y2_l, y2_r) + margin_y)
            roi_x1 = max(0, min(x1_l, x1_r) - margin_x)
            roi_x2 = min(w, max(x2_l, x2_r) + margin_x)
        elif box_l is not None:
            x1_l, y1_l, x2_l, y2_l = box_l
            roi_x1 = max(0, x1_l - margin_x)
            roi_y1 = max(0, y1_l - margin_y)
            roi_x2 = min(w, x2_l + margin_x)
            roi_y2 = min(h, y2_l + margin_y)
        else:  # box_r is not None
            x1_r, y1_r, x2_r, y2_r = box_r
            roi_x1 = max(0, x1_r - margin_x)
            roi_y1 = max(0, y1_r - margin_y)
            roi_x2 = min(w, x2_r + margin_x)
            roi_y2 = min(h, y2_r + margin_y)

        # Enforce minimum ROI size — too small a crop means SGBM has insufficient
        # context to match. Fall back to full image in that case.
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        if roi_w < SGBM_WIDTH // 2 or roi_h < SGBM_HEIGHT // 2:
            _debug_log(
                "processing/stereo_depth.py:compute_roi",
                "roi_too_small_fallback_full",
                {"roi_w": roi_w, "roi_h": roi_h},
                "SGBM",
            )
            return self.compute(left_rect, right_rect)

        # ── 2. Crop left and right to the SAME ROI ─────────────────
        # IMPORTANT: both eyes cropped to identical (roi_w × roi_h) so that
        # cv2.StereoSGBM's size assertion passes.
        crop_l = left_rect[roi_y1:roi_y2, roi_x1:roi_x2]
        crop_r = right_rect[roi_y1:roi_y2, roi_x1:roi_x2]
        if crop_l.shape != crop_r.shape:
            _debug_log(
                "processing/stereo_depth.py:compute_roi",
                "crop_shape_mismatch_fallback_full",
                {
                    "crop_l_shape": list(crop_l.shape),
                    "crop_r_shape": list(crop_r.shape),
                },
                "SGBM",
            )
            return self.compute(left_rect, right_rect)

        gray_l = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)

        # ── 3. CLAHE + brightness balance (same as full-image path) ──
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)
        mean_l = gray_l.mean()
        mean_r = gray_r.mean()
        if mean_l > 1 and mean_r > 1:
            balance = np.clip(mean_l / mean_r, 0.6, 1.8)
            gray_r = np.clip(gray_r * balance, 0, 255).astype(np.uint8)

        # ── 4. Resize cropped ROI to SGBM internal size ─────────────
        gray_l_sgbm, scale = resize_for_sgbm(gray_l)
        gray_r_sgbm, _ = resize_for_sgbm(gray_r)

        # ── 5. Run SGBM on the smaller, content-focused region ──────
        _debug_log(
            "processing/stereo_depth.py:compute_roi",
            "before_sgbm_compute_roi",
            {
                "crop_shape": list(crop_l.shape[:2]),
                "sgbm_shape": list(gray_l_sgbm.shape),
                "scale": round(scale, 4),
                "speedup_full_to_roi": round(
                    (1080 * 1920) / max(1, gray_l_sgbm.size * (scale ** -2)),
                    2,
                ),
            },
            "SGBM",
        )
        # ── SGBM 动态参数(2026-07-03):若 runtime 改了 SGBM_NUM_DISPARITIES /
        #    SGBM_BLOCK_SIZE,这一行重建 matcher(只读一次 dict,O(1))
        self._maybe_rebuild_matcher()
        t_sgbm_start = time.perf_counter()
        disp_sgbm_x16 = self._matcher.compute(gray_l_sgbm, gray_r_sgbm)
        t_sgbm_end = time.perf_counter()
        disp_sgbm = disp_sgbm_x16.astype(np.float32) / 16.0

        # Median-blur cleanup at SGBM resolution
        max_disp = max(1.0, disp_sgbm.max())
        disp_u8 = np.clip(disp_sgbm / max_disp * 255.0, 0, 255).astype(np.uint8)
        if SGBM_MEDIAN_KSIZE > 1:
            disp_u8 = cv2.medianBlur(disp_u8, SGBM_MEDIAN_KSIZE)
        disp_sgbm = disp_u8.astype(np.float32) / 255.0 * max_disp
        _debug_log(
            "processing/stereo_depth.py:compute_roi",
            "sgbm_compute_done_roi",
            {"sgbm_ms": round((t_sgbm_end - t_sgbm_start) * 1000, 1)},
            "SGBM",
        )

        # ── 6. Map disparity back to crop coords then to mono coords ─
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        # Resize disparity to crop size (interp at small region resolution)
        disp_at_crop = cv2.resize(
            disp_sgbm,
            (crop_l.shape[1], crop_l.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ) * np.float32(inv_scale)
        disp_at_crop = np.where(disp_at_crop > 0, disp_at_crop, 0.0).astype(np.float32)

        # ── 7. Place into full-size output (zero outside ROI) ───────
        disp_at_mono = np.zeros((h, w), dtype=np.float32)
        # SGBM's disparity output is referenced to LEFT-eye pixel positions:
        # for pixel (x_l, y) in left, d = x_l - x_r. The cropped left and
        # right regions share the SAME x-range [roi_x1, roi_x2) in mono
        # coords, so disp_at_crop[y, x] corresponds to mono pixel
        # (roi_y1 + y, roi_x1 + x).
        disp_at_mono[roi_y1:roi_y2, roi_x1:roi_x2] = disp_at_crop[
            : roi_y2 - roi_y1, : roi_x2 - roi_x1
        ]
        # Zero out any negative values leaking through
        disp_at_mono = np.where(disp_at_mono > 0, disp_at_mono, 0.0).astype(np.float32)

        # ── 8. Match-confidence (same formula as compute) ────────────
        with self._state_lock:
            valid = disp_at_mono > 0
            valid_count = int(valid.sum())
            total_count = int(disp_at_mono.size)
            valid_ratio = valid_count / max(1, total_count)
            if valid_ratio > 0.005 and valid_count > 100:
                valid_disp = disp_at_mono[valid]
                cv = float(valid_disp.std() / (valid_disp.mean() + 1e-6))
                self._match_confidence = round(
                    float(np.clip(valid_ratio * (1.0 - cv), 0.0, 1.0)), 3
                )
            else:
                # ROI has few valid pixels (e.g., empty ROI); keep previous
                self._match_confidence = max(0.0, self._match_confidence - 0.05)

        with self._state_lock:
            self._last_disp_at_mono = disp_at_mono
        return disp_at_mono

    def compute(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        roi_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """对已校正左右图算 SGBM 视差,返回 mono 分辨率(1920x1080)的视差图。

        Args:
            left_rect, right_rect: 已校正 BGR 图,``(MONO_HEIGHT, MONO_WIDTH, 3)``。

            Returns:
            ``disp_at_mono``: 视差图(浮点,像素),``(MONO_HEIGHT, MONO_WIDTH)``。
            无效点 = 0。

            roi_mask: 可选 ROI mask ``(MONO_HEIGHT, MONO_WIDTH)``,255=感兴趣区域。
                若提供,SGBM 只在 ROI 内搜索(ROI 外填均匀灰抑制错误匹配)。
                未提供时退化为整图 SGBM(向后兼容)。

        Notes:
            - SGBM 输出是 16 倍定点,这里除以 16 折回像素。
            - 视差图与原图同分辨率(1920x1080),``(cx, cy)`` 采样 1:1 对应。
        """
        gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # ── 光照鲁棒预处理 ───────────────────────────────────────────────────
        # 1. CLAHE: 局部直方图均衡化 → 提升弱光区域对比度，抑制局部光斑
        #    clipLimit 越大对比度越强，tileGridSize 控制局部窗口
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

        # 2. 左右亮度均衡: 以左眼均值为基准，右眼乘比例系数
        #    防止左/右相机 auto-exposure 漂移导致匹配退化
        #    公式: 右眼校正 = 右眼原值 × (左眼均值 / 右眼均值)
        #    若右眼比左眼暗(比值>1)则提亮；若右眼比左眼亮则压暗
        mean_l = gray_l.mean()
        mean_r = gray_r.mean()
        if mean_l > 1 and mean_r > 1:
            scale = mean_l / mean_r
            scale = np.clip(scale, 0.6, 1.8)
            gray_r = np.clip(gray_r * scale, 0, 255).astype(np.uint8)

        # 3. ROI 预处理:在原分辨率上把非 ROI 区域填均匀灰 → resize → SGBM
        #    这样做的好处:ROI 外填均值后 resize 仍然均匀,SGBM 在 ROI 外不产生有效匹配
        if roi_mask is not None:
            roi_l_valid = (roi_mask > 127)
            roi_r_valid = (roi_mask > 127)
            roi_mean_l = float(gray_l[roi_l_valid].mean()) if roi_l_valid.any() else 128.0
            roi_mean_r = float(gray_r[roi_r_valid].mean()) if roi_r_valid.any() else 128.0
            gray_l = np.where(roi_l_valid, gray_l, int(roi_mean_l)).astype(np.uint8)
            gray_r = np.where(roi_r_valid, gray_r, int(roi_mean_r)).astype(np.uint8)

        # 等比例缩放到 SGBM 分辨率
        gray_l_sgbm, scale = resize_for_sgbm(gray_l)
        gray_r_sgbm, _ = resize_for_sgbm(gray_r)

        # #region agent log — 确认 SGBM 输入尺寸和 compute 耗时
        _debug_log(
            "processing/stereo_depth.py:compute",
            "before_sgbm_compute",
            {
                "gray_l_sgbm_shape": list(gray_l_sgbm.shape),
                "scale": round(scale, 4),
                "numDisparities": int(get_hw("SGBM_NUM_DISPARITIES")),
                "blockSize": int(get_hw("SGBM_BLOCK_SIZE")),
                "P1": SGBM_P1_MULT * 3 * int(get_hw("SGBM_BLOCK_SIZE")) ** 2,
                "P2": SGBM_P2_MULT * 3 * int(get_hw("SGBM_BLOCK_SIZE")) ** 2,
            },
            "SGBM",
        )
        # ── SGBM 动态参数(2026-07-03):runtime 改了 SGBM_NUM_DISPARITIES /
        #    SGBM_BLOCK_SIZE 时,下一行重建 matcher
        self._maybe_rebuild_matcher()
        t_sgbm_start = time.perf_counter()
        # SGBM 在 16 倍定点上输出
        disp_sgbm_x16 = self._matcher.compute(gray_l_sgbm, gray_r_sgbm)
        t_sgbm_end = time.perf_counter()
        _debug_log(
            "processing/stereo_depth.py:compute",
            "sgbm_compute_done",
            {"sgbm_ms": round((t_sgbm_end - t_sgbm_start) * 1000, 1)},
            "SGBM",
        )
        # #endregion
        disp_sgbm = disp_sgbm_x16.astype(np.float32) / 16.0

        # 去无效区域散斑(在 SGBM 分辨率上做,比在 mono 分辨率上做更高效)
        # medianBlur 只支持 uint8，先把视差缩放到 [0,255] 范围
        max_disp = max(1.0, disp_sgbm.max())
        disp_u8 = np.clip(disp_sgbm / max_disp * 255.0, 0, 255).astype(np.uint8)
        if SGBM_MEDIAN_KSIZE > 1:
            disp_u8 = cv2.medianBlur(disp_u8, SGBM_MEDIAN_KSIZE)
        disp_sgbm = disp_u8.astype(np.float32) / 255.0 * max_disp

        # 把 sgbm 视差折回 mono 坐标系的等效视差(像素):
        #   sgbm 像素 = mono 像素 * scale
        #   sgbm 视差(像素) = mono 视差(像素) * scale
        #   -> mono 视差(像素) = sgbm 视差(像素) / scale
        # 然后用 INTER_LINEAR 把 sgbm 视差图放大回 mono 尺寸(不改变每个像素对应的 mono 像素位置),
        # 同时用 cv2.multiply(1/scale) 把视差值折回 mono 坐标系。
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        disp_at_mono = cv2.resize(
            disp_sgbm,
            (MONO_WIDTH, MONO_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        ) * np.float32(inv_scale)
        # resize 把无效区域(< 0)插值出小负值,清零
        disp_at_mono = np.where(disp_at_mono > 0, disp_at_mono, 0.0).astype(np.float32)

        # ── 匹配置信度:视差有效像素比例 × 视差值分布一致性 ────────────────
        with self._state_lock:
            valid = disp_at_mono > 0
            valid_ratio = valid.sum() / float(disp_at_mono.size)
            if valid_ratio > 0.01:
                valid_disp = disp_at_mono[valid]
                cv = float(valid_disp.std() / (valid_disp.mean() + 1e-6))
                self._match_confidence = round(float(np.clip(valid_ratio * (1.0 - cv), 0.0, 1.0)), 3)
            else:
                self._match_confidence = 0.0

        with self._state_lock:
            self._last_disp_at_mono = disp_at_mono
        return disp_at_mono

    def read_depth_in_box(
        self,
        box: tuple[int, int, int, int],
        disp: np.ndarray | None = None,
        grid: int = 40,
        min_valid_ratio: float = 0.005,
        min_valid_absolute: int = 30,
    ) -> Optional[float]:
        """在 cup 检测框内做网格多点采样,取中位数作为深度。

        解决问题:cup 中心点常落在杯柄、杯身反光区、SGBM 不可靠的纹理弱区,
        单点采样会拿到无效视差。框内多点中位数更鲁棒。

        Args:
            box: ``(x1, y1, x2, y2)`` 像素坐标(基于单眼 1920x1080)。
            disp: 视差图;为 None 时用 :meth:`compute` 缓存。
            grid: 保留参数兼容(实际逻辑已改为"遍历 box 全部像素"取有效视差中位数)。
                早期版本用等距网格采样,但 SGBM 有效点聚集(杯身轮廓/杯柄)时
                网格采样命中率极低(80x80 网格 0/6400 = 0%)。
                现在改为直接遍历 box 内全部像素(典型 box 20 万像素,向量化 < 10ms),
                利用 SGBM 算出来的所有有效点,不限位置。
            min_valid_ratio: 框内**有效**像素比例阈值(对大 box 用)。
            min_valid_absolute: 框内**有效**像素绝对数量阈值(2026-06-17 修复,
                对小 box/远距离 cup 用)。
                取 **两者中的较大值**作为实际阈值。

        Returns:
            深度(cm),1 位小数;失败返回 ``None``。
        """
        if disp is None:
            disp = self._last_disp_at_mono
        if disp is None:
            return None

        h, w = disp.shape[:2]
        x1, y1, x2, y2 = box
        # 夹紧到画面内
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # 直接遍历 box 全部像素(向量化, < 10ms)
        region = disp[y1:y2, x1:x2]  # shape (h, w)
        valid_mask = region > 0
        valid_count = int(valid_mask.sum())
        region_size = int(region.size)

        # 动态阈值 (2026-06-17 修复): 比例 vs 绝对值取较大者
        effective_threshold = max(
            int(region_size * min_valid_ratio),
            min_valid_absolute,
        )

        if valid_count == 0:
            return None
        if valid_count < effective_threshold:
            return None

        # 把所有有效视差转成深度,过滤超界值
        valid_d = region[valid_mask]
        f_px = compute_focal_px()
        ds = _disp_scale()
        z = (f_px * float(get_hw("BASELINE_CM"))) / (valid_d * ds)
        in_range = (z >= DEPTH_MIN_CM) & (z <= DEPTH_MAX_CM)
        z_vals = z[in_range]
        valid_d_in_range = valid_d[in_range]

        if len(z_vals) < effective_threshold:
            self._last_d_median = None
            if len(valid_d) > 0:
                _debug_log(
                    "processing/stereo_depth.py:read_depth_in_box",
                    "z_valid count < threshold",
                    {
                        "box": list(box),
                        "valid_count": int(valid_count),
                        "in_range_count": int(in_range.sum()),
                        "threshold": int(effective_threshold),
                    },
                    "JITTER",
                )
            return None

        # 深度跳动修复(2026-06-22):
        # box 里有杯子和背景两种深度的像素混在一起,全局中位数被污染。
        # 用直方图峰值代替全局中位数:
        #   1. 把 z_vals 分进若干等宽 bin
        #   2. 找最高 bin(杯面深度的像素最多)
        #   3. 取该 bin 内像素的中位数
        # 背景像素(深度与杯面差异大)落在低频 bin 里被忽略,
        # 杯子主体像素(深度一致)形成主峰。
        n_bins = max(10, int(len(z_vals) / 500))
        z_min_b, z_max_b = z_vals.min(), z_vals.max()
        if z_max_b - z_min_b < 0.5:
            z_final = z_vals
        else:
            bin_edges = np.linspace(z_min_b, z_max_b, n_bins + 1)
            hist, _ = np.histogram(z_vals, bins=bin_edges)
            peak_bin = int(np.argmax(hist))
            z_low = bin_edges[peak_bin]
            z_high = bin_edges[peak_bin + 1]
            z_final = z_vals[(z_vals >= z_low) & (z_vals < z_high)]
            if len(z_final) < 5:
                z_final = z_vals

        # ── 2D 空间加权 + trimmed mean(2026-07-03,平滑优化)──────────────────
        # 杯面中心比边缘更可靠(中心光照均匀、纹理清晰、视差匹配率高)。
        # 直方图峰值之后留下的"主体像素"绝大多数是杯子表面,但仍有少量
        # 杯柄/杯底/反光边缘的离群点。trim 5% 极端值 + 中心高斯加权双重
        # 抑制残余跳动。
        if len(z_final) >= 20:
            # Trimmed mean:去最大最小各 5%
            z_sorted = np.sort(z_final)
            n_trim = max(1, int(len(z_sorted) * 0.05))
            z_trimmed = z_sorted[n_trim:len(z_sorted) - n_trim] if len(z_sorted) > 2 * n_trim else z_sorted

            # 2D 高斯权重:中心 = 1.0,边缘 = 0.3
            # box 中心 = ((x1+x2)/2, (y1+y2)/2);用 (dx, dy) 距离归一化到 [-1, 1]
            cx_box = (x1 + x2) / 2.0
            cy_box = (y1 + y2) / 2.0
            rx_box = max(1.0, (x2 - x1) / 2.0)
            ry_box = max(1.0, (y2 - y1) / 2.0)

            # 提取 z_final 对应的像素 (yy, xx) 坐标
            # region[yy, xx] > 0 → in_mask。重新做一遍对齐:把 z_final 的 mask
            # 在 in_range 后的 valid_mask 上累加权重。最简实现:用一个均匀权重
            # 数组,后续对裁剪后的 z_final 用 1D 索引(假设 trim 前后顺序稳定)。
            # np.sort 默认 stable,保持 z_final 的原顺序 → 取 z_final 排序前的索引。
            z_orig_idx = np.argsort(z_final)  # 升序排序的索引
            # 重新映射 trimmed 范围到 z_final 索引
            if n_trim > 0 and len(z_final) > 2 * n_trim:
                kept_idx = z_orig_idx[n_trim:len(z_final) - n_trim]
            else:
                kept_idx = z_orig_idx
            # kept_idx 在 z_final 里的位置;再在 region 坐标系里取 (yy, xx)
            ys_in_region, xs_in_region = np.where(valid_mask)
            # 注意:这里 in_range 还会进一步过滤;先在 valid_mask 上统计 z_final 对应的点
            # 因为 z_final ⊆ in_range ⊆ valid_mask,所以 z_final 对应的 (y, x) 是
            # valid_mask 的子集。
            #
            # 做法:对 z_vals 重建 (y_rel, x_rel) 索引。in_range 内的 (y, x) 等于
            # np.where(valid_mask)[0][in_range_in_valid]。
            in_range_in_valid = np.zeros_like(valid_mask)
            in_range_in_valid[valid_mask] = in_range
            # 现在 in_range_in_valid 是和 region 同形状的 bool mask
            ys_ir, xs_ir = np.where(in_range_in_region := in_range_in_valid)
            # z_final 的长度 == len(ys_ir)  (z_final ⊆ in_range)
            if len(ys_ir) == len(z_final):
                # 重新做 trim:z_final 排序前后索引映射
                # 把 z_final 排序后,kept 像素的 (yy, xx) 拿来做高斯加权
                kept_y = ys_ir[kept_idx]
                kept_x = xs_ir[kept_idx]
                # 归一化到 [-1, 1]
                nx = (kept_x - cx_box + (x1 - x1)) / rx_box  # 简化:直接用 region 内坐标
                # 上面 cx_box 已经是绝对坐标,region 内坐标 = 绝对 - x1
                nx = (kept_x - (cx_box - x1)) / rx_box
                ny = (kept_y - (cy_box - y1)) / ry_box
                weights = np.exp(-(nx * nx + ny * ny) / 2.0)
                w_sum = float(weights.sum())
                if w_sum > 0:
                    z_gauss = float(np.sum(z_final[kept_idx] * weights) / w_sum)
                    z_median = float(np.median(z_trimmed))
                    # 加权均值 + 中位数 8:2 混合(均值得主,中位数抗离群)
                    result_z = 0.8 * z_gauss + 0.2 * z_median
                else:
                    result_z = float(np.median(z_trimmed))
            else:
                # 长度对不齐,fall back 到 trimmed median
                result_z = float(np.median(z_trimmed))
        else:
            # 样本太少,直接用中位数
            result_z = float(np.median(z_final))

        d_median = float(np.median(valid_d_in_range))
        with self._state_lock:
            self._last_d_median = d_median
        result = round(float(result_z), 1)
        
        # 距离合理性校验:超出预设工作范围则判定为无效值
        if result < DEPTH_MIN_CM or result > DEPTH_MAX_CM:
            _debug_log(
                "processing/stereo_depth.py:read_depth_in_box",
                "depth_out_of_range_rejected",
                {
                    "box": list(box),
                    "z_returned": result,
                    "depth_range": [DEPTH_MIN_CM, DEPTH_MAX_CM],
                },
                "JITTER",
            )
            self._last_d_median = None
            return None
        
        _debug_log(
            "processing/stereo_depth.py:read_depth_in_box",
            "SUCCESS z returned (histogram peak)",
            {
                "box": list(box),
                "valid_count": int(valid_count),
                "valid_in_range": len(z_vals),
                "d_median": d_median,
                "f_px": f_px,
                "BASELINE_CM": float(get_hw("BASELINE_CM")),
                "DISP_SCALE": ds,
                "z_peak": float(np.median(z_final)),
                "z_weighted": result_z,
                "z_returned": result,
                "suggested_SCALE_for_realZ_25cm": (f_px * float(get_hw("BASELINE_CM"))) / (25.0 * d_median),
                "suggested_SCALE_for_realZ_30cm": (f_px * float(get_hw("BASELINE_CM"))) / (30.0 * d_median),
            },
            "JITTER",
        )
        return result

    def smoothed_depth(self, current_depth_cm: Optional[float]) -> Optional[float]:
        """把当前帧深度推入 EMA 平滑器,返回置信度加权的指数移动均值。

        解决"静止时数值闪烁"问题的核心:
        - 旧方案:窗口均值+离群值拒绝，差帧的随机误差仍然被累加进均值
        - 新方案:EMA + 置信度自适，conf 越高 alpha 越大 → 高质量帧主导输出，
          低质量帧贡献被压到 5-15%，从根本上消除闪烁

        EMA 公式:
        - ema = conf * alpha * value + (1 - conf * alpha) * ema
        - conf=1 时 alpha=0.2 → 收敛到 11 帧窗口的稳态分布
        - conf=0.5 时 alpha=0.1 → 低质量帧权重再减半
        """
        MIN_ALPHA = 0.05   # 单帧最低权重(极低置信时)
        BASE_ALPHA = 0.20 # 高置信帧的基础权重(稳态 ~11 帧等效)
        MIN_CONF_FOR_UPDATE = 0.10  # 置信度低于此值则跳过更新

        if current_depth_cm is None:
            with self._state_lock:
                self._last_raw_depth_cm = None
                return round(self._ema_value, 1) if self._ema_value is not None else None

        incoming = float(current_depth_cm)
        with self._state_lock:
            conf = max(MIN_CONF_FOR_UPDATE, self._match_confidence)

            alpha = max(MIN_ALPHA, BASE_ALPHA * conf)

            if self._ema_value is None:
                self._ema_value = incoming
                self._ema_confidence = conf
            else:
                self._ema_value = (
                    alpha * incoming + (1.0 - alpha) * self._ema_value
                )
                self._ema_confidence = (
                    alpha * conf + (1.0 - alpha) * self._ema_confidence
                )

            raw_result = round(self._ema_value, 1)
            self._last_raw_depth_cm = raw_result

        # 分段线性插值校正(内置 4 个校准点)
        if len(self._calib_points) >= 2:
            return round(self._interpolate(raw_result), 1)
        return raw_result

    def reset_smoothing(self) -> None:
        """清空 EMA 状态(用于相机切换、标定变化等场景)。"""
        self._ema_value = None
        self._ema_confidence = 1.0
        self._consecutive_rejects = 0
        self._last_raw_depth_cm = None

    @property
    def last_d_median(self) -> Optional[float]:
        """上一次 read_depth_in_box 成功时的原始视差中位数(px),供校准用。"""
        return self._last_d_median

    @property
    def last_raw_depth_cm(self) -> Optional[float]:
        """上一次 smoothed_depth 返回的未校正深度(cm),用于校准 API。"""
        return self._last_raw_depth_cm

    @property
    def focal_px(self) -> float:
        return self._focal_px

    @property
    def baseline_m(self) -> float:
        return self._baseline_m

    @property
    def match_confidence(self) -> float:
        """上一次 SGBM 匹配置信度(0-1)。低光照/剧烈光影变化时下降。

        用法:
        - > 0.7: 高置信，深度值可靠
        - 0.4–0.7: 中等置信，深度值可参考但有波动
        - < 0.4: 低置信，深度值可能不可靠，建议触发平滑或报警
        """
        return self._match_confidence

    # ── 分段线性校准层 ─────────────────────────────────────────────────────────

    def _interpolate(self, sensor_z: float) -> float:
        """对原始传感器读数 sensor_z 做分段线性插值校正。

        calib_points = [(s0,t0), (s1,t1), ...] 按 s 升序排列。
        - 少于 2 点: 返回原始值(无校正)。
        - sensor_z <= 第一个点 s0: 用 s0-t0 斜率外推。
        - sensor_z >= 最后一个点 sn: 用 sn-tn 斜率外推。
        - 中间段: 线性插值。
        """
        pts = self._calib_points
        if len(pts) < 2 or sensor_z is None:
            return sensor_z

        if sensor_z <= pts[0][0]:
            s0, t0 = pts[0]
            if len(pts) == 2:
                s1, t1 = pts[1]
            else:
                s1, t1 = pts[1]
            slope = (t1 - t0) / max(s1 - s0, 0.001)
            return t0 + slope * (sensor_z - s0)

        if sensor_z >= pts[-1][0]:
            s_prev, t_prev = pts[-2] if len(pts) >= 2 else pts[-1]
            s_last, t_last = pts[-1]
            slope = (t_last - t_prev) / max(s_last - s_prev, 0.001)
            return t_last + slope * (sensor_z - s_last)

        # 二分查找所在段
        lo, hi = 0, len(pts) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if pts[mid][0] <= sensor_z:
                lo = mid
            else:
                hi = mid
        s0, t0 = pts[lo]
        s1, t1 = pts[hi]
        t = t0 + (t1 - t0) * (sensor_z - s0) / max(s1 - s0, 0.001)
        return t

    def add_calibration_point(self, sensor_z_cm: float, true_z_cm: float) -> list[tuple[float, float]]:
        """添加一个校准点(sensor_reading, true_depth)。

        不做拟合,只追加点;由 smoothed_depth 输出端调用 _interpolate 实时校正。

        Returns: 排序后的校准点列表。
        """
        if sensor_z_cm <= 0 or true_z_cm <= 0:
            return self._calib_points
        # 去重: 同一 sensor_z 只保留最后一个
        self._calib_points = [(s, t) for (s, t) in self._calib_points if s != float(sensor_z_cm)]
        self._calib_points.append((float(sensor_z_cm), float(true_z_cm)))
        self._calib_points.sort(key=lambda p: p[0])
        _debug_log(
            "processing/stereo_depth.py:add_calibration_point",
            "calibration point added",
            {
                "sensor_z_cm": sensor_z_cm,
                "true_z_cm": true_z_cm,
                "all_points": [(round(s, 2), round(t, 2)) for s, t in self._calib_points],
            },
            "CALIB",
        )
        return self._calib_points

    @property
    def calib_points(self) -> list[tuple[float, float]]:
        """线程安全读取校准点列表。"""
        with self._state_lock:
            return list(self._calib_points)


def map_right_compute_roi(
    left_box: tuple[int, int, int, int],
    min_disp: int,
    num_disp: int,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """由左图检测框推导右图 SGBM 计算用 ROI（几何映射）。

    核心映射公式:
    - 竖直方向(y轴): 极线校正后左右图行严格对齐，直接复用左图 y 坐标，
      上下各加 10px padding 兜底标定微小误差
    - 水平方向(x轴): 按最大视差范围向左扩展，保证所有深度的目标对应点
      都落在 ROI 内

    :param left_box: 左图检测框 (lx1, ly1, lx2, ly2)
    :param min_disp: SGBM 参数 minDisparity
    :param num_disp: SGBM 参数 numDisparities
    :param img_width: 右图图像宽度
    :param img_height: 右图图像高度
    :return: right_compute_roi (rx1, ry1, rx2, ry2)
    """
    lx1, ly1, lx2, ly2 = left_box

    # 竖直方向: 复用左图 y 坐标，上下各加 10px padding
    ry1 = max(0, ly1 - 10)
    ry2 = min(img_height, ly2 + 10)

    # 水平方向: 按最大视差范围向左扩展
    # 目标在右图最靠左的位置 = 左图 x1 - 最大视差
    # 目标在右图最靠右的位置 = 左图 x2 - 最小视差
    rx1 = max(0, lx1 - (min_disp + num_disp))
    rx2 = min(img_width, lx2 - min_disp)

    return (rx1, ry1, rx2, ry2)


def map_right_display_box(
    left_box: tuple[int, int, int, int],
    avg_disp: float,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    """由左图检测框 + 当前帧平均视差，推导右图显示用示意框（仅用于前端绘制）。

    映射公式:
    - 竖直方向: 和左图检测框完全一致
    - 水平方向: 整体向左平移平均视差，框的宽高与左图完全相同

    :param left_box: 左图检测框 (lx1, ly1, lx2, ly2)
    :param avg_disp: 当前帧目标主体平均视差（像素）
    :param img_width: 右图图像宽度
    :param img_height: 右图图像高度
    :return: right_display_box (dx1, dy1, dx2, dy2)
    """
    lx1, ly1, lx2, ly2 = left_box

    # 竖直方向: 和左图检测框完全一致
    dy1 = ly1
    dy2 = ly2

    # 水平方向: 整体向左平移平均视差，框的宽高与左图完全相同
    dx1 = max(0, int(lx1 - avg_disp))
    dx2 = min(img_width, int(lx2 - avg_disp))

    return (dx1, dy1, dx2, dy2)


__all__ = ["StereoDepthSolver", "map_right_compute_roi", "map_right_display_box"]


if __name__ == "__main__":
    # 公式自检:不依赖摄像头
    print("formula self-check (live values):")
    print(f"  Z_cm(d) = (f_px * BASELINE_CM) / (d * DISP_SCALE)")
    _b = float(get_hw("BASELINE_CM"))
    _s = float(get_hw("DISP_SCALE"))
    print(f"  f_px = {compute_focal_px():.2f}, BASELINE_CM = {_b} "
          f"(default = {BASELINE_CM}), DISP_SCALE = {_s} (default = {DISP_SCALE})")
    for d in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        z_cm_real = (compute_focal_px() * _b) / (d * _s)
        # 反推:如果看到 z_cm_real,需要多少视差
        print(f"  d={d:6.1f} px -> Z_cm = {z_cm_real:7.2f} cm")
    print(f"  DEPTH_SMOOTH_WINDOW={DEPTH_SMOOTH_WINDOW}, "
          f"DEPTH_RANGE=[{DEPTH_MIN_CM}, {DEPTH_MAX_CM}] cm")
    print("OK")
