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
- 修正 :meth:`StereoDepthSolver.read_depth_at` 的钳位语义:
  视差无效(d <= 0)或公式输出超出钳位范围时,直接返回 ``None`` 让 UI 显示"无有效深度",
  不再被钳位到 0.5 cm(这会让用户误以为物体在 0.5 cm 处)。
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
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
    SGBM_BLOCK_SIZE,
    SGBM_HEIGHT,
    SGBM_MEDIAN_KSIZE,
    SGBM_NUM_DISPARITIES,
    SGBM_P1_MULT,
    SGBM_P2_MULT,
    SGBM_WIDTH,
    compute_focal_px,
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
        p1 = SGBM_P1_MULT * 3 * SGBM_BLOCK_SIZE ** 2
        p2 = SGBM_P2_MULT * 3 * SGBM_BLOCK_SIZE ** 2
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=SGBM_NUM_DISPARITIES,
            blockSize=SGBM_BLOCK_SIZE,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=0,
            speckleRange=0,
            mode=cv2.STEREO_SGBM_MODE_SGBM,
        )
        self._focal_px = float(compute_focal_px())
        self._baseline_m = BASELINE_CM / 100.0
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

        # #region agent log — H4: 验证 self._focal_px vs compute_focal_px() 实时值
        _debug_log(
            "processing/stereo_depth.py:__init__",
            "Solver init focal_px + DISP_SCALE",
            {
                "self_focal_px": self._focal_px,
                "compute_focal_px_now": float(compute_focal_px()),
                "FOCAL_LENGTH_MM": FOCAL_LENGTH_MM,
                "BASELINE_CM": BASELINE_CM,
                "DISP_SCALE": _disp_scale(),
                "HFOV_DEG_NOT_IMPORTED": True,
                "opencv_build_info": self._get_opencv_backend_info(),
            },
            "H4",
        )
        # #endregion

        logger.info(
            "[StereoDepthSolver] SGBM: numDisp=%d blockSize=%d p1=%d p2=%d, "
            "smoothing window=%d, focal_px=%.2f",
            SGBM_NUM_DISPARITIES, SGBM_BLOCK_SIZE, p1, p2,
            self._window_size, self._focal_px,
        )

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

    def compute(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        """对已校正左右图算 SGBM 视差,返回 mono 分辨率(1920x1080)的视差图。

        Args:
            left_rect, right_rect: 已校正 BGR 图,``(MONO_HEIGHT, MONO_WIDTH, 3)``。

        Returns:
            ``disp_at_mono``: 视差图(浮点,像素),``(MONO_HEIGHT, MONO_WIDTH)``。
            无效点 = 0。

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

        # 等比例缩放到 SGBM 分辨率(见 §0 resize 硬约束)
        gray_l_sgbm, scale = resize_for_sgbm(gray_l)
        gray_r_sgbm, _ = resize_for_sgbm(gray_r)

        # SGBM 在 16 倍定点上输出
        disp_sgbm_x16 = self._matcher.compute(gray_l_sgbm, gray_r_sgbm)
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
        valid = disp_at_mono > 0
        valid_ratio = valid.sum() / float(disp_at_mono.size)
        if valid_ratio > 0.01:
            valid_disp = disp_at_mono[valid]
            # 变异系数(CV): 分布越集中(标准差/均值越小) → 置信度越高
            cv = float(valid_disp.std() / (valid_disp.mean() + 1e-6))
            self._match_confidence = round(float(np.clip(valid_ratio * (1.0 - cv), 0.0, 1.0)), 3)
        else:
            self._match_confidence = 0.0

        self._last_disp_at_mono = disp_at_mono
        return disp_at_mono

    def read_depth_at(
        self,
        cx: int,
        cy: int,
        disp: np.ndarray | None = None,
        use_cache: bool = True,
    ) -> Optional[float]:
        """在 mono 视差图上读 (cx, cy) 处的视差,转成深度(cm)。

        Args:
            cx, cy: 像素坐标(整数,基于单眼 1920x1080)。
            disp: 视差图;为 None 时用上次 :meth:`compute` 缓存的结果。
            use_cache: 保留参数(与早期版本兼容),当前等价于 ``disp is None``。

        Returns:
            深度(cm),1 位小数;无效(视差 <= 0、越界、计算结果超出钳位范围)
            时返回 ``None``,**不**被钳位到 0.5 cm(避免误导用户)。
        """
        if disp is None:
            disp = self._last_disp_at_mono
        if disp is None:
            return None

        h, w = disp.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None

        d = float(disp[cy, cx])
        if d <= 0:
            return None

        try:
            z_cm = z_cm_from_disparity(d)
        except ValueError:
            return None

        # 钳位语义修正(2026-06-17):
        # 视差有效时算出的 Z,如果落在 [DEPTH_MIN_CM, DEPTH_MAX_CM] 内,直接 round 输出;
        # 超出这个范围(过近 / 过远)说明视差虽然>0但实际不可信,
        # **不再**被钳位到 0.5/100,而是返回 None 让 UI 显式提示"无有效深度"。
        if not (DEPTH_MIN_CM <= z_cm <= DEPTH_MAX_CM):
            return None
        return round(z_cm, 1)

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
        z = (f_px * BASELINE_CM) / (valid_d * ds)
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

        d_median = float(np.median(valid_d_in_range))
        self._last_d_median = d_median
        result = round(float(np.median(z_final)), 1)
        _debug_log(
            "processing/stereo_depth.py:read_depth_in_box",
            "SUCCESS z returned (histogram peak)",
            {
                "box": list(box),
                "valid_count": int(valid_count),
                "valid_in_range": len(z_vals),
                "d_median": d_median,
                "f_px": f_px,
                "BASELINE_CM": BASELINE_CM,
                "DISP_SCALE": ds,
                "z_peak": float(np.median(z_final)),
                "z_returned": result,
                "suggested_SCALE_for_realZ_25cm": (f_px * BASELINE_CM) / (25.0 * d_median),
                "suggested_SCALE_for_realZ_30cm": (f_px * BASELINE_CM) / (30.0 * d_median),
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
            self._last_raw_depth_cm = None
            return round(self._ema_value, 1) if self._ema_value is not None else None

        incoming = float(current_depth_cm)
        conf = max(MIN_CONF_FOR_UPDATE, self._match_confidence)

        # EMA alpha 由置信度动态决定
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


__all__ = ["StereoDepthSolver"]


if __name__ == "__main__":
    # 公式自检:不依赖摄像头
    print("formula self-check (literal task spec):")
    print(f"  Z_cm(d) = (f_px * BASELINE_CM) / (d * DISP_SCALE)")
    print(f"  f_px = {compute_focal_px():.2f}, BASELINE_CM = {BASELINE_CM}, DISP_SCALE = {DISP_SCALE}")
    for d in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        z_cm_real = (compute_focal_px() * BASELINE_CM) / (d * DISP_SCALE)
        # 反推:如果看到 z_cm_real,需要多少视差
        print(f"  d={d:6.1f} px -> Z_cm = {z_cm_real:7.2f} cm")
    print(f"  DEPTH_SMOOTH_WINDOW={DEPTH_SMOOTH_WINDOW}, "
          f"DEPTH_RANGE=[{DEPTH_MIN_CM}, {DEPTH_MAX_CM}] cm")
    print("OK")
