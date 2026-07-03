"""VR Box SBS 一帧总入口管线。

按任务规约把阶段 0-3 串成一条:

    camera.read_rectified_pair()  -- 阶段 0
        -> detector.detect(left)   -- 阶段 1 (左眼)
        -> detector.detect(right)  -- 阶段 1 (右眼, 独立跑!)
        -> solver.compute()        -- 阶段 2
        -> solver.read_depth_in_box()  -- 阶段 2 (框内多点采样, 左右眼各算一次取均值)
        -> solver.smoothed_depth()     -- 阶段 2 (防抖)
        -> warning.render/render_text  -- 阶段 3
    -> cv2.hconcat([left, right])   -- 阶段 4: 3840x1080 SBS

约束:
- 不缩放、不裁切、不留白:输出严格 (SBS_HEIGHT, SBS_WIDTH, 3) = (1080, 3840, 3)。
- 异常隔离:任意子步骤失败,继续后续(不让管线挂掉),只记录 warning。

Bug fix(2026-06-17):
- 旧版:左右眼共用一个 YOLO box。问题:右眼视角下 box 位置/形状不对(杯柄漏框)。
- 新版:左右眼各跑一次 YOLO,保证每个视图里 box 都准确包裹 cup。
- 旧版:单点采样 + 公式输出被钳位到 0.5 cm。问题:cup 中心点常落在反光区,测距持续 0.5。
- 新版:框内 5x5 网格中位数采样 + 左右眼深度平均,无效时返回 None 让 UI 显式提示。
"""
from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Optional
import json
import os

import cv2
import numpy as np

# Ring buffer size: keep last 60 frame stats for live stats display
_STATS_RING_SIZE = 60

# Debug logging (session=c7ffa8)
_DEBUG_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "debug-c7ffa8.log",
)


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "?") -> None:
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
    MONO_HEIGHT,
    MONO_WIDTH,
    SBS_HEIGHT,
    SBS_WIDTH,
)
from processing.detector import CupDetector
from processing.stereo_depth import StereoDepthSolver
from processing.warning import Box, WarningOverlay

logger = logging.getLogger(__name__)


def _fuse_depths(depths: list[Optional[float]]) -> Optional[float]:
    """融合左右眼各自算出的深度值。

    规则:
    - 全部 None -> None
    - 只有一个有效 -> 返回那一个
    - 两个都有效 -> 取均值(理论上左右眼视差几何对称,均值更稳)

    Args:
        depths: 深度值列表(允许 None)。

    Returns:
        融合后的深度(cm),1 位小数;全部无效时 None。
    """
    valid = [d for d in depths if d is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return round(valid[0], 1)
    return round(float(np.mean(valid)), 1)


class SBSPipeline:
    """一帧总入口:raw -> 校正 -> YOLO(双眼) -> SGBM -> 预警 -> 3840x1080 SBS.

    解耦后的数据流:
        LatestFrameQueue  ──pull()──  process_one_frame()
                                                     │
                                         push(SBS frame) to
                                                     │
                                                  LatestSBSQueue
                                                     ▲
                           MJPEG / WebRTC ──────────┘

    旧架构中每个 MJPEG 请求 / 每个 WebRTC 客户端独立跑 YOLO×2 + SGBM,
    现在所有消费者共享 SBSPipeline 的处理结果。

    Args:
        frame_queue: LatestFrameQueue 实例,摄像头线程往里写已校正的左右图。
        sbs_queue: LatestSBSQueue 实例,管线处理完后往里写 SBS 帧。
        detector: cup 检测器(单例)。
        solver: SGBM 视差 + 深度解算器。
        warn: 预警渲染器。
    """

    def __init__(
        self,
        frame_queue,
        sbs_queue,
        detector: CupDetector,
        solver: StereoDepthSolver,
        warn: WarningOverlay,
    ) -> None:
        self._frame_queue = frame_queue
        self._sbs_queue = sbs_queue
        self._detector = detector
        self._solver = solver
        self._warn = warn

        # 帧统计(供调试)
        self._frames = 0
        self._last_latency_ms: float = 0.0
        self._last_stage_times: dict[str, float] = {}

        # Live stats ring buffer (thread-safe)
        self._stats_lock = Lock()
        self._stats_ring: deque[dict] = deque(maxlen=_STATS_RING_SIZE)

        # ── Leader's #7: Decouple YOLO from per-frame processing ──────
        # YOLO is the most expensive step (~80ms/call × 2 eyes). Running it
        # every frame is wasteful. Instead, refresh YOLO detections every
        # ``DETECT_REFRESH_INTERVAL`` frames and reuse the cached boxes on
        # in-between frames. The cup typically moves slowly; a stale box
        # for a few frames is acceptable.
        self._DETECT_REFRESH_INTERVAL = 2   # refresh every 2 frames
        self._cached_box_l: Optional[Box] = None
        self._cached_box_r: Optional[Box] = None
        self._frames_since_detect: int = 0   # counter (also det-skip rate visible in logs)

    def process_one_frame(
        self,
        left: Optional[np.ndarray] = None,
        right: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """处理一帧。

        双入口模式:
        - 有 left/right 参数时:直接用这两帧(standalone profiler 用,无需队列)。
        - 无参数时:从 LatestFrameQueue 拉最新帧(web 服务用,摄像头线程往里写)。

        Returns:
            SBS BGR 图 (1080, 3840, 3) 或 None(无新帧时跳过)。
        """
        # 0. 获得已校正帧
        if left is not None and right is not None:
            # 直接喂帧模式(standalone profiler / 测试用)
            rect = (left, right)
        else:
            # 队列模式(生产服务用)
            rect = self._frame_queue.pull()
            if rect is None:
                return None  # 摄像头速率 > 处理速率时,旧帧被静默丢弃

        left, right = rect
        self._frames += 1
        t0 = time.perf_counter()

        # 防御:确保 shape 是 mono 尺寸
        if left.shape[:2] != (MONO_HEIGHT, MONO_WIDTH):
            logger.error(
                "[SBSPipeline] unexpected left shape %s, expected (%d, %d)",
                left.shape[:2], MONO_HEIGHT, MONO_WIDTH,
            )
            return None

        box_l: Optional[Box] = None
        box_r: Optional[Box] = None
        depth_cm: Optional[float] = None

        t0 = time.perf_counter()
        t_yolo_l = t_yolo_r = t_sgbm = t_depth = t_render = 0.0

        # 2. YOLO cup 检测 - 解耦(Leader's #7)
        #    每 N 帧刷新一次,其他帧复用缓存的 box → 把 ~80ms 的 YOLO 成本从
        #    每帧压低到每 2-3 帧一次。cup 通常移动很慢,几个 frame 内 box 漂移可忽略。
        t1 = time.perf_counter()
        need_detect = (
            self._cached_box_l is None
            or self._cached_box_r is None
            or self._frames_since_detect >= self._DETECT_REFRESH_INTERVAL
        )
        if need_detect:
            try:
                self._cached_box_l, _ = self._detector.detect(left)
            except Exception as ex:  # noqa: BLE001
                logger.warning("[SBSPipeline] detector(left) failed: %s", ex)
                self._cached_box_l = None
            try:
                self._cached_box_r, _ = self._detector.detect(right)
            except Exception as ex:  # noqa: BLE001
                logger.warning("[SBSPipeline] detector(right) failed: %s", ex)
                self._cached_box_r = None
            self._frames_since_detect = 0
        else:
            self._frames_since_detect += 1
        box_l = self._cached_box_l
        box_r = self._cached_box_r
        t2 = time.perf_counter()
        t_yolo_l = (t2 - t1) / 2 if need_detect else 0.0
        t_yolo_r = t_yolo_l

        # #region agent log — H3: 记录 YOLO 在左右眼返回的 box (定位"box 不在 cup 上"问题)
        _debug_log(
            "processing/sbs_pipeline.py:after_detect",
            "YOLO box_l / box_r (cached={})".format(not need_detect),
            {
                "box_l": list(box_l) if box_l is not None else None,
                "box_r": list(box_r) if box_r is not None else None,
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
            },
            "H3",
        )
        # #endregion

        # 3. SGBM 视差 - ROI-bounded(Leader's #6)
        #    有 box → 在 box 周围做 ROI-裁剪 + ROI-SGBM,效率高、噪声小
        #    无 box → 退化到整图 SGBM(向后兼容)
        t4 = time.perf_counter()
        try:
            if box_l is not None or box_r is not None:
                disp = self._solver.compute_roi(left, right, box_l=box_l, box_r=box_r)
            else:
                roi_mask = self._build_roi_mask(box_l, box_r)
                disp = self._solver.compute(left, right, roi_mask=roi_mask)
        except Exception as ex:  # noqa: BLE001
            logger.warning("[SBSPipeline] solver.compute failed: %s", ex)
            disp = None
        t5 = time.perf_counter()
        t_sgbm = t5 - t4

        # 4. 深度解算 - 框内多点采样(Bug fix 2026-06-17)
        #    左右眼各算一次取平均:视差几何上左右眼是对称的,各算一次能消除单眼 SGBM 漏匹配。
        #    这里用左右眼各自的 box,因为左右眼的 cup 位置/大小不完全一致。
        t6 = time.perf_counter()
        if disp is not None:
            depth_l = self._solver.read_depth_in_box(box_l, disp=disp) if box_l is not None else None
            depth_r = self._solver.read_depth_in_box(box_r, disp=disp) if box_r is not None else None
            d_raw = _fuse_depths([depth_l, depth_r])
            if d_raw is not None:
                depth_cm = self._solver.smoothed_depth(d_raw)
        t7 = time.perf_counter()
        t_depth = t7 - t6

        # 5. 渲染:预警框(各眼自己的 box) + 距离文字(左右眼都画,VR 对称)
        t8 = time.perf_counter()
        try:
            self._warn.render(left, box_l, depth_cm)
            self._warn.render(right, box_r, depth_cm)
            self._warn.render_text(left, depth_cm)
            self._warn.render_text(right, depth_cm)
        except Exception as ex:  # noqa: BLE001
            logger.warning("[SBSPipeline] warning render failed: %s", ex)
        t9 = time.perf_counter()
        t_render = t9 - t8

        # 6. 拼 SBS(无 resize、无裁切,严格 3840x1080)
        sbs = cv2.hconcat([left, right])
        if sbs.shape != (SBS_HEIGHT, SBS_WIDTH, 3):
            # 防御:任何情况下不输出错误形状
            logger.error(
                "[SBSPipeline] SBS shape %s != expected (%d, %d, 3)",
                sbs.shape, SBS_HEIGHT, SBS_WIDTH,
            )
            return None

        t_end = time.perf_counter()
        total_ms = (t_end - t0) * 1000.0

        self._last_latency_ms = total_ms
        self._last_stage_times = {
            "yolo_l_ms": round(t_yolo_l * 1000, 1),
            "yolo_r_ms": round(t_yolo_r * 1000, 1),
            "sgbm_ms": round(t_sgbm * 1000, 1),
            "depth_ms": round(t_depth * 1000, 1),
            "render_ms": round(t_render * 1000, 1),
            "total_ms": round(total_ms, 1),
        }

        # #region agent log — 帧级性能日志
        _debug_log(
            "processing/sbs_pipeline.py:frame_perf",
            "frame latency breakdown",
            {
                "t_yolo_l_ms": round(t_yolo_l * 1000, 1),
                "t_yolo_r_ms": round(t_yolo_r * 1000, 1),
                "t_sgbm_ms": round(t_sgbm * 1000, 1),
                "t_depth_ms": round(t_depth * 1000, 1),
                "t_render_ms": round(t_render * 1000, 1),
                "t_total_ms": round(total_ms, 1),
                "fps_approx": round(1000.0 / total_ms, 1) if total_ms > 0 else 0,
                "depth_cm": depth_cm,
                "box_l": list(box_l) if box_l is not None else None,
                "box_r": list(box_r) if box_r is not None else None,
                "frame": self._frames,
            },
            "PERF",
        )
        # #endregion

        # Push to ring buffer for live stats (thread-safe)
        d_median = self._solver.last_d_median
        match_conf = self._solver.match_confidence
        record = {
            "frame": self._frames,
            "total_ms": round(total_ms, 1),
            "yolo_l_ms": round(t_yolo_l * 1000, 1),
            "yolo_r_ms": round(t_yolo_r * 1000, 1),
            "sgbm_ms": round(t_sgbm * 1000, 1),
            "depth_ms": round(t_depth * 1000, 1),
            "render_ms": round(t_render * 1000, 1),
            "depth_cm": depth_cm,
            "match_confidence": match_conf,
            "fps_approx": round(1000.0 / total_ms, 1) if total_ms > 0 else 0,
            "d_median": d_median,
            "ts": time.time(),
        }
        with self._stats_lock:
            self._stats_ring.append(record)

        # 7. 推 SBS 到共享队列(MJPEG/WebRTC 从这里读,不再各自跑管线)
        self._sbs_queue.push(sbs)

        return sbs

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def last_stage_times(self) -> dict[str, float]:
        return self._last_stage_times

    @property
    def frames_processed(self) -> int:
        return self._frames

    def stats_summary(self) -> dict:
        """返回最近 N 帧的统计摘要（用于实时性能监控页面）。"""
        with self._stats_lock:
            ring = list(self._stats_ring)

        if not ring:
            return {"fps": 0, "latency_p50_ms": 0, "latency_p95_ms": 0,
                    "latency_max_ms": 0, "frames_served": self._frames,
                    "frames_in_window": 0}

        totals = [r["total_ms"] for r in ring]
        totals_sorted = sorted(totals)
        n = len(totals_sorted)

        # FPS based on time span of window (robust against gaps)
        if n > 1:
            window_sec = ring[-1]["ts"] - ring[0]["ts"]
            fps = n / window_sec if window_sec > 0 else 0
        else:
            fps = 0

        def pct(arr, p):
            idx = max(0, int(len(arr) * p / 100) - 1)
            return round(arr[idx], 1)

        from config.hardware import compute_focal_px as _fp, BASELINE_CM as _b
        fp = float(_fp())
        b  = float(_b)

        # Last depth record with raw (un-calibrated) sensor reading
        last = ring[-1]
        d_med = last.get("d_median")
        # z_measured_cm = 未校正的原始传感器读数(用于校准计算)
        z_raw = self._solver.last_raw_depth_cm

        suggested_scales = {}
        if d_med and d_med > 0 and z_raw and z_raw > 0:
            # Real Z known by user → SCALE = z_raw / z_displayed = z_real / z_measured
            # => SCALE = real_cm / z_measured
            for ref_cm in [10, 15, 20, 25, 30, 40, 50, 60]:
                suggested_scales[ref_cm] = round((fp * b) / (ref_cm * d_med), 3)

        return {
            "fps": round(fps, 1),
            "latency_p50_ms": pct(totals_sorted, 50),
            "latency_p95_ms": pct(totals_sorted, 95),
            "latency_max_ms": round(max(totals), 1),
            "latency_avg_ms": round(sum(totals) / n, 1),
            "frames_served": self._frames,
            "frames_in_window": n,
            "stages": {
                "yolo_l_ms": round(sum(r["yolo_l_ms"] for r in ring) / n, 1),
                "yolo_r_ms": round(sum(r["yolo_r_ms"] for r in ring) / n, 1),
                "sgbm_ms": round(sum(r["sgbm_ms"] for r in ring) / n, 1),
                "depth_ms": round(sum(r["depth_ms"] for r in ring) / n, 1),
                "render_ms": round(sum(r["render_ms"] for r in ring) / n, 1),
            },
            "depth_cm": last.get("depth_cm"),
            "d_median_px": d_med,
            "z_measured_cm": z_raw,
            "match_confidence": self._solver.match_confidence,
            "calib_points": [(round(s, 2), round(t, 2)) for s, t in self._solver.calib_points],
            "suggested_disp_scales": suggested_scales,
        }

    def calibrate_depth(self, sensor_z_cm: float, true_z_cm: float) -> dict:
        """添加一个分段线性校准点。

        传感器在 sensor_z_cm (未校正原始读数)处,
        但真实距离是 true_z_cm;追加到校准表,
        由 smoothed_depth 输出端自动用分段线性插值校正。

        Args:
            sensor_z_cm: 传感器未校正原始读数。
            true_z_cm: 杯子真实物理距离(cm)。
        Returns:
            {"calib_points": 所有校准点列表}
        """
        pts = self._solver.add_calibration_point(sensor_z_cm, true_z_cm)
        _debug_log(
            "processing/sbs_pipeline.py:calibrate_depth",
            "calibration point added",
            {
                "sensor_z_cm": sensor_z_cm,
                "true_z_cm": true_z_cm,
                "all_points": [(round(s, 2), round(t, 2)) for s, t in pts],
            },
            "CALIB",
        )
        return {"calib_points": pts, "sensor_z_cm": sensor_z_cm, "true_z_cm": true_z_cm}

    # ── ROI mask 辅助 ─────────────────────────────────────────────────────────

    ROI_MARGIN_RATIO: float = 0.20  # box 外扩 20%

    def _build_roi_mask(
        self,
        box_l: Optional[tuple[int, int, int, int]],
        box_r: Optional[tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        """把左右检测框合并成一个 ROI mask。

        Returns:
            shape=(MONO_HEIGHT, MONO_WIDTH), dtype=uint8。
            255=感兴趣区域,0=忽略区域。
            两框都没有时返回 None（退化为整图 SGBM）。
        """
        if box_l is None and box_r is None:
            return None

        mask = np.zeros((MONO_HEIGHT, MONO_WIDTH), dtype=np.uint8)

        def _fill_box(box: tuple[int, int, int, int]) -> None:
            x1, y1, x2, y2 = box
            # 外扩 MARGIN%
            mx = int((x2 - x1) * self.ROI_MARGIN_RATIO)
            my = int((y2 - y1) * self.ROI_MARGIN_RATIO)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(MONO_WIDTH, x2 + mx)
            y2 = min(MONO_HEIGHT, y2 + my)
            mask[y1:y2, x1:x2] = 255

        if box_l is not None:
            _fill_box(box_l)
        if box_r is not None:
            _fill_box(box_r)

        return mask


__all__ = ["SBSPipeline"]
