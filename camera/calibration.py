"""双目立体校正器(StereoCalibrator)。

任务规约约束:
- 仅读取外部产出的 npz 标定文件,不做棋盘格采集/标定生成。
- 标定坐标系 = 单眼 1920x1080(单设备 SBS 模式下,左/右单眼图在 ``USBCamera``
  拆出后立刻进入本校正器)。
- 校正后左右图同 y 行应在极线上对齐(垂直视差 < 1 px)。

npz 文件约定(优先按以下 key 顺序查找):
  1. ``map1_l, map2_l, map1_r, map2_r`` —— 现成 remap 映射表(优先)
  2. ``K_l, D_l, K_r, D_r, R, T`` —— 内参+畸变+旋转+平移,运行时算 remap
  3. ``Q`` —— 重投影矩阵(可选,供 ``StereoDepthSolver`` 推导像素焦距)

若 npz 文件不存在,``StereoCalibrator`` 进入 "无校正" 降级模式:
  - 视差靠纯 SGBM 在原图上算(精度会差),程序不崩。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.hardware import CALIB_NPZ_PATH, MONO_HEIGHT, MONO_WIDTH

logger = logging.getLogger(__name__)


class StereoCalibrator:
    """双目立体校正器。

    Args:
        calib_npz_path: 标定 npz 文件路径,默认从 ``config.hardware.CALIB_NPZ_PATH`` 取。
        mono_width / mono_height: 期望的单眼图分辨率,默认 1920x1080。

    Attributes:
        is_rectified (bool): 是否成功加载了 remap 映射(``True``)/降级模式(``False``)。
        map1_l, map2_l, map1_r, map2_r: remap 映射表(``is_rectified=False`` 时为 None)。
        Q: 重投影矩阵(若 npz 里有)。
    """

    def __init__(
        self,
        calib_npz_path: Path | None = None,
        mono_width: int = MONO_WIDTH,
        mono_height: int = MONO_HEIGHT,
    ) -> None:
        self._calib_path = Path(calib_npz_path) if calib_npz_path is not None else CALIB_NPZ_PATH
        self._mono_w = mono_width
        self._mono_h = mono_height

        self.is_rectified: bool = False
        self.map1_l: Optional[np.ndarray] = None
        self.map2_l: Optional[np.ndarray] = None
        self.map1_r: Optional[np.ndarray] = None
        self.map2_r: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None

        self._load()

    def _load(self) -> None:
        if not self._calib_path.exists():
            logger.warning(
                "[StereoCalibrator] No calibration file at %s, "
                "running in UNRECTIFIED mode (degraded accuracy)",
                self._calib_path,
            )
            return

        try:
            data = np.load(self._calib_path, allow_pickle=False)
        except Exception as ex:
            logger.error("[StereoCalibrator] Failed to load %s: %s", self._calib_path, ex)
            return

        keys = set(data.files)
        logger.info("[StereoCalibrator] Loaded %s, keys: %s", self._calib_path.name, sorted(keys))

        if {"map1_l", "map2_l", "map1_r", "map2_r"}.issubset(keys):
            self.map1_l = data["map1_l"].astype(np.float32)
            self.map2_l = data["map2_l"].astype(np.float32)
            self.map1_r = data["map1_r"].astype(np.float32)
            self.map2_r = data["map2_r"].astype(np.float32)
            self.is_rectified = True
        elif {"K_l", "D_l", "K_r", "D_r", "R", "T"}.issubset(keys):
            K_l = data["K_l"].astype(np.float64)
            D_l = data["D_l"].astype(np.float64)
            K_r = data["K_r"].astype(np.float64)
            D_r = data["D_r"].astype(np.float64)
            R = data["R"].astype(np.float64)
            T = data["T"].astype(np.float64)
            image_size = (self._mono_w, self._mono_h)
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K_l, D_l, K_r, D_r, image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            )
            self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
                K_l, D_l, R1, P1, image_size, cv2.CV_32FC1,
            )
            self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
                K_r, D_r, R2, P2, image_size, cv2.CV_32FC1,
            )
            self.Q = Q
            self.is_rectified = True
        else:
            logger.error(
                "[StereoCalibrator] %s does not contain required keys "
                "(need map{1,2}_{l,r} or K_{l,r},D_{l,r},R,T)",
                self._calib_path.name,
            )
            return

        if "Q" in keys:
            self.Q = data["Q"].astype(np.float64)

        logger.info(
            "[StereoCalibrator] Calibrator ready: rectified=%s, mono=%dx%d",
            self.is_rectified, self._mono_w, self._mono_h,
        )

    def rectify(
        self,
        left_raw: np.ndarray,
        right_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对单眼左/右图做极线校正,返回 (left_rect, right_rect)。

        Args:
            left_raw: 左眼 BGR 图,``H x W x 3``,``(H, W) == (mono_h, mono_w)``。
            right_raw: 右眼 BGR 图,同上。

        Returns:
            (left_rect, right_rect): 校正后左右图,``shape == left_raw.shape``。

        Notes:
            - ``is_rectified=False`` 时,降级直接返回输入(不 remap,程序不崩)。
            - 不做缩放(规约硬约束:单目分辨率锁死 1920x1080)。
        """
        if not self.is_rectified:
            return left_raw, right_raw

        if left_raw.shape[:2] != (self._mono_h, self._mono_w):
            logger.warning(
                "[StereoCalibrator] left shape %s != expected (%d, %d), "
                "running raw remap (may distort epipolar alignment)",
                left_raw.shape[:2], self._mono_h, self._mono_w,
            )

        left_rect = cv2.remap(left_raw, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def verify_vertical_parity(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        samples: int = 50,
    ) -> float:
        """调试用:在多行 y 上比较左右图水平灰度差的均值,作为垂直对齐度量。

        校正良好时,左右图同 y 行的灰度差(沿 x 移动 d 像素)应接近 0。
        该函数对``samples`` 个 y 取平均的"最佳视差处"灰度差,越接近 0 越好。

        Args:
            left_rect, right_rect: 已校正图。
            samples: 采样行数。

        Returns:
            平均灰度差(浮点,越小越好)。
        """
        if left_rect.shape != right_rect.shape:
            return float("nan")
        gl = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gl.shape
        diffs: list[float] = []
        rng = np.random.default_rng(0)
        ys = rng.integers(low=0, high=h, size=samples)
        for y in ys:
            row_l = gl[y]
            row_r = gr[y]
            # 用简单 SSD 找最佳水平视差
            max_d = min(64, w // 4)
            best = float("inf")
            for d in range(0, max_d):
                if d == 0:
                    diff = float(np.mean((row_l - row_r) ** 2))
                else:
                    diff = float(np.mean((row_l[:-d] - row_r[d:]) ** 2))
                if diff < best:
                    best = diff
            diffs.append(best)
        return float(np.mean(diffs))


__all__ = ["StereoCalibrator"]
