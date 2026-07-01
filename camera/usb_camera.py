"""USB 双目摄像头抽象层。

通过 OpenCV 读取 USB 摄像头(支持两种模式):
1. 双设备模式:左右各用一个独立摄像头索引
2. 单设备模式:同一个索引输出左右拼接帧(需自行分割)

立体校正(阶段 0):
- 启动时加载 ``StereoCalibrator``,从外部产出的 npz 标定文件读 remap 映射
  表,对每帧左/右图做极线校正。无 npz 时降级为直通(警告 + 精度变差)。
- 新公开接口 :meth:`USBCamera.read_rectified_pair` 直接返回校正后的左右
  单眼图,供阶段 2 SGBM 算深度用(避免重复 remap)。

深度估计: ``_compute_depth`` 接收"已校正"的左右图,跑 SGBM。
"""
from __future__ import annotations

import cv2
import numpy as np
import threading
import time

from .calibration import StereoCalibrator
from .stereo_camera_base import StereoCameraBase


class USBStatus:
    NOT_AVAILABLE = "not_available"
    NOT_OPENED = "not_opened"
    OPEN = "open"
    ERROR = "error"


_BACKENDS = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF, "MSMF"),
]


def _try_open(
    index: int,
    fps: int = 15,
    target_width: int | None = None,
    target_height: int | None = None,
    timeout_s: float = 10.0,
    preferred_backend: int | None = None,
) -> tuple[cv2.VideoCapture | None, str]:
    """尝试用多个后端打开摄像头，每个后端有超时限制。

    双设备模式下应传入 preferred_backend，强制两个摄像头走同一个后端，
    避免 DSHOW 与 MSMF 对同一个物理设备给出的 index 顺序不一致。

    如果传入 target_width / target_height，会先 set 再 read，
    避免廉价摄像头在默认协商分辨率下能开、set 高分辨率却静默失败的问题。
    """
    import time
    order = list(_BACKENDS)
    if preferred_backend is not None:
        # 把首选后端排到最前
        order.sort(key=lambda b: 0 if b[0] == preferred_backend else 1)

    for backend, name in order:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # 先在目标分辨率下 set，再 sleep+read 验证
        # 这样能尽早发现"set 失败但默认分辨率能开"的伪装设备
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if target_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        if target_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        time.sleep(0.5)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[USBCamera] Index {index} opened with {name}, frame shape: {frame.shape}")
            return cap, name

        cap.release()

    return None, "none"


class USBCamera(StereoCameraBase):
    """读取 USB 双目摄像头（支持单设备双目或双设备独立摄像头），拼接为 SBS 并估计深度。

    依赖 OpenCV（opencv-python），无需 ZED SDK。
    """

    def __init__(
        self,
        left_index: int = 1,
        right_index: int = 1,
        target_width: int = 1920,
        target_height: int = 1080,
        fps: int = 10,
        stereo_scale: float = 0.5,
    ) -> None:
        """USB 双目摄像头初始化。

        分辨率语义：
        - 单设备 SBS 模式（left == right）：摄像头整帧 = 2 * target_width x target_height
          例：target=1920x1080 → 摄像头输出 3840x1080（左 1920x1080 + 右 1920x1080）
        - 双设备独立模式（left != right）：左右各开 target_width x target_height
          例：target=1920x1080 → 每只眼 1920x1080
        """
        self._left_index = left_index
        self._right_index = right_index
        self._target_width = target_width
        self._target_height = target_height
        self._fps = fps
        self._stereo_scale = stereo_scale

        # 单设备模式：左右索引相同，同一个摄像头输出左右拼接帧
        self._single_device = (left_index == right_index)

        # Note: StereoSGBM, depth computation, focal, baseline are no longer
        # defined here.  The SBSPipeline uses processing.stereo_depth.StereoDepthSolver
        # (a separate, better-configured SGBM instance) instead.
        # The old _compute_depth() method has been removed — it was dead code
        # (never called by any consumer in the refactored wiring/lifecycle).
        self._status = USBStatus.OPEN

        if self._single_device:
            self._cap = self._open_single()
            if self._cap is None:
                self._status = USBStatus.NOT_AVAILABLE
                raise RuntimeError(f"Stereo camera at index {left_index} failed to open or read frames")
        else:
            # 双设备模式：先开左摄像头锁定 backend，避免 DSHOW/MSMF 索引顺序不同导致
            # 第二个摄像头被错误地关联到另一台物理设备。
            self._cap_left, left_ok = self._open_and_verify(left_index)
            preferred_backend: int | None = None
            if self._cap_left is not None:
                # 从已打开的 cap 反推后端，让右摄像头走相同后端
                preferred_backend = self._detect_backend(self._cap_left)

            import time as _t
            _t.sleep(0.2)

            self._cap_right, right_ok = self._open_and_verify(right_index, preferred_backend=preferred_backend)

            if not left_ok and not right_ok:
                self._status = USBStatus.NOT_AVAILABLE
                if self._cap_left:
                    self._cap_left.release()
                if self._cap_right:
                    self._cap_right.release()
                raise RuntimeError(f"Both cameras at index {left_index}/{right_index} failed to read frames")
            if not left_ok:
                self._status = USBStatus.ERROR
                if self._cap_left:
                    self._cap_left.release()
                    self._cap_left = None
                raise RuntimeError(f"Left camera (index {left_index}) failed to read frames")
            if not right_ok:
                self._status = USBStatus.ERROR
                if self._cap_right:
                    self._cap_right.release()
                    self._cap_right = None
                raise RuntimeError(f"Right camera (index {right_index}) failed to read frames")

            self._cap = None

        # 阶段 0:加载立体校正器。无 npz 时降级为直通(警告已由 calibrator 自己打印)。
        self._calibrator: StereoCalibrator = StereoCalibrator(
            mono_width=self._target_width,
            mono_height=self._target_height,
        )

        # USB 摄像头切换分辨率后，MJPG 解码缓冲有 1-3 帧的旧/黑画面残留。
        # 启动时 warmup 几帧丢弃，避免 WebRTC 前端第一秒看到黑屏。
        self._warmup_frames()

    @staticmethod
    def _detect_backend(cap) -> int:
        """从一个已打开的 VideoCapture 反推它实际使用的 backend。"""
        # OpenCV 4.x 没有直接 API；用 backend name 字符串判断
        try:
            name = cap.getBackendName()
        except Exception:
            name = ""
        if name == "DSHOW":
            return cv2.CAP_DSHOW
        if name == "MSMF":
            return cv2.CAP_MSMF
        # 兜底：DSHOW
        return cv2.CAP_DSHOW

    def _open_single(self) -> cv2.VideoCapture | None:
        """单设备模式：尝试打开一个摄像头并验证能读帧。

        单设备模式下摄像头直接输出 SBS 整帧：
        - 请求宽度 = 2 * target_width（SBS 拼接后 = 左眼 + 右眼）
        - 请求高度 = target_height
        例：target=1920x1080 → 摄像头输出 3840x1080（左 1920 + 右 1920）
        """
        cap, name = _try_open(
            self._left_index,
            self._fps,
            target_width=self._target_width * 2,
            target_height=self._target_height,
            timeout_s=5.0,
        )
        if cap is None:
            print(f"[USBCamera] No backend could open index {self._left_index}")
            return None
        return cap

    def _open_and_verify(
        self,
        index: int,
        preferred_backend: int | None = None,
    ) -> tuple[cv2.VideoCapture | None, bool]:
        """双设备模式：打开摄像头并验证能读帧。

        双设备模式下应传入 preferred_backend，强制左右摄像头走同一个后端，
        避免 DSHOW 与 MSMF 对同一物理设备给出的 index 顺序不一致。
        """
        cap, name = _try_open(
            index,
            self._fps,
            target_width=self._target_width,
            target_height=self._target_height,
            timeout_s=5.0,
            preferred_backend=preferred_backend,
        )
        if cap is None:
            return None, False
        return cap, True

    def status(self) -> str:
        return self._status

    def _warmup_frames(self, n: int = 5, settle_ms: int = 200) -> None:
        """MJPG 切换分辨率后丢弃前 n 帧（通常会有 1-3 帧黑屏/旧分辨率残留）。"""
        deadline = time.time() + (n * settle_ms / 1000.0) + 1.0
        count = 0
        while count < n and time.time() < deadline:
            try:
                _ = self._read_raw()
            except Exception:
                pass
            count += 1
            time.sleep(settle_ms / 1000.0)

    def _read_raw(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._single_device:
            if self._cap is None:
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            h, w = frame.shape[:2]
            eye_w = w // 2
            left = frame[:, :eye_w]
            right = frame[:, eye_w:]
            return left, right
        else:
            if self._cap_left is None or self._cap_right is None:
                return None
            ret_l, frame_l = self._cap_left.read()
            ret_r, frame_r = self._cap_right.read()
            if not ret_l or not ret_r:
                return None
            return frame_l, frame_r

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        # Single-device SBS mode: 摄像头已经直接输出 (2*target_width, target_height) 的 SBS 整帧
        # 不需要再 resize（resize 会降低画质；如需缩放请用 _stereo_scale 在生成时处理）
        if self._single_device:
            return img
        # Dual-device mode: hconcat output is (2 * target_width) x target_height.
        # We keep that SBS size; downstream display will split it for the two eyes.
        if w != self._target_width * 2 or h != self._target_height:
            return cv2.resize(
                img,
                (self._target_width * 2, self._target_height),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def _apply_rectify(
        self, img_l: np.ndarray, img_r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """用 ``StereoCalibrator`` 对左右单眼图做极线校正。

        注意:深度计算已移至 processing.stereo_depth.StereoDepthSolver,
        这里只负责校正(remap),不负责 SGBM 或深度公式。
        """
        return self._calibrator.rectify(img_l, img_r)

    def read_rectified_pair(self) -> tuple[np.ndarray, np.ndarray] | None:
        """读一帧并返回**已校正**的左右单眼图 (left_rect, right_rect)。

        这是给阶段 2 SGBM 算深度 + 阶段 4 管线拼 SBS 用的"原子接口":
        - 读一次 raw 帧
        - 做极线校正
        - 返回 (mono_h, mono_w, 3) × 2

        无帧可读(摄像头断开)时返回 ``None``。
        """
        raw = self._read_raw()
        if raw is None:
            return None
        return self._apply_rectify(*raw)

    def read_stereo(self) -> np.ndarray | None:
        """Return SBS BGR image (left+right concatenated) or None on error.

        走 :meth:`_apply_rectify` 校正后再 ``hconcat``,输出形状 = ``(target_h, 2*target_w, 3)``
        = ``(1080, 3840, 3)``,与本任务硬约束(无 resize)一致。
        """
        rect = self.read_rectified_pair()
        if rect is None:
            return None
        img_l_rect, img_r_rect = rect
        return cv2.hconcat([img_l_rect, img_r_rect])

    def close(self) -> None:
        if self._single_device:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        else:
            if self._cap_left is not None:
                self._cap_left.release()
                self._cap_left = None
            if self._cap_right is not None:
                self._cap_right.release()
                self._cap_right = None
        self._status = USBStatus.NOT_OPENED
