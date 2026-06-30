"""ZED camera abstraction.

Lazy-loaded: must be opened after app startup, not at import time.
Provides stereo RGB, depth map, and confidence map.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .stereo_camera_base import StereoCameraBase

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    sl = None  # type: ignore


class ZEDStatus:
    NOT_AVAILABLE = "not_available"
    NOT_OPENED = "not_opened"
    OPEN = "open"
    ERROR = "error"


@dataclass
class ZEDFrame:
    """All data captured in one grab cycle."""
    stereo: np.ndarray           # SBS BGR (left+right), e.g. 2560x720
    depth: np.ndarray            # Depth map (float32, meters), same WxH
    confidence: np.ndarray       # Confidence map (uint8 0-255)
    timestamp_ns: int            # Nanoseconds since camera start


class ZEDCamera(StereoCameraBase):
    """Reads left/right stereo frames and depth from a ZED camera.

    All ZED-dependent code lives here. If pyzed is unavailable,
    the service still starts (other routes remain usable).
    """

    def __init__(
        self,
        resolution: str = "HD720",
        fps: int = 30,
        target_width: int = 2560,
        target_height: int = 720,
        depth_quality: str = "MEDIUM",
        depth_range_min: int = 500,
        depth_range_max: int = 20000,
    ) -> None:
        if not ZED_AVAILABLE:
            raise RuntimeError("pyzed.sl is not installed or ZED SDK is missing.")

        self._cam = sl.Camera()
        init_params = sl.InitParameters()

        res_map = {
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD720)
        init_params.camera_fps = fps
        init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_quality.upper(), sl.DEPTH_MODE.MEDIUM)
        init_params.depth_minimum_distance = depth_range_min  # mm
        init_params.depth_maximum_distance = depth_range_max  # mm
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        status = self._cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        self._mat_left = sl.Mat()
        self._mat_right = sl.Mat()
        self._mat_depth = sl.Mat()
        self._mat_conf = sl.Mat()
        self._runtime_params = sl.RuntimeParameters(
            sensing_mode=sl.SENSING_MODE.STANDARD,
            depth_mode=init_params.depth_mode,
            remove_saturated_areas=True,
        )

        self._target_width = target_width
        self._target_height = target_height
        self._status = ZEDStatus.OPEN

    def status(self) -> str:
        return self._status

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w != self._target_width or h != self._target_height:
            return cv2.resize(
                img,
                (self._target_width, self._target_height),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def grab(self) -> bool:
        """Trigger a new capture. Must be called before read_*."""
        return self._cam.grab(self._runtime_params) == sl.ERROR_CODE.SUCCESS

    def read_stereo(self) -> np.ndarray | None:
        """Return stereo BGR image (left+right concatenated) or None on error."""
        if not self.grab():
            return None

        self._cam.retrieve_image(self._mat_left, sl.VIEW.LEFT)
        self._cam.retrieve_image(self._mat_right, sl.VIEW.RIGHT)

        left_bgra = self._mat_left.get_data()
        right_bgra = self._mat_right.get_data()

        left = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2BGR)

        stereo = cv2.hconcat([left, right])
        return self._resize_if_needed(stereo)

    def read_depth(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (depth_m, confidence) in millimetres, or None on error.

        Returns:
            - depth: float32 array in metres (same HxW as stereo)
            - confidence: uint8 array 0-255
        """
        if not self.grab():
            return None

        self._cam.retrieve_measure(self._mat_depth, sl.MEASURE.DEPTH)
        self._cam.retrieve_measure(self._mat_conf, sl.MEASURE.CONFIDENCE)

        depth_data = self._mat_depth.get_data()
        conf_data = self._mat_conf.get_data()

        depth_mm = np.nan_to_num(depth_data, nan=0.0).astype(np.float32)
        depth_m = depth_mm / 1000.0  # mm → m

        conf_u8 = np.clip(conf_data, 0, 255).astype(np.uint8)

        depth_m = self._resize_if_needed(depth_m)
        conf_u8 = self._resize_if_needed(conf_u8)

        return depth_m, conf_u8

    def read_frame(self) -> ZEDFrame | None:
        """Return a full ZEDFrame (stereo + depth + confidence + timestamp)."""
        if not self.grab():
            return None

        self._cam.retrieve_image(self._mat_left, sl.VIEW.LEFT)
        self._cam.retrieve_image(self._mat_right, sl.VIEW.RIGHT)
        self._cam.retrieve_measure(self._mat_depth, sl.MEASURE.DEPTH)
        self._cam.retrieve_measure(self._mat_conf, sl.MEASURE.CONFIDENCE)

        left_bgra = self._mat_left.get_data()
        right_bgra = self._mat_right.get_data()
        left = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2BGR)
        stereo = cv2.hconcat([left, right])
        stereo = self._resize_if_needed(stereo)

        depth_data = self._mat_depth.get_data()
        conf_data = self._mat_conf.get_data()
        depth_m = np.nan_to_num(depth_data, nan=0.0).astype(np.float32) / 1000.0
        conf_u8 = np.clip(conf_data, 0, 255).astype(np.uint8)
        depth_m = self._resize_if_needed(depth_m)
        conf_u8 = self._resize_if_needed(conf_u8)

        ts_ns = int(self._mat_left.timestamp.get_milliseconds() * 1_000_000) if self._mat_left.timestamp else 0

        return ZEDFrame(
            stereo=stereo,
            depth=depth_m,
            confidence=conf_u8,
            timestamp_ns=ts_ns,
        )

    def close(self) -> None:
        self._cam.close()
        self._status = ZEDStatus.NOT_OPENED
