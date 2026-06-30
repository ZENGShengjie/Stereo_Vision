"""Wiring: lazy camera loading, pipeline assembly and graceful shutdown."""
from __future__ import annotations

from aiohttp import web

from camera import ZEDCamera, USBCamera, ZEDStatus, USBStatus
from processing.detector import CupDetector
from processing.sbs_pipeline import SBSPipeline
from processing.stereo_depth import StereoDepthSolver
from processing.warning import WarningOverlay
from state import CAMERA_KEY, PIPELINE_KEY, STEREO_AVAILABLE_KEY, WEBRTC_PEERS_KEY
from transport import ZEDTrack
from utils import timestamped_print

from config import (
    CAMERA_TYPE,
    ZED_RESOLUTION,
    ZED_FPS,
    USB_LEFT_INDEX,
    USB_RIGHT_INDEX,
    USB_TARGET_WIDTH,
    USB_TARGET_HEIGHT,
    USB_FPS,
    USB_STEREO_SCALE,
)


def _build_pipeline(camera):
    """组装 SBSPipeline: 摄像头 + 校正器 + 检测器 + 解算 + 预警。

    失败返回 ``None``(上层 fallback 到老的 ``cam.read_stereo()`` 路径)。
    """
    try:
        timestamped_print("Loading cup detector (YOLOv8s, classes=[41])...")
        detector = CupDetector()
    except Exception as ex:
        timestamped_print(f"ERROR: cup detector failed to load: {ex}")
        return None

    try:
        timestamped_print("Loading stereo depth solver (SGBM)...")
        solver = StereoDepthSolver()
    except Exception as ex:
        timestamped_print(f"ERROR: depth solver failed to init: {ex}")
        return None

    warn = WarningOverlay()
    calibrator = getattr(camera, "_calibrator", None)
    pipeline = SBSPipeline(
        camera=camera,
        calibrator=calibrator,
        detector=detector,
        solver=solver,
        warn=warn,
    )
    timestamped_print("SBSPipeline assembled")
    return pipeline


def _open_camera(app: web.Application) -> None:
    camera = None
    camera_type = CAMERA_TYPE.strip()
    status_str = "unknown"

    if camera_type == "usb":
        try:
            camera = USBCamera(
                left_index=USB_LEFT_INDEX,
                right_index=USB_RIGHT_INDEX,
                target_width=USB_TARGET_WIDTH,
                target_height=USB_TARGET_HEIGHT,
                fps=USB_FPS,
                stereo_scale=USB_STEREO_SCALE,
            )
            status_str = camera.status()
            app[STEREO_AVAILABLE_KEY] = True
            timestamped_print(f"USB camera opened: {status_str}")
        except Exception as ex:
            app[STEREO_AVAILABLE_KEY] = False
            timestamped_print(f"USB camera not available: {ex}")
    else:
        try:
            camera = ZEDCamera(resolution=ZED_RESOLUTION, fps=ZED_FPS)
            status_str = camera.status()
            app[STEREO_AVAILABLE_KEY] = True
            timestamped_print(f"ZED camera opened: {status_str}")
        except Exception as ex:
            timestamped_print(f"ZED not available (fallback): {ex}")
            try:
                camera = USBCamera(
                    left_index=USB_LEFT_INDEX,
                    right_index=USB_RIGHT_INDEX,
                    target_width=USB_TARGET_WIDTH,
                    target_height=USB_TARGET_HEIGHT,
                    fps=USB_FPS,
                    stereo_scale=USB_STEREO_SCALE,
                )
                status_str = camera.status()
                app[STEREO_AVAILABLE_KEY] = True
                timestamped_print(f"Fallback USB camera opened: {status_str}")
            except Exception as fallback_ex:
                timestamped_print(f"Fallback USB camera also failed: {fallback_ex}")
                app[STEREO_AVAILABLE_KEY] = False

    app[CAMERA_KEY] = camera

    # 阶段 0–4:在摄像头打开后,组装 SBSPipeline;失败时 PIPELINE_KEY=None,
    # 路由层会走老的 ``cam.read_stereo()`` 兜底路径(避免整个 aiohttp 启动失败)。
    if camera is not None and app[STEREO_AVAILABLE_KEY]:
        pipeline = _build_pipeline(camera)
        if pipeline is not None:
            app[PIPELINE_KEY] = pipeline
        else:
            timestamped_print(
                "SBSPipeline assembly failed; stream routes will use cam.read_stereo() fallback"
            )
            app[PIPELINE_KEY] = None
    else:
        app[PIPELINE_KEY] = None


async def _close_camera(app: web.Application) -> None:
    cam = app.get(CAMERA_KEY)
    if cam is not None:
        try:
            cam.close()
            timestamped_print("Camera closed")
        except Exception as ex:
            timestamped_print(f"Error closing camera: {ex}")
        finally:
            app[CAMERA_KEY] = None
    app[PIPELINE_KEY] = None


async def _close_webrtc(app: web.Application) -> None:
    peers = app.get(WEBRTC_PEERS_KEY, {})
    for pc in list(peers):
        track = peers.pop(pc, None)
        if track is not None:
            track.stop()
        if pc.connectionState != "closed":
            await pc.close()
    timestamped_print(f"Closed {len(peers)} WebRTC peers")


def register_lifecycle(app: web.Application) -> None:
    """Register on_startup and on_cleanup hooks."""

    async def on_startup(_app: web.Application) -> None:
        _open_camera(_app)

    async def on_cleanup(_app: web.Application) -> None:
        await _close_webrtc(_app)
        await _close_camera(_app)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
