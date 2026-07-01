"""Wiring: camera capture thread, pipeline assembly and graceful shutdown.

Architecture (after refactor):
    Camera capture thread
          │
          │ writes rectified (left, right) to LatestFrameQueue
          ▼
    LatestFrameQueue  (length=1, thread-safe)
          │
          │ pull() — non-blocking, returns latest or None
          ▼
    SBSPipeline.process_one_frame()
          │
          │ writes processed SBS frame
          ▼
    MJPEG / WebRTC  (multiple consumers share same processed frame)

Before refactor every MJPEG/WebRTC consumer called camera.read_rectified_pair()
independently, causing:
  (a) N WebRTC clients ran N copies of YOLO×2 + SGBM simultaneously
  (b) no frame-drop policy when camera rate > processing rate
"""
from __future__ import annotations

import threading
import time
from aiohttp import web

from camera import ZEDCamera, USBCamera, ZEDStatus, USBStatus
from processing.detector import CupDetector
from processing.sbs_pipeline import SBSPipeline
from processing.stereo_depth import StereoDepthSolver
from processing.warning import WarningOverlay
from state import (
    CAMERA_KEY,
    FRAME_QUEUE_KEY,
    PIPELINE_KEY,
    SBS_QUEUE_KEY,
    STEREO_AVAILABLE_KEY,
    WEBRTC_PEERS_KEY,
)
from transport import ZEDTrack
from utils import timestamped_print
from wiring.frame_queue import LatestFrameQueue, LatestSBSQueue

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


# Live references for the capture thread so on_cleanup can stop it
_capture_threads: list[threading.Thread] = []
_capture_running: list[bool] = []   # shared booleans, written by main, read by threads


def _camera_capture_loop(camera, queue: LatestFrameQueue, running_flag: list[bool]):
    """Daemon thread: reads camera and pushes latest rectified pair into queue.

    Ignores slow frames — if processing is slower than camera FPS the deque(1)
    silently drops them.  This prevents frame accumulation when the browser
    tab is paused (queue stays at length 1, not 1000).
    """
    timestamped_print("[capture] Camera capture thread started")
    while running_flag[0]:
        try:
            rect = camera.read_rectified_pair()
            if rect is not None:
                queue.push(*rect)
        except Exception as ex:
            timestamped_print(f"[capture] read_rectified_pair error: {ex}")
        # Small sleep prevents tight-spin on camera disconnect
        time.sleep(0.001)
    timestamped_print("[capture] Camera capture thread stopped")


def _build_pipeline(frame_queue: LatestFrameQueue, sbs_queue: LatestSBSQueue):
    """Assemble SBSPipeline: reads from frame_queue, not directly from camera.

    Failure returns ``None`` (upstream falls back to ``cam.read_stereo()``).
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
    pipeline = SBSPipeline(
        frame_queue=frame_queue,
        sbs_queue=sbs_queue,
        detector=detector,
        solver=solver,
        warn=warn,
    )
    timestamped_print("SBSPipeline assembled (frame-queue mode)")
    return pipeline


def _open_camera(app: web.Application) -> None:
    global _capture_threads, _capture_running

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

    if camera is not None and app[STEREO_AVAILABLE_KEY]:
        # Create shared queues
        frame_queue = LatestFrameQueue()
        sbs_queue = LatestSBSQueue()
        app[FRAME_QUEUE_KEY] = frame_queue
        app[SBS_QUEUE_KEY] = sbs_queue

        # Start camera capture thread
        running_flag = [True]
        _capture_running.append(running_flag)
        cap_thread = threading.Thread(
            target=_camera_capture_loop,
            args=(camera, frame_queue, running_flag),
            daemon=True,
            name="camera-capture",
        )
        _capture_threads.append(cap_thread)
        cap_thread.start()

        # Build pipeline (reads from queue, not camera directly)
        pipeline = _build_pipeline(frame_queue, sbs_queue)
        if pipeline is not None:
            app[PIPELINE_KEY] = pipeline
        else:
            timestamped_print(
                "SBSPipeline assembly failed; stream routes will use cam.read_stereo() fallback"
            )
            app[PIPELINE_KEY] = None
    else:
        app[FRAME_QUEUE_KEY] = None
        app[PIPELINE_KEY] = None
    app[SBS_QUEUE_KEY] = None


async def _close_camera(app: web.Application) -> None:
    global _capture_threads, _capture_running

    # Signal capture threads to stop
    for flag in _capture_running:
        flag[0] = False

    # Wait for threads to finish (daemon=True ensures they die with the process)
    for t in _capture_threads:
        t.join(timeout=2.0)

    _capture_threads.clear()
    _capture_running.clear()

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
    app[FRAME_QUEUE_KEY] = None


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
