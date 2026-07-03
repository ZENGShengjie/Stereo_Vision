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

import json
import os
import threading
import time
from aiohttp import web

# #region agent log — DEBUG mode instrumentation
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug-c7ffa8.log")
def _dbg(hyp, msg, data):
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "c7ffa8",
                "location": "wiring/lifecycle.py",
                "hypothesisId": hyp,
                "message": msg,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion

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
from config.hardware import MONO_HEIGHT


# Live references for the capture thread so on_cleanup can stop it
_capture_threads: list[threading.Thread] = []
_capture_running: list[bool] = []   # shared booleans, written by main, read by threads


def _camera_capture_loop(camera, queue: LatestFrameQueue, running_flag: list[bool]):
    """Daemon thread: reads camera and pushes latest rectified pair into queue.

    Ignores slow frames — if processing is slower than camera FPS the deque(1)
    silently drops them.  This prevents frame accumulation when the browser
    tab is paused (queue stays at length 1, not 1000).
    """
    # #region agent log — DEBUG mode instrumentation
    _push_count = [0]
    _consecutive_read_failures = [0]
    _read_ms_sum = [0.0]
    _read_ms_max = [0.0]
    # #endregion
    timestamped_print("[capture] Camera capture thread started")
    while running_flag[0]:
        try:
            t_read_start = time.perf_counter()
            rect = camera.read_rectified_pair()
            t_read_end = time.perf_counter()
            read_ms = (t_read_end - t_read_start) * 1000.0
            _read_ms_sum[0] += read_ms
            if read_ms > _read_ms_max[0]:
                _read_ms_max[0] = read_ms
            if rect is not None:
                queue.push(*rect)
                # #region agent log — DEBUG mode instrumentation
                _push_count[0] += 1
                _consecutive_read_failures[0] = 0
                if _push_count[0] % 10 == 0:
                    avg_ms = _read_ms_sum[0] / max(_push_count[0], 1)
                    fps_eff = 1000.0 / max(avg_ms, 1e-3)
                    _dbg("CAP1", "frame_queue.push", {
                        "count": _push_count[0],
                        "read_avg_ms": round(avg_ms, 1),
                        "read_max_ms": round(_read_ms_max[0], 1),
                        "fps_effective": round(fps_eff, 2),
                    })
                # #endregion
            else:
                # #region agent log — DEBUG mode instrumentation
                _consecutive_read_failures[0] += 1
                if _consecutive_read_failures[0] == 30 or _consecutive_read_failures[0] % 100 == 0:
                    _dbg("CAP1", "capture_read_failed", {
                        "consecutive_failures": _consecutive_read_failures[0],
                        "pushed_so_far": _push_count[0],
                        "read_max_ms": round(_read_ms_max[0], 1),
                    })
                # #endregion
        except Exception as ex:
            timestamped_print(f"[capture] read_rectified_pair error: {ex}")
        # Adaptive backoff: when frames are flowing, sleep briefly;
        # when reads keep failing, sleep longer so we don't hammer the camera.
        sleep_ms = 5 if _consecutive_read_failures[0] == 0 else min(100, _consecutive_read_failures[0] * 2)
        time.sleep(sleep_ms / 1000.0)
    timestamped_print("[capture] Camera capture thread stopped")


def _pipeline_processing_loop(pipeline, running_flag: list[bool]):
    """Daemon thread: pulls from frame_queue, runs YOLO+SGBM+overlay, pushes to sbs_queue.

    Separated from capture loop so the pipeline can run at its own pace
    (a slow SGBM won't starve the camera capture). When pipeline is slower
    than camera, the latest-frame queue silently drops stale frames.
    """
    # #region agent log — DEBUG mode instrumentation
    _proc_count = [0]
    _idle_iters = [0]
    _prev_frames_served = [0]
    _dbg("PL0", "pipeline_loop_start", {"pipeline_obj_id": id(pipeline)})
    # #endregion
    timestamped_print("[pipeline] Processing thread started")
    # Adaptive pacing:
    # - when a frame is available, target ~30 Hz (33ms) so we don't over-process
    # - when idle (pull() returned None), back off to avoid CPU saturation
    active_period = 0.033   # 33 ms — cap processing at ~30 fps
    idle_period = 0.050     # 50 ms when waiting for the next frame
    while running_flag[0]:
        t0 = time.perf_counter()
        try:
            pipeline.process_one_frame()
        except Exception as ex:  # noqa: BLE001
            timestamped_print(f"[pipeline] process_one_frame error: {ex}")
        # #region agent log — DEBUG mode instrumentation
        _proc_count[0] += 1
        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
        stages = getattr(pipeline, "_last_stage_times", {}) or {}
        frames_served_now = getattr(pipeline, "_frames", None) or 0
        if frames_served_now > _prev_frames_served[0]:
            _idle_iters[0] = 0
            _prev_frames_served[0] = frames_served_now
        else:
            _idle_iters[0] += 1
        # Throttled logging: first 10 frames always, then every 100 iterations,
        # plus a 1-Hz summary when idle. Avoids flooding the log when pull() is
        # repeatedly returning None (the previous 1-ms spin flooded debug-c7ffa8.log
        # with tens of thousands of entries and starved the camera thread).
        should_log = (
            _proc_count[0] <= 10
            or _proc_count[0] % 100 == 0
            or _idle_iters[0] == 100
            or (_idle_iters[0] > 100 and _idle_iters[0] % 500 == 0)
        )
        if should_log:
            _dbg("PR1", "pipeline_frame_timing", {
                "count": _proc_count[0],
                "idle_iters": _idle_iters[0],
                "total_ms": round(elapsed_total_ms, 1),
                "stage_ms": {k: round(v * 1000.0, 1) for k, v in stages.items()},
                "frames_served": frames_served_now,
            })
        # #endregion
        # Adaptive sleep: longer when idle so we don't spin, shorter when active.
        elapsed = time.perf_counter() - t0
        if _idle_iters[0] == 0:
            sleep_for = active_period - elapsed
        else:
            sleep_for = idle_period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
    timestamped_print("[pipeline] Processing thread stopped")


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
        # #region agent log — DEBUG mode instrumentation / 启动横幅 + 自动检测
        import os as _os
        _h_low = USB_TARGET_HEIGHT <= 600 or MONO_HEIGHT <= 600
        _dbg("CAM_LAUNCH", "usb_resolution", {
            "STEREO_USB_HEIGHT": _os.getenv("STEREO_USB_HEIGHT"),
            "STEREO_MONO_HEIGHT": _os.getenv("STEREO_MONO_HEIGHT"),
            "USB_TARGET_HEIGHT": USB_TARGET_HEIGHT,
            "MONO_HEIGHT": MONO_HEIGHT,
            "low_height_mode": _h_low,
            "hint": (
                "Set STEREO_USB_HEIGHT=540 to switch camera to fast-read mode "
                "(~5 fps instead of ~1 fps). PowerShell: "
                "$env:STEREO_USB_HEIGHT = 540"
            ) if not _h_low else "low-height mode active",
        })
        # #endregion
        requested_target_h = USB_TARGET_HEIGHT
        try:
            camera = USBCamera(
                left_index=USB_LEFT_INDEX,
                right_index=USB_RIGHT_INDEX,
                target_width=USB_TARGET_WIDTH,
                target_height=USB_TARGET_HEIGHT,
                fps=USB_FPS,
                stereo_scale=USB_STEREO_SCALE,
            )
            # #region agent log — DEBUG mode instrumentation / 自动性能降级
            # 某些廉价 USB 摄像头在 1080 模式下 read() 耗时 ~1000ms 导致 fps=1。
            # 自动检测:启动时测一次 read 耗时,若 >300ms 自动降到 540 模式。
            try:
                import time as _t_low
                _probe_n = 2
                _probe_total = 0.0
                _probe_ok = 0
                for _i in range(_probe_n):
                    _ts = _t_low.perf_counter()
                    _probe = camera.read_rectified_pair()
                    if _probe is not None:
                        _probe_total += (_t_low.perf_counter() - _ts) * 1000.0
                        _probe_ok += 1
                if _probe_ok > 0:
                    _avg_ms = _probe_total / _probe_ok
                    _auto_action = "kept"
                    if (
                        _avg_ms > 300.0
                        and requested_target_h > 600
                        and os.getenv("STEREO_USB_HEIGHT") is None
                    ):
                        # Auto-downgrade to 540 for the current session
                        from camera.usb_camera import USBCamera as _UC
                        try:
                            camera.close()
                        except Exception:
                            pass
                        try:
                            camera = _UC(
                                left_index=USB_LEFT_INDEX,
                                right_index=USB_RIGHT_INDEX,
                                target_width=USB_TARGET_WIDTH,
                                target_height=540,
                                fps=USB_FPS,
                                stereo_scale=USB_STEREO_SCALE,
                            )
                            _auto_action = "auto-downgraded"
                            timestamped_print(
                                f"[capture] AUTO DOWNGRADE: read() took {round(_avg_ms,1)}ms "
                                f"at {requested_target_h}p — re-opened at 540p for ~5 fps. "
                                f"Set STEREO_USB_HEIGHT=1080 to keep full resolution."
                            )
                        except Exception as _auto_ex:
                            # Fall back to keeping the slow camera rather than failing the
                            # whole service.
                            _auto_action = "auto-downgrade-failed"
                            timestamped_print(
                                f"[capture] AUTO DOWNGRADE failed: {_auto_ex}; keeping original camera."
                            )
                    _dbg("CAM_PROBE", "auto_perf_downgrade", {
                        "probe_avg_ms": round(_avg_ms, 1),
                        "requested_target_h": requested_target_h,
                        "action": _auto_action,
                        "rationale": "read() too slow (>300ms), auto-downgraded to 540 height"
                            if _auto_action == "auto-downgraded" else
                            "OK or already in low-height mode",
                    })
            except Exception as _ex_probe:
                _dbg("CAM_PROBE", "auto_perf_downgrade_failed", {"err": str(_ex_probe)})
            # #endregion
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
            # Start pipeline processing thread (reads frame_queue, pushes sbs_queue)
            pipe_thread = threading.Thread(
                target=_pipeline_processing_loop,
                args=(pipeline, running_flag),
                daemon=True,
                name="pipeline-processing",
            )
            _capture_threads.append(pipe_thread)
            pipe_thread.start()
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
