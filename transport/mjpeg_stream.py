"""MJPEG stream generator for HTTP streaming."""

from __future__ import annotations

import asyncio
import json
import os
import time

import cv2

# #region agent log — DEBUG mode instrumentation
_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "debug-c7ffa8.log",
)


def _dbg(hyp, msg, data):
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {
                    "sessionId": "c7ffa8",
                    "location": "transport/mjpeg_stream.py",
                    "hypothesisId": hyp,
                    "message": msg,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                }
            ) + "\n")
    except Exception:
        pass
# #endregion


def mjpeg_generator(frame_fn, fps: int = 15):
    """Yields multipart MJPEG chunks for HTTP streaming (sync version).

    Args:
        frame_fn: callable returning the next processed frame (np.ndarray or None)
        fps: target frames per second
    """
    period = 1.0 / fps
    next_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        delay = next_time - now
        if delay > 0:
            time.sleep(delay)
        next_time = max(next_time + period, time.perf_counter())

        frame = frame_fn()
        if frame is None:
            continue
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


async def mjpeg_generator_async(frame_fn, fps: int = 15):
    """Yields multipart MJPEG chunks for HTTP streaming (async version).

    All blocking work (frame capture + encode) runs in a thread pool
    so it never blocks the aiohttp event loop.

    Args:
        frame_fn: callable returning the next processed frame (np.ndarray or None)
        fps: target frames per second
    """
    loop = asyncio.get_running_loop()
    period = 1.0 / fps
    next_time = loop.time()

    # ── Per-connection instrumentation ──────────────────────────────────
    enc_total_ms = 0.0
    enc_count = 0
    enc_max_ms = 0.0
    pull_total_ms = 0.0
    pull_count = 0
    pull_max_ms = 0.0
    pull_none = 0
    log_counter = 0

    while True:
        now = loop.time()
        sleep_time = next_time - now
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        next_time += period

        # ── pull frame + time it ───────────────────────────────────
        t_pull_start = time.perf_counter()
        frame = await loop.run_in_executor(None, frame_fn)
        t_pull_end = time.perf_counter()
        pull_ms = (t_pull_end - t_pull_start) * 1000
        pull_total_ms += pull_ms
        pull_count += 1
        if pull_ms > pull_max_ms:
            pull_max_ms = pull_ms

        if frame is None:
            pull_none += 1
            continue

        # ── encode frame + time it (was UNTIMED before) ──────────
        t_enc_start = time.perf_counter()
        ret, jpeg = await loop.run_in_executor(
            None,
            lambda f=frame: cv2.imencode(
                ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75]
            ),
        )
        t_enc_end = time.perf_counter()
        enc_ms = (t_enc_end - t_enc_start) * 1000
        enc_total_ms += enc_ms
        enc_count += 1
        if enc_ms > enc_max_ms:
            enc_max_ms = enc_ms

        if not ret:
            continue

        # ── periodic log every 30 sends ─────────────────────────────
        log_counter += 1
        if log_counter >= 30:
            log_counter = 0
            _dbg(
                "MJ_ENC",
                "mjpeg_encode_stats",
                {
                    "frame_shape": list(frame.shape),
                    "enc_avg_ms": round(enc_total_ms / max(enc_count, 1), 1),
                    "enc_max_ms": round(enc_max_ms, 1),
                    "pull_avg_ms": round(
                        pull_total_ms / max(pull_count, 1), 1
                    ),
                    "pull_max_ms": round(pull_max_ms, 1),
                    "pull_none_rate": pull_none / max(pull_count, 1),
                    "kb_per_frame": round(len(jpeg) / 1024, 1),
                },
            )
            enc_total_ms = 0.0
            enc_count = 0
            enc_max_ms = 0.0
            pull_total_ms = 0.0
            pull_count = 0
            pull_max_ms = 0.0
            pull_none = 0

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
