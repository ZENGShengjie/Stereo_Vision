"""MJPEG stream generator for HTTP streaming."""

from __future__ import annotations

import asyncio
import time

import cv2


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

    while True:
        now = loop.time()
        sleep_time = next_time - now
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        next_time += period

        frame = await loop.run_in_executor(None, frame_fn)
        if frame is None:
            continue
        ret, jpeg = await loop.run_in_executor(
            None,
            lambda f=frame: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75]),
        )
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
