"""Transport module.

ZEDTrack is lazily imported to avoid a hard aiortc dependency at import time.
"""
from __future__ import annotations

from .mjpeg_stream import mjpeg_generator

__all__ = ["ZEDTrack", "mjpeg_generator"]


def ZEDTrack(frame_fn_or_queue, fps=15, fallback_shape=(480, 640, 3)):
    """Lazily import and return a ZEDTrack instance.

    After the pipeline refactor, frame_fn_or_queue is a LatestSBSQueue instance.
    Each ZEDTrack pulls from the shared queue in recv() — no local capture
    thread, no redundant pipeline runs per client.

    Kept for backwards compatibility: if a plain callable frame_fn is passed
    (old calling convention), behaviour is unchanged.
    """
    from aiortc import VideoStreamTrack
    from av import VideoFrame
    import numpy as np
    import time

    # Detect whether we received a LatestSBSQueue or a plain frame_fn
    from wiring.frame_queue import LatestSBSQueue
    _is_queue = isinstance(frame_fn_or_queue, LatestSBSQueue)

    class _ZEDTrack(VideoStreamTrack):
        def __init__(self, _fps, _shape):
            super().__init__()
            self._fps = _fps
            self._shape = _shape
            self._period = 1.0 / _fps
            self._last_send = 0.0

        def _pull_frame(self):
            if _is_queue:
                result = frame_fn_or_queue.pull()
                if result is not None:
                    return result[0]   # (frame, timestamp)
                return None
            else:
                try:
                    return frame_fn_or_queue()
                except Exception:
                    return None

        async def recv(self):
            import cv2
            # Rate-limit: sleep until the next frame slot
            now = time.monotonic()
            wait = self._period - (now - self._last_send)
            if wait > 0:
                import asyncio
                await asyncio.sleep(wait)
            self._last_send = time.monotonic()

            arr = self._pull_frame()
            if arr is None:
                arr = np.zeros(self._shape, dtype=np.uint8)

            if arr.shape != self._shape:
                arr = cv2.resize(arr, (self._shape[1], self._shape[0]))

            frame = VideoFrame.from_ndarray(arr, format="bgr24")
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            return frame

        def stop(self):
            try:
                super().stop()
            except Exception:
                pass
            print("[ZEDTrack] stopped")

    return _ZEDTrack(fps, fallback_shape)
