"""Transport module.

ZEDTrack is lazily imported to avoid a hard aiortc dependency at import time.
"""

from .mjpeg_stream import mjpeg_generator

__all__ = ["ZEDTrack", "mjpeg_generator"]


def ZEDTrack(frame_fn, fps=15, fallback_shape=(480, 640, 3)):
    """Lazily import and return a ZEDTrack instance."""
    from aiortc import VideoStreamTrack
    from av import VideoFrame
    import numpy as np
    import threading
    import time

    class _ZEDTrack(VideoStreamTrack):
        def __init__(self, _frame_fn, _fps, _shape):
            super().__init__()
            self._frame_fn = _frame_fn
            self._fps = _fps
            self._period = 1.0 / _fps
            self._latest = None
            self._last = None
            self._running = True

            t = threading.Thread(target=self._capture_loop, daemon=True)
            t.start()

        def _capture_loop(self):
            while self._running:
                try:
                    arr = self._frame_fn()
                    if arr is not None:
                        self._latest = arr
                        self._last = arr
                except Exception:
                    pass
                time.sleep(self._period)

        async def recv(self):
            import cv2
            arr = self._latest
            if arr is None:
                arr = self._last
            if arr is None:
                arr = np.zeros(_shape, dtype=np.uint8)

            if arr.shape != _shape:
                arr = cv2.resize(arr, (_shape[1], _shape[0]))

            frame = VideoFrame.from_ndarray(arr, format="bgr24")
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            return frame

        def stop(self):
            self._running = False
            try:
                super().stop()
            except Exception:
                pass
            print("[ZEDTrack] stopped")

    return _ZEDTrack(frame_fn, fps, fallback_shape)
