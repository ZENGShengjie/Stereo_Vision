"""Thread-safe single-frame queues for the decoupled pipeline.

Architecture:
    Camera capture thread → LatestFrameQueue → SBSPipeline
                                                        │
                                                        ▼ pushes processed SBS
                                                  LatestSBSQueue
                                                    ▲         │
                            MJPEG ──────────────────┘         │
                            WebRTC recv() ─────────────────┘

Rationale:
  - Before refactor every MJPEG/WebRTC consumer called camera.read_rectified_pair()
    independently, causing N clients to run N copies of YOLO×2 + SGBM simultaneously.
  - Camera capture thread runs at camera FPS, SBSPipeline runs at processing FPS.
  - Both queues are length-1: producer overwrites, consumer reads latest.
  - If camera is faster, old frames are silently dropped (no accumulation).
  - If processing is slower, pull() returns None and pipeline skips that cycle.
"""
from __future__ import annotations

from threading import Lock
from typing import Optional

import numpy as np


class LatestFrameQueue:
    """Thread-safe length-1 queue for raw rectified frame pairs.

    Producer (camera capture thread) calls push(left, right) — always succeeds.
    Consumer (SBSPipeline) calls pull() — non-blocking, returns None if no new frame.
    """

    __slots__ = ("_frame", "_has_new", "_lock")

    def __init__(self) -> None:
        self._frame: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._has_new: bool = False
        self._lock = Lock()

    def push(self, left: np.ndarray, right: np.ndarray) -> None:
        with self._lock:
            self._frame = (left, right)
            self._has_new = True

    def pull(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if not self._has_new:
                return None
            self._has_new = False
            return self._frame

    @property
    def has_new(self) -> bool:
        with self._lock:
            return self._has_new


class LatestSBSQueue:
    """Thread-safe length-1 queue for processed SBS frames.

    Producer (SBSPipeline) calls push(sbs_frame) after processing.
    Consumer (MJPEG, WebRTC) calls pull() — non-blocking.
    """

    __slots__ = ("_frame", "_has_new", "_lock", "_timestamp")

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._has_new: bool = False
        self._lock = Lock()
        self._timestamp: float = 0.0

    def push(self, sbs_frame: np.ndarray) -> None:
        import time as _t
        with self._lock:
            self._frame = sbs_frame
            self._has_new = True
            self._timestamp = _t.time()

    def pull(self) -> Optional[tuple[np.ndarray, float]]:
        with self._lock:
            if not self._has_new:
                return None
            self._has_new = False
            return self._frame, self._timestamp

    def peek(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._frame, self._timestamp

    @property
    def has_new(self) -> bool:
        with self._lock:
            return self._has_new


__all__ = ["LatestFrameQueue", "LatestSBSQueue"]
