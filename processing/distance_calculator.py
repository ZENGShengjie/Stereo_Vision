"""Distance statistics over a depth map or ROI."""

from __future__ import annotations

import numpy as np


class DistanceCalculator:
    """Computes per-frame distance statistics for a depth map."""

    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        confidence_threshold: int = 80,
    ) -> None:
        self._min_d = min_depth
        self._max_d = max_depth
        self._conf_thresh = confidence_threshold

    def compute_stats(
        self,
        depth_m: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> dict:
        """Return distance statistics over the whole frame."""
        mask = self._build_mask(depth_m, confidence)
        if not mask.any():
            return self._empty_stats()

        vals = depth_m[mask].flatten()
        return {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "valid_px": int(mask.sum()),
            "total_px": int(depth_m.size),
            "coverage": float(mask.sum() / depth_m.size),
        }

    def compute_roi_stats(
        self,
        depth_m: np.ndarray,
        confidence: np.ndarray | None,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> dict:
        """Return distance statistics for a rectangular ROI (pixels)."""
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(depth_m.shape[1], x + w)
        y2 = min(depth_m.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return self._empty_stats()

        roi_depth = depth_m[y1:y2, x1:x2]
        roi_conf = confidence[y1:y2, x1:x2] if confidence is not None else None
        mask = self._build_mask(roi_depth, roi_conf)
        if not mask.any():
            return self._empty_stats()

        vals = roi_depth[mask].flatten()
        return {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "valid_px": int(mask.sum()),
            "total_px": int(roi_depth.size),
            "coverage": float(mask.sum() / roi_depth.size),
            "roi": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
        }

    def _build_mask(self, depth_m: np.ndarray, confidence: np.ndarray | None) -> np.ndarray:
        mask = (depth_m > self._min_d) & (depth_m < self._max_d)
        if confidence is not None:
            mask &= (confidence >= self._conf_thresh)
        return mask

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0,
            "median": 0.0, "std": 0.0,
            "valid_px": 0, "total_px": 0, "coverage": 0.0,
        }
