"""Obstacle detection from a depth map.

Detects obstacles by finding regions whose median depth is below a
configurable threshold — useful for ground robots that must stop
or slow down when an obstacle is too close.
"""

from __future__ import annotations

import numpy as np


class ObstacleDetector:
    """Detects obstacles in a depth map using a grid-cell approach."""

    def __init__(
        self,
        cell_width: int = 64,
        cell_height: int = 64,
        min_confidence: int = 80,
        depth_threshold: float = 0.5,   # metres; below this = obstacle
        coverage_ratio: float = 0.3,    # cell fraction of deep pixels to flag as obstacle
    ) -> None:
        self._cell_w = cell_width
        self._cell_h = cell_height
        self._conf_thresh = min_confidence
        self._depth_thresh = depth_threshold
        self._cov_ratio = coverage_ratio

    def detect(
        self,
        depth_m: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> dict:
        """Return obstacle detection result for the whole frame.

        Returns:
            {
              "clear": bool,          # True if path is clear
              "obstacles": [...],     # List of grid cells flagged as obstacle
              "min_distance": float,  # Closest valid depth (m)
              "grid_cols": int,
              "grid_rows": int,
            }
        """
        h, w = depth_m.shape
        cols = max(1, w // self._cell_w)
        rows = max(1, h // self._cell_h)

        obstacles: list[dict] = []
        min_dist = float("inf")

        for r in range(rows):
            for c in range(cols):
                x1 = c * self._cell_w
                y1 = r * self._cell_h
                x2 = min(x1 + self._cell_w, w)
                y2 = min(y1 + self._cell_h, h)

                cell_d = depth_m[y1:y2, x1:x2]
                cell_c = confidence[y1:y2, x1:x2] if confidence is not None else None

                # Build valid mask
                mask = cell_d > 0.05
                if cell_c is not None:
                    mask &= (cell_c >= self._conf_thresh)

                if not mask.any():
                    continue

                cell_valid = cell_d[mask]
                cell_min = float(np.min(cell_valid))
                min_dist = min(min_dist, cell_min)

                deep_count = int(np.sum(cell_valid < self._depth_thresh))
                cov = deep_count / cell_d.size

                if cov >= self._cov_ratio:
                    obstacles.append({
                        "col": c,
                        "row": r,
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "min_depth": cell_min,
                        "coverage": float(cov),
                    })

        return {
            "clear": len(obstacles) == 0,
            "obstacles": obstacles,
            "min_distance": float(min_dist) if min_dist != float("inf") else 0.0,
            "grid_cols": cols,
            "grid_rows": rows,
        }
