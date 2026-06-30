"""Depth processing: colormap, filtering, ROI stats."""

from __future__ import annotations

import cv2
import numpy as np


class DepthProcessor:
    """Converts raw depth maps into displayable/coloured images."""

    def __init__(
        self,
        colormap: int = cv2.COLORMAP_JET,
        min_depth: float = 0.1,    # metres
        max_depth: float = 3.0,   # metres
        median_ksize: int = 5,
    ) -> None:
        self._colormap = colormap
        self._min_d = min_depth
        self._max_d = max_depth
        self._ksize = median_ksize

    def update_range(self, min_depth: float, max_depth: float) -> None:
        """Update depth range at runtime (called by depth service)."""
        self._min_d = min_depth
        self._max_d = max_depth

    def update_colormap(self, colormap: int) -> None:
        """Update colormap at runtime."""
        self._colormap = colormap

    def update_median_ksize(self, ksize: int) -> None:
        """Update median blur kernel size at runtime (must be odd)."""
        if ksize >= 3 and ksize % 2 == 1:
            self._ksize = ksize

    def process(self, depth_m: np.ndarray) -> np.ndarray | None:
        """Convert depth (m) to a uint8 BGR colormap image, or None."""
        if depth_m is None:
            return None

        valid_mask = (depth_m > self._min_d) & (depth_m < self._max_d)
        if not valid_mask.any():
            h, w = depth_m.shape
            blank = np.full((h, w, 3), fill_value=40, dtype=np.uint8)
            cv2.putText(blank, "No depth data", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)
            return blank

        d = np.clip(depth_m, self._min_d, self._max_d)
        norm = ((d - self._min_d) / (self._max_d - self._min_d) * 255).astype(np.uint8)

        if self._ksize > 1:
            norm = cv2.medianBlur(norm, self._ksize)

        coloured = cv2.applyColorMap(norm, self._colormap)
        return coloured

    def process_raw(self, depth_m: np.ndarray) -> np.ndarray:
        """Return a grayscale (0-255) depth image without colormap."""
        if depth_m is None:
            return np.zeros((1, 1), dtype=np.uint8)
        d = np.clip(depth_m, self._min_d, self._max_d)
        return ((d - self._min_d) / (self._max_d - self._min_d) * 255).astype(np.uint8)

    def depth_histogram(
        self, depth_m: np.ndarray, bins: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (counts, bin_edges) for a histogram of valid depth values."""
        valid = depth_m[(depth_m > self._min_d) & (depth_m < self._max_d)]
        if valid.size == 0:
            return np.zeros(bins), np.linspace(0, 1, bins + 1)
        return np.histogram(valid.flatten(), bins=bins)

    def get_params(self) -> dict:
        """Return current display params."""
        return {
            "min_depth": self._min_d,
            "max_depth": self._max_d,
            "colormap": self._colormap,
            "median_ksize": self._ksize,
        }
