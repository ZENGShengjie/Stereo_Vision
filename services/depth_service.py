"""Depth service: wraps depth_processor + distance_calculator."""

from processing import DepthProcessor, DistanceCalculator


class DepthService:
    def __init__(self) -> None:
        # 深度范围 0.1~3.0m，JET 色图（蓝=近，红=远）
        self._processor = DepthProcessor(
            colormap=2,       # COLORMAP_JET
            min_depth=0.1,
            max_depth=3.0,
            median_ksize=3,   # 减小模糊核，加速
        )
        self._calculator = DistanceCalculator()

    def process(self, depth_m):
        return self._processor.process(depth_m)

    def process_raw(self, depth_m):
        return self._processor.process_raw(depth_m)

    def stats(self, depth_m, confidence=None):
        return self._calculator.compute_stats(depth_m, confidence)

    def roi_stats(self, depth_m, confidence, x, y, w, h):
        return self._calculator.compute_roi_stats(depth_m, confidence, x, y, w, h)

    def update_display_params(self, min_depth=None, max_depth=None, colormap=None, median_ksize=None) -> dict:
        """Update depth display params at runtime."""
        if min_depth is not None:
            self._processor.update_range(min_depth, self._processor._max_d)
        if max_depth is not None:
            self._processor.update_range(self._processor._min_d, max_depth)
        if colormap is not None:
            self._processor.update_colormap(colormap)
        if median_ksize is not None:
            self._processor.update_median_ksize(median_ksize)
        return self._processor.get_params()

    def get_display_params(self) -> dict:
        return self._processor.get_params()


depth_service = DepthService()
