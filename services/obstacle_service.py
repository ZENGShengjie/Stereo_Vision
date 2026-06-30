"""Obstacle service: wraps obstacle_detector."""

from processing import ObstacleDetector


class ObstacleService:
    def __init__(
        self,
        cell_width: int = 64,
        cell_height: int = 64,
        min_confidence: int = 80,
        depth_threshold: float = 0.5,
        coverage_ratio: float = 0.3,
    ) -> None:
        self._detector = ObstacleDetector(
            cell_width=cell_width,
            cell_height=cell_height,
            min_confidence=min_confidence,
            depth_threshold=depth_threshold,
            coverage_ratio=coverage_ratio,
        )

    def detect(self, depth_m, confidence=None):
        return self._detector.detect(depth_m, confidence)


obstacle_service = ObstacleService()
