"""Stereo camera abstract base class.

Defines the interface that all camera backends (ZED, USB, etc.) must implement.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class StereoCameraBase(ABC):
    """Interface for all stereo camera backends."""

    @abstractmethod
    def status(self) -> str:
        """Return camera status string."""
        ...

    @abstractmethod
    def read_stereo(self) -> np.ndarray | None:
        """Return SBS BGR image (left+right concatenated) or None on error."""
        ...

    @abstractmethod
    def read_rectified_pair(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return rectified (left, right) pair or None on error."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""
        ...
