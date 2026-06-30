"""Runtime-adjustable display parameters for the VR stereo view."""

from typing import TypedDict


class DisplayParams(TypedDict, total=False):
    crop: float
    xshift: float
    k1: float
    k2: float
    sep: int
    gshift_x: float
    gshift_y: float


DEFAULT_DISPLAY_PARAMS: DisplayParams = {
    "crop": 0.7,
    "xshift": 0,
    "k1": -0.18,
    "k2": 0.04,
    "sep": 40,
    "gshift_x": 0,
    "gshift_y": 0,
}


class DisplayParamsStore:
    """Thread-safe mutable store for display parameters."""

    def __init__(self) -> None:
        self._params: DisplayParams = DEFAULT_DISPLAY_PARAMS.copy()

    def get(self) -> DisplayParams:
        return self._params.copy()

    def update(self, params: DisplayParams) -> None:
        validated = self._validate(params)
        self._params = validated

    def reset(self) -> None:
        self._params = DEFAULT_DISPLAY_PARAMS.copy()

    @staticmethod
    def _validate(params: DisplayParams) -> DisplayParams:
        return DisplayParams(
            crop=max(0.5, min(1.0, float(params.get("crop", 0.7)))),
            xshift=max(-80, min(80, float(params.get("xshift", 0)))),
            k1=float(params.get("k1", -0.18)),
            k2=float(params.get("k2", 0.04)),
            sep=max(0, int(params.get("sep", 40))),
            gshift_x=float(params.get("gshift_x", 0)),
            gshift_y=float(params.get("gshift_y", 0)),
        )
