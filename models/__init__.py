"""Models module."""

from .display_params import DEFAULT_DISPLAY_PARAMS, DisplayParams, DisplayParamsStore

# Shared global store — imported by processing/display.py and services/control_service.py
current_params_store = DisplayParamsStore()

__all__ = [
    "DisplayParams",
    "DEFAULT_DISPLAY_PARAMS",
    "DisplayParamsStore",
    "current_params_store",
]
