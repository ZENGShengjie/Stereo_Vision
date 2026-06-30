"""Routers module."""

from .page_router import register_page_routes
from .stream_router import register_stream_routes
from .control_router import register_control_routes
from .depth_router import register_depth_routes
from .obstacle_router import register_obstacle_routes

__all__ = [
    "register_page_routes",
    "register_stream_routes",
    "register_control_routes",
    "register_depth_routes",
    "register_obstacle_routes",
]
