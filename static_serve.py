"""Serve static files (CSS, JS) from the static/ directory."""

from aiohttp import web
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"


def register_static_routes(app: web.Application) -> None:
    app.router.add_static("/static", str(STATIC_DIR), show_index=True)
