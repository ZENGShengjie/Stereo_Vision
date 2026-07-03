"""Page routers: home, depth, control, debug."""

from pathlib import Path
from aiohttp import web
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _build_jinja(app: web.Application):
    if "jinja" not in app:
        app["jinja"] = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
        )
    return app["jinja"]


async def home_page(request: web.Request) -> web.Response:
    jinja = _build_jinja(request.app)
    template = jinja.get_template("home.html")
    return web.Response(
        text=template.render(remote=request.remote or "local"),
        content_type="text/html",
    )


async def depth_page(request: web.Request) -> web.Response:
    jinja = _build_jinja(request.app)
    template = jinja.get_template("depth.html")
    return web.Response(text=template.render(), content_type="text/html")


async def control_page(request: web.Request) -> web.Response:
    jinja = _build_jinja(request.app)
    template = jinja.get_template("control.html")
    return web.Response(text=template.render(), content_type="text/html")


async def debug_page(request: web.Request) -> web.Response:
    jinja = _build_jinja(request.app)
    template = jinja.get_template("debug.html")
    return web.Response(text=template.render(), content_type="text/html")


async def stereo_page(request: web.Request) -> web.Response:
    jinja = _build_jinja(request.app)
    template = jinja.get_template("stereo.html")
    return web.Response(text=template.render(), content_type="text/html")


async def perf_page(request: web.Request) -> web.Response:
    jinja = _build_jinja(request.app)
    template = jinja.get_template("perf.html")
    return web.Response(text=template.render(), content_type="text/html")


async def hud_page(request: web.Request) -> web.Response:
    """2026-07-04: HUD page = video feed + on-video control panel."""
    jinja = _build_jinja(request.app)
    template = jinja.get_template("hud.html")
    return web.Response(text=template.render(), content_type="text/html")


def register_page_routes(app: web.Application) -> None:
    app.router.add_get("/", home_page)
    app.router.add_get("/stereo", stereo_page)
    app.router.add_get("/depth", depth_page)
    app.router.add_get("/control", control_page)
    app.router.add_get("/debug", debug_page)
    app.router.add_get("/perf", perf_page)
    app.router.add_get("/hud", hud_page)
