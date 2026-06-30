"""Depth router: depth colormap stream and per-frame stats."""

from __future__ import annotations

import cv2
import numpy as np

from aiohttp import web

from services import depth_service
from transport.mjpeg_stream import mjpeg_generator_async


async def depth_colormap_handler(request: web.Request) -> web.Response:
    """Streaming MJPEG of the depth colormap."""
    cam = request.app.get("stereo_camera")
    if cam is None or not request.app.get("stereo_available"):
        return web.Response(text="Camera not available", status=503)

    def frame_fn():
        result = cam.read_depth()
        if result is None:
            return None
        depth_m, _ = result
        return depth_service.process(depth_m)

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache",
            "Connection": "close",
        },
    )
    await response.prepare(request)
    try:
        async for chunk in mjpeg_generator_async(frame_fn, fps=10):
            try:
                await response.write(chunk)
            except ConnectionResetError:
                break
    except Exception:
        pass
    finally:
        try:
            await response.write_eof()
        except Exception:
            pass
    return response


async def depth_snapshot_handler(request: web.Request) -> web.Response:
    """Return a single JPEG frame of the depth colormap."""
    cam = request.app.get("stereo_camera")
    if cam is None or not request.app.get("stereo_available"):
        return web.Response(text="Camera not available", status=503)

    result = cam.read_depth()
    if result is None:
        return web.Response(text="Grab failed", status=500)

    depth_m, _ = result
    frame = depth_service.process(depth_m)
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return web.Response(body=jpeg.tobytes(), content_type="image/jpeg")


async def depth_stats_handler(request: web.Request) -> web.Response:
    """Return per-frame depth statistics as JSON."""
    cam = request.app.get("stereo_camera")
    if cam is None or not request.app.get("stereo_available"):
        return web.json_response({"error": "Camera not available"}, status=503)

    result = cam.read_depth()
    if result is None:
        return web.json_response({"error": "grab failed"}, status=500)

    depth_m, confidence = result
    stats = depth_service.stats(depth_m, confidence)
    return web.json_response(stats)


async def depth_roi_handler(request: web.Request) -> web.Response:
    """Return depth statistics for a rectangular ROI.

    Query params: x, y, w, h (all integers, pixels).
    """
    try:
        x = int(request.query.get("x", 0))
        y = int(request.query.get("y", 0))
        w = int(request.query.get("w", 100))
        h = int(request.query.get("h", 100))
    except ValueError:
        return web.json_response({"error": "invalid ROI params"}, status=400)

    cam = request.app.get("stereo_camera")
    if cam is None or not request.app.get("stereo_available"):
        return web.json_response({"error": "Camera not available"}, status=503)

    result = cam.read_depth()
    if result is None:
        return web.json_response({"error": "grab failed"}, status=500)

    depth_m, confidence = result
    stats = depth_service.roi_stats(depth_m, confidence, x, y, w, h)
    return web.json_response(stats)


def register_depth_routes(app: web.Application) -> None:
    app.router.add_get("/depth/snapshot", depth_snapshot_handler)
    app.router.add_get("/depth/feed", depth_colormap_handler)
    app.router.add_get("/api/depth/stats", depth_stats_handler)
    app.router.add_get("/api/depth/roi", depth_roi_handler)
    app.router.add_get("/api/depth/display", _depth_display_get)
    app.router.add_post("/api/depth/display", _depth_display_set)


async def _depth_display_get(request: web.Request) -> web.Response:
    return web.json_response(depth_service.get_display_params())


async def _depth_display_set(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    params = depth_service.update_display_params(
        min_depth=body.get("min_depth"),
        max_depth=body.get("max_depth"),
        colormap=body.get("colormap"),
        median_ksize=body.get("median_ksize"),
    )
    return web.json_response(params)
