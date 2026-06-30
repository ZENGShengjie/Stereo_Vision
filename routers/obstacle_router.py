"""Obstacle router: obstacle detection results."""

from __future__ import annotations

from aiohttp import web

from services import obstacle_service


async def obstacle_status_handler(request: web.Request) -> web.Response:
    """Return obstacle detection result for the current frame."""
    cam = request.app.get("stereo_camera")
    if cam is None or not request.app.get("stereo_available"):
        return web.json_response({"error": "Camera not available"}, status=503)

    result = cam.read_depth()
    if result is None:
        return web.json_response({"error": "grab failed"}, status=500)

    depth_m, confidence = result
    detection = obstacle_service.detect(depth_m, confidence)
    return web.json_response(detection)


async def obstacle_params_handler(request: web.Request) -> web.Response:
    """Update obstacle detection thresholds (GET = read, POST = write)."""
    if request.method == "GET":
        cfg = {
            "cell_width": obstacle_service._detector._cell_w,
            "cell_height": obstacle_service._detector._cell_h,
            "min_confidence": obstacle_service._detector._conf_thresh,
            "depth_threshold": obstacle_service._detector._depth_thresh,
            "coverage_ratio": obstacle_service._detector._cov_ratio,
        }
        return web.json_response(cfg)

    # POST
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    det = obstacle_service._detector
    det._cell_w = int(data.get("cell_width", det._cell_w))
    det._cell_h = int(data.get("cell_height", det._cell_h))
    det._conf_thresh = int(data.get("min_confidence", det._conf_thresh))
    det._depth_thresh = float(data.get("depth_threshold", det._depth_thresh))
    det._cov_ratio = float(data.get("coverage_ratio", det._cov_ratio))

    cfg = {
        "cell_width": det._cell_w,
        "cell_height": det._cell_h,
        "min_confidence": det._conf_thresh,
        "depth_threshold": det._depth_thresh,
        "coverage_ratio": det._cov_ratio,
    }
    return web.json_response(cfg)


def register_obstacle_routes(app: web.Application) -> None:
    app.router.add_get("/api/obstacle/status", obstacle_status_handler)
    app.router.add_route("GET", "/api/obstacle/params", obstacle_params_handler)
    app.router.add_route("POST", "/api/obstacle/params", obstacle_params_handler)
