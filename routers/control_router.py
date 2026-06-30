"""Control router: get/update display params + depth calibration."""

from aiohttp import web

from config import CAMERA_TYPE
from services import control_service


async def get_params_handler(_request: web.Request) -> web.Response:
    return web.json_response(control_service.get_params())


async def update_params_handler(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    allowed_keys = {"crop", "xshift", "k1", "k2", "sep", "gshift_x", "gshift_y"}
    update = {k: v for k, v in data.items() if k in allowed_keys}
    result = control_service.update_params(update)
    return web.json_response(result)


async def reset_params_handler(_request: web.Request) -> web.Response:
    result = control_service.reset_params()
    return web.json_response(result)


async def status_handler(request: web.Request) -> web.Response:
    cam = request.app.get("stereo_camera")
    cam_available = request.app.get("stereo_available", False)
    status_str = "open" if cam_available else "not_available"
    return web.json_response({
        "camera": status_str,
        "camera_type": CAMERA_TYPE.strip(),
        "params": control_service.get_params(),
    })


async def calibrate_disp_scale_handler(request: web.Request) -> web.Response:
    """Live-set DISP_SCALE without restarting.

    POST /api/calibrate/disp_scale
    Body: { "real_cm": 25.0 }

    Reads the latest pipeline stats (d_median, z_measured) and computes:
        DISP_SCALE = real_cm / z_measured
    Then patches config.hardware.DISP_SCALE at runtime.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    real_cm = data.get("real_cm")
    if real_cm is None or not (0.5 <= float(real_cm) <= 500):
        return web.json_response(
            {"error": "real_cm must be a number between 0.5 and 500"}, status=400
        )
    real_cm = float(real_cm)

    pipeline = request.app.get("stereo_pipeline")
    if pipeline is None:
        return web.json_response({"error": "pipeline not available"}, status=503)

    stats = pipeline.stats_summary()
    z_measured = stats.get("z_measured_cm")
    d_median = stats.get("d_median_px")

    if z_measured is None or z_measured == 0:
        return web.json_response({
            "error": "no valid depth measurement yet",
            "hint": "Ensure cup is detected and visible, then try again",
        }, status=422)

    new_scale = round(real_cm / z_measured, 4)

    # Patch the module-level constant at runtime
    import config.hardware as hw
    old_scale = hw.DISP_SCALE
    hw.DISP_SCALE = new_scale

    return web.json_response({
        "old_disp_scale": old_scale,
        "new_disp_scale": new_scale,
        "real_cm": real_cm,
        "z_measured_cm": z_measured,
        "d_median_px": d_median,
        "formula": f"DISP_SCALE = real_cm / z_measured = {real_cm} / {z_measured} = {new_scale}",
        "note": "Restart the server to persist; current session updated live",
    })


async def calibrate_depth_ratio_handler(request: web.Request) -> web.Response:
    """在线单点校准深度比例系数,无需改底层 DISP_SCALE。

    POST /api/calibrate/depth_ratio
    Body: { "true_cm": 20.0 }

    工作原理:
    - 传感器在 true_cm 处读出 z_measured_cm (当前 UI 显示值)
    - 计算 ratio = true_cm / z_measured_cm
    - 之后所有 depth_cm 输出都乘以 ratio

    使用方法:
    1. 把杯子放在已知距离 true_cm 的位置
    2. 等数值稳定后,记下 UI 显示的 z_measured_cm
    3. curl -X POST http://localhost:9000/api/calibrate/depth_ratio
          -H "Content-Type: application/json" -d '{"true_cm": 20.0}'
    4. 之后 14cm+ 的所有读数都会乘以校正系数

    注意:ratio 跨全距离生效,只需校准一次。
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    true_cm = data.get("true_cm")
    if true_cm is None or not (0.5 <= float(true_cm) <= 200):
        return web.json_response(
            {"error": "true_cm must be a number between 0.5 and 200"}, status=400
        )
    true_cm = float(true_cm)

    pipeline = request.app.get("stereo_pipeline")
    if pipeline is None:
        return web.json_response({"error": "pipeline not available"}, status=503)

    stats = pipeline.stats_summary()
    z_measured = stats.get("z_measured_cm")   # 未校正原始传感器读数

    if z_measured is None or z_measured == 0:
        return web.json_response({
            "error": "no valid depth measurement yet",
            "hint": "Ensure cup is detected and visible, then try again",
        }, status=422)

    result = pipeline.calibrate_depth(z_measured, true_cm)

    return web.json_response({
        "status": "calibration point added",
        "sensor_z_cm": z_measured,
        "true_cm": true_cm,
        "calib_points": result["calib_points"],
        "note": "Need at least 2 points for interpolation. Current points: " +
                str(len(result["calib_points"])),
    })


def register_control_routes(app: web.Application) -> None:
    app.router.add_get("/api/params", get_params_handler)
    app.router.add_post("/api/params", update_params_handler)
    app.router.add_post("/api/params/reset", reset_params_handler)
    app.router.add_get("/api/status", status_handler)
    app.router.add_post("/api/calibrate/disp_scale", calibrate_disp_scale_handler)
    app.router.add_post("/api/calibrate/depth_ratio", calibrate_depth_ratio_handler)
