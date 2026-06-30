"""Stereo_Vision service entry point."""
import torch
try:
    import torch_directml
    # 初始化AMD DML加速设备
    dml_device = torch_directml.device()
    torch.set_default_device(dml_device)
    print(f"[GPU ACCEL] AMD Radeon 780M DirectML 已启用，设备：{dml_device}")
except ImportError:
    print("[WARNING] 未安装torch-directml，YOLO将仅使用CPU运行，延迟/掉帧风险高")
    dml_device = torch.device("cpu")
    
import argparse
import sys

from aiohttp import web

from config import HOST, PORT, create_ssl_context
from state import init_app_state
from wiring import register_lifecycle
from routers import (
    register_page_routes,
    register_stream_routes,
    register_control_routes,
    register_depth_routes,
    register_obstacle_routes,
)
from static_serve import register_static_routes


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo_Vision VR service")
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Start ngrok tunnel alongside HTTPS server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override default port (default: 9000)",
    )
    return parser.parse_args()


async def health_check(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "Stereo_Vision"})


def create_app() -> web.Application:
    app = web.Application()
    init_app_state(app)
    register_lifecycle(app)

    app.router.add_get("/health", health_check)
    register_page_routes(app)
    register_stream_routes(app)
    register_control_routes(app)
    register_depth_routes(app)
    register_obstacle_routes(app)
    register_static_routes(app)

    return app


def main():
    args = parse_args()
    port = args.port if args.port is not None else PORT

    app = create_app()

    ssl_ctx = None
    try:
        ssl_ctx = create_ssl_context()
    except FileNotFoundError as ex:
        print(f"[Stereo_Vision] WARNING: {ex}")
        print("  -> Falling back to HTTP (for local debugging only)")
        print("  -> For production, provide TLS cert/key:")
        print("       ./config/gen_self_signed_cert.sh (or .ps1)")
        print("       then set STEREO_CERT_FILE / STEREO_KEY_FILE env vars")

    scheme = "https" if ssl_ctx else "http"
    print(f"[Stereo_Vision] Starting on {scheme}://{HOST}:{port}/")
    print(f"[Stereo_Vision] Pages:")
    print(f"  {scheme}://{HOST}:{port}/           # home")
    print(f"  {scheme}://{HOST}:{port}/control    # parameter control panel")

    try:
        web.run_app(
            app,
            host=HOST,
            port=port,
            ssl_context=ssl_ctx,
            shutdown_timeout=5.0,
            access_log=None,
        )
    except KeyboardInterrupt:
        print("\n[Stereo_Vision] Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
