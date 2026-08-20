"""Configuration for Stereo_Vision service.

合并自原 ``config.py``(根目录单文件)+ 子模块 ``hardware``。
"""
from __future__ import annotations

import os
import ssl
from pathlib import Path

from .hardware import (  # noqa: F401  (re-export)
    BASELINE_CM,
    CALIB_NPZ_PATH,
    CHARUCO_COLS,
    CHARUCO_DICT,
    CHARUCO_MARKER_MM,
    CHARUCO_MIN_CORNERS,
    CHARUCO_ROWS,
    CHARUCO_SQUARE_MM,
    DANGER_CM,
    DANGER_FLASH_HZ,
    DEFAULT_CONF_THRESHOLD,
    DEPTH_MAX_CM,
    DEPTH_MIN_CM,
    DEPTH_SMOOTH_WINDOW,
    FOCAL_LENGTH_MM,
    HFOV_DEG,
    MONO_HEIGHT,
    MONO_WIDTH,
    SAFE_CM,
    SGBM_BLOCK_SIZE,
    SGBM_HEIGHT,
    SGBM_MEDIAN_KSIZE,
    SGBM_NUM_DISPARITIES,
    SGBM_P1_MULT,
    SGBM_P2_MULT,
    SGBM_WIDTH,
    SBS_HEIGHT,
    SBS_WIDTH,
    TARGET_CLS_ID,
    TARGET_CLS_NAME,
    compute_focal_px,
    resize_for_sgbm,
    resize_to_mono,
    z_cm_from_disparity,
)

# 注意: BASE_DIR 是仓库根,不是 config 包根
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

HOST = os.getenv("STEREO_HOST", "0.0.0.0")
PORT = int(os.getenv("STEREO_PORT", "9000"))

CAMERA_TYPE = os.getenv("STEREO_CAMERA_TYPE", "usb").lower()

ZED_RESOLUTION = os.getenv("STEREO_ZED_RESOLUTION", "HD720")
ZED_FPS = int(os.getenv("STEREO_ZED_FPS", "30"))

USB_LEFT_INDEX = int(os.getenv("STEREO_USB_LEFT_INDEX", "1"))
USB_RIGHT_INDEX = int(os.getenv("STEREO_USB_RIGHT_INDEX", "1"))
# 单目(每只眼)原生分辨率:1920x1080
# 双目 SBS 整帧:3840x1080(左 1920x1080 + 右 1920x1080)
USB_TARGET_WIDTH = int(os.getenv("STEREO_USB_WIDTH", "1920"))
USB_TARGET_HEIGHT = int(os.getenv("STEREO_USB_HEIGHT", "1080"))
USB_FPS = int(os.getenv("STEREO_USB_FPS", "10"))
USB_STEREO_SCALE = float(os.getenv("STEREO_USB_SCALE", "0.5"))

# 1920x1080 单目 / 3840x1080 双目 SBS 带宽需求大幅上升:
# 4 Mbps 在 1920p 下会产生严重马赛克,提到 8 Mbps;
# 帧率从 30 降到 10,避免 USB 2.0 上行带宽(实测 index 1 在 1080p 最高 ~10fps)。
WEBRTC_MAX_BITRATE = int(os.getenv("STEREO_WEBRTC_MAX_BITRATE", "8000000"))
WEBRTC_MAX_FRAMERATE = int(os.getenv("STEREO_WEBRTC_MAX_FRAMERATE", "10"))

ENABLE_NGROK = os.getenv("STEREO_NGROK", "0") == "1"
NGROK_API_PORT = int(os.getenv("STEREO_NGROK_API_PORT", "4040"))


def _config_path(env_name: str, default_name: str) -> Path:
    value = os.getenv(env_name)
    if value:
        return Path(value).expanduser().resolve()
    candidates = [
        CONFIG_DIR / default_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def get_webrtc_ice_servers() -> list[dict]:
    stun_raw = os.getenv(
        "STEREO_STUN_URLS", "stun:stun.l.google.com:19302"
    )
    stun_urls = [u.strip() for u in stun_raw.split(",") if u.strip()]
    turn_raw = os.getenv("STEREO_TURN_URLS", "")
    turn_urls = [u.strip() for u in turn_raw.split(",") if u.strip()]

    ice_servers: list[dict] = []
    if stun_urls:
        ice_servers.append({"urls": stun_urls})
    if turn_urls:
        server: dict = {"urls": turn_urls}
        username = os.getenv("STEREO_TURN_USERNAME", "")
        credential = os.getenv("STEREO_TURN_PASSWORD", "")
        if username:
            server["username"] = username
        if credential:
            server["credential"] = credential
        ice_servers.append(server)
    return ice_servers


def build_webrtc_client_config() -> dict:
    return {"iceServers": get_webrtc_ice_servers()}


def create_rtc_configuration(RTCConfiguration, RTCIceServer):
    ice_servers = []
    for server in get_webrtc_ice_servers():
        ice_servers.append(
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
        )
    return RTCConfiguration(iceServers=ice_servers)


def create_ssl_context() -> ssl.SSLContext:
    cert_file = _config_path("STEREO_CERT_FILE", "cert.pem")
    key_file = _config_path("STEREO_KEY_FILE", "key.pem")

    if not cert_file.exists() or not key_file.exists():
        raise FileNotFoundError(
            f"TLS certificate not found. "
            f"cert={cert_file}, key={key_file}. "
            f"Run: ./config/gen_self_signed_cert.sh (or .ps1 on Windows)"
        )

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(cert_file, key_file)
    return ssl_ctx
