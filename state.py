"""Application state keys for Stereo_Vision service."""
from __future__ import annotations

from typing import Any

WEBRTC_PEERS_KEY = "stereo_webrtc_peers"
CAMERA_KEY = "stereo_camera"
STEREO_AVAILABLE_KEY = "stereo_available"
PIPELINE_KEY = "stereo_pipeline"
"""SBSPipeline 实例的 app state key;启动后由 wiring.lifecycle 写入。"""

APP_STATE_KEYS = [WEBRTC_PEERS_KEY, CAMERA_KEY, STEREO_AVAILABLE_KEY, PIPELINE_KEY]


def init_app_state(app: Any) -> None:
    """初始化 aiohttp app 状态。

    - ``stereo_pipeline`` 由 :mod:`wiring.lifecycle` 启动时写入 SBSPipeline。
    - ``stereo_camera`` 由 lifecycle 写入 USBCamera / ZEDCamera 实例。
    - ``stereo_webrtc_peers`` 用于 WebRTC 端点管理。
    """
    app[WEBRTC_PEERS_KEY] = {}
    app[CAMERA_KEY] = None
    app[STEREO_AVAILABLE_KEY] = False
    app[PIPELINE_KEY] = None
