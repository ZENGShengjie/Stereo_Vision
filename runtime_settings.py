"""Runtime settings loaded from config file or environment.

Currently mirrors config.py for simplicity; split out so that
runtime updates (e.g. via control panel) do not require a restart.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "config"

STEREO_HOST = "0.0.0.0"
STEREO_PORT = 9000
STEREO_WEBRTC_MAX_BITRATE = 4000000
STEREO_WEBRTC_MAX_FRAMERATE = 30
STEREO_NGROK = False
