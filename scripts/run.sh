#!/usr/bin/env bash
# Run Stereo_Vision service with USB dual-camera configuration.

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# === USB dual-camera configuration (single-device SBS mode: index 1 outputs SBS 3840x1080) ===
# Each eye: 1920x1080 native; SBS combined: 3840x1080
export STEREO_CAMERA_TYPE=usb
export STEREO_USB_LEFT_INDEX=1
export STEREO_USB_RIGHT_INDEX=1
export STEREO_USB_WIDTH=1920
export STEREO_USB_HEIGHT=1080
export STEREO_USB_FPS=10
export STEREO_USB_SCALE=0.5
export STEREO_PORT=9000

$PYTHON main.py "$@"
