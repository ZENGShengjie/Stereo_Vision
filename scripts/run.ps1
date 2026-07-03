@echo off
REM Run Stereo_Vision service on Windows.

REM Activate virtual environment if present
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM === USB 双目摄像头配置 ===
REM 2026-07-03: 改用 540 height — 某些廉价 USB 摄像头在 1080 模式下 read() 要 ~1000ms 只能
REM 跑 1fps。STEREO_USB_HEIGHT=540 让摄像头切到快读模式(实测 ~180ms, 5.5fps 上限)。
REM 想切回 1080 全分辨率编辑此文件,把 540 改回 1080 即可。
set STEREO_CAMERA_TYPE=usb
set STEREO_USB_LEFT_INDEX=1
set STEREO_USB_RIGHT_INDEX=1
set STEREO_USB_WIDTH=1920
set STEREO_USB_HEIGHT=540
set STEREO_USB_FPS=10
set STEREO_USB_SCALE=0.5
set STEREO_PORT=9000

%PYTHON% main.py %*
