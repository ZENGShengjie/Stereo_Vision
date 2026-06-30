@echo off
REM Run Stereo_Vision service on Windows.

REM Activate virtual environment if present
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM === USB 双目摄像头配置（单设备 SBS 模式：index 1 摄像头直接输出 SBS 整帧 3840x1080） ===
REM 单目每眼原生 1920x1080；双目 SBS 整帧 3840x1080
set STEREO_CAMERA_TYPE=usb
set STEREO_USB_LEFT_INDEX=1
set STEREO_USB_RIGHT_INDEX=1
set STEREO_USB_WIDTH=1920
set STEREO_USB_HEIGHT=1080
set STEREO_USB_FPS=10
set STEREO_USB_SCALE=0.5
set STEREO_PORT=9000

%PYTHON% main.py %*
