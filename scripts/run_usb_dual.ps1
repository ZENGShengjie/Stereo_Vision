@echo off
REM 用单设备 USB 双目摄像头（index 1 直接输出 SBS 整帧 3840x1080）启动 Stereo_Vision 服务。
REM 单目每眼原生 1920x1080；双目 SBS 整帧 3840x1080

set PYTHON=.venv\Scripts\python.exe

REM 摄像头选择（单设备模式：左右都用 index 1）
set STEREO_CAMERA_TYPE=usb
set STEREO_USB_LEFT_INDEX=1
set STEREO_USB_RIGHT_INDEX=1

REM 分辨率与帧率（单设备 SBS：摄像头开 3840x1080 = 2*1920 x 1080）
set STEREO_USB_WIDTH=1920
set STEREO_USB_HEIGHT=1080
set STEREO_USB_FPS=10

REM 立体校正缩放比例（用于深度图计算的小分辨率）
set STEREO_USB_SCALE=0.5

REM 端口
set STEREO_PORT=9000

%PYTHON% main.py
