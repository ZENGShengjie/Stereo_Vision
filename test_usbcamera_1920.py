"""验证 USBCamera 在 index 1 上能开 3840x1080 SBS，并正确拆出左右眼。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接传环境变量，模拟 .env / scripts/run.sh 的配置
os.environ["STEREO_CAMERA_TYPE"] = "usb"
os.environ["STEREO_USB_LEFT_INDEX"] = "1"
os.environ["STEREO_USB_RIGHT_INDEX"] = "1"
os.environ["STEREO_USB_WIDTH"] = "1920"
os.environ["STEREO_USB_HEIGHT"] = "1080"
os.environ["STEREO_USB_FPS"] = "10"

from camera.usb_camera import USBCamera
import cv2, time

cam = USBCamera()  # 使用 config.py 默认值
print(f"[test] USBCamera created: left_idx={cam._left_index} right_idx={cam._right_index}")
print(f"[test] target: {cam._target_width}x{cam._target_height} fps={cam._fps} stereo_scale={cam._stereo_scale}")
print(f"[test] single_device={cam._single_device}")

# 抓 3 帧
os.makedirs("diag_frames6", exist_ok=True)
for i in range(3):
    sbs = cam.read_stereo()
    if sbs is None:
        print(f"[test] Frame {i}: read returned None")
        time.sleep(0.5)
        continue
    h, w = sbs.shape[:2]
    eye_w = w // 2
    left = sbs[:, :eye_w]
    right = sbs[:, eye_w:]
    diff = abs(left.astype(int) - right.astype(int)).mean()
    print(f"[test] Frame {i}: sbs={w}x{h} left={left.shape} right={right.shape} L-R diff={diff:.2f}")
    cv2.imwrite(f"diag_frames6/usbcam_full_{i}.jpg", sbs)
    cv2.imwrite(f"diag_frames6/usbcam_left_{i}.jpg", left)
    cv2.imwrite(f"diag_frames6/usbcam_right_{i}.jpg", right)
    time.sleep(0.3)

print("[test] Done. Check diag_frames6/usbcam_left_0.jpg vs usbcam_right_0.jpg")
