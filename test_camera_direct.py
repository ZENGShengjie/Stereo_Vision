"""Direct test of USBCamera with same params as the running server."""
import sys
sys.path.insert(0, '.')
import cv2
from camera import USBCamera

print("[TEST] Initializing USBCamera with index=1, 640x480...")
try:
    cam = USBCamera(
        left_index=1,
        right_index=1,
        target_width=640,
        target_height=480,
        fps=10,
        stereo_scale=0.5,
    )
    print("[TEST] Camera opened, status:", cam.status())
    for i in range(5):
        sbs = cam.read_stereo()
        if sbs is None:
            print(f"[TEST] Frame {i}: None")
        else:
            print(f"[TEST] Frame {i}: shape={sbs.shape} dtype={sbs.dtype} mean={sbs.mean():.1f}")
    cam.close()
    print("[TEST] OK")
except Exception as ex:
    print(f"[TEST] FAIL: {ex}")