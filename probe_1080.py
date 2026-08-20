"""Probe camera index 1 at 1920x1080 per-eye widths to find a1080 capable mode."""
import cv2
import time

for backend_id, backend_name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
    print(f"\n=== {backend_name} (index=1) ===")
    cap = cv2.VideoCapture(1, backend_id)
    if not cap.isOpened():
        print(f"  index 1 not opened")
        cap.release()
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for w, h in [(3840, 1080), (1920, 1080), (2560, 720), (1920, 540), (1280, 720), (1280, 960)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.4)
        ret, frame = cap.read()
        if ret and frame is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  req {w}x{h} -> got {actual_w}x{actual_h} (read shape {frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"  req {w}x{h} -> READ FAILED")
    cap.release()