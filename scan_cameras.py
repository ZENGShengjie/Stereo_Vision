"""Scan all camera indices and their properties."""
import cv2, sys

print(f"OpenCV version: {cv2.__version__}")
print()

for idx in range(10):
    for backend, backend_name in [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_VFW, "VFW"),
        (cv2.CAP_ANY, "ANY"),
    ]:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Determine if it's likely the phone (1280x720 or wider than tall)
        tag = " <-- likely PHONE" if w >= 1280 and w > h else ""
        tag = tag or (" <-- likely WEBCAM" if h >= 480 and h >= w else "")

        fourcc_str = ""
        if fourcc > 0:
            fourcc_str = f" (fourcc=0x{fourcc:08X})"

        print(f"Index {idx} [{backend_name}]: {w}x{h}, fps={fps:.1f}{fourcc_str}{tag}")
        print(f"  Frame mean={frame.mean():.1f}, shape={frame.shape}")
        cap.release()
        break  # one backend per index is enough
