"""Lightweight camera probe (works with OpenCV 5.0+)."""
import cv2, sys

print(f"OpenCV version: {cv2.__version__}")
print()

backends = []
for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
    if hasattr(cv2, name):
        backends.append((getattr(cv2, name), name.replace("CAP_", "")))

found = []
for idx in range(6):
    opened_idx = None
    for backend, backend_name in backends:
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception as ex:
            print(f"Index {idx} [{backend_name}]: backend error {ex}")
            continue
        if not cap.isOpened():
            cap.release()
            continue
        # Try a quick read
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        tag = f" [{backend_name}] "
        if ret and frame is not None:
            print(f"Index {idx}{tag}OK -> {w}x{h} fps={fps:.1f} mean={frame.mean():.1f}")
            found.append((idx, backend_name, w, h))
            cap.release()
            opened_idx = backend_name
            break
        else:
            print(f"Index {idx}{tag}opened but read() failed")
        cap.release()
    if opened_idx is None:
        print(f"Index {idx}: not opened by any backend")

print()
print(f"FOUND {len(found)} cameras: {found}")