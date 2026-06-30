"""扫描所有摄像头索引（扩大到 0-15），并打印更多信息用于调试。"""
import cv2, sys

print(f"OpenCV version: {cv2.__version__}")
print(f"Python: {sys.version}")
print()

found = []
for idx in range(16):
    for backend, backend_name in [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
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
        backend_id = int(cap.get(cv2.CAP_PROP_BACKENDNAME) if hasattr(cv2, "CAP_PROP_BACKENDNAME") else 0)

        # 解码 fourcc
        fourcc_str = ""
        if fourcc > 0:
            try:
                fourcc_str = chr((fourcc >> 0) & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
            except:
                fourcc_str = f"0x{fourcc:08X}"

        # 标签
        if w == 1280 and h == 480:
            tag = " <-- 疑似双目摄像头 (SBS) "
        elif w == 2560 and h == 720:
            tag = " <-- 疑似双目 (SBS 2560x720) "
        elif w == 2560 and h == 480:
            tag = " <-- 疑似双目 (SBS 2560x480) "
        elif w >= 1280 and w > h:
            tag = " <-- likely PHONE"
        elif h >= 480 and h >= w:
            tag = " <-- likely WEBCAM"
        else:
            tag = ""

        print(f"[OK ] Index {idx} [{backend_name}]: {w}x{h}, fps={fps:.1f}, fourcc={fourcc_str}{tag}")
        print(f"      Frame mean={frame.mean():.1f}, shape={frame.shape}")
        found.append((idx, backend_name, w, h))
        cap.release()
        break  # one backend per index is enough
    else:
        # 没有backend能用，但可能在某个backend能打开但读不到帧
        # 再尝试ANY
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[???] Index {idx} [ANY]: opened but read failed, reported {w}x{h}")
        cap.release()

print()
print(f"Total found: {len(found)}")
