"""对单个 index 试多种分辨率，输出每种下左右画面 diff。
对 USB 双目摄像头，会找到一个分辨率使得左右画面不同。
"""
import cv2, time, sys

def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {idx} 打开失败")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(0.3)

    for w, h in [(640, 480), (1280, 480), (1280, 720), (2560, 480), (2560, 720)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.4)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Set {w}x{h} -> read failed")
            continue
        actual_w = frame.shape[1]
        actual_h = frame.shape[0]
        eye_w = actual_w // 2
        diff = abs(frame[:, :eye_w].astype(int) - frame[:, eye_w:].astype(int)).mean()
        status = "SBS OK" if diff > 5 else "单画面"
        print(f"Set {w}x{h} -> Got {actual_w}x{actual_h}, diff={diff:.1f} [{status}]")

    cap.release()

if __name__ == "__main__":
    main()
