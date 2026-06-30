"""验证指定索引的摄像头是否为真正的 SBS（Side-by-Side）双目摄像头。

用法：
    python verify_sbs.py --index 1

会打印 3 帧的左右半边像素差值。diff > 5 表示左右是不同画面（即真正的双目摄像头）。
"""
import argparse
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1)
    args = parser.parse_args()

    index = args.index
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {index} 打开失败")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # 双目相机 SBS 整帧 = 3840x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(0.5)

    for i in range(3):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Frame {i}: 读取失败")
            continue
        h, w = frame.shape[:2]
        eye_w = w // 2
        left = frame[:, :eye_w]
        right = frame[:, eye_w:]
        diff = abs(left.astype(int) - right.astype(int)).mean()
        status = "OK: 左右不同" if diff > 5 else "ERROR: 左右相同"
        print(f"Frame {i}: {w}x{h}, diff={diff:.1f} [{status}]")

    cap.release()


if __name__ == "__main__":
    main()
