"""快速检查：把指定 index 的左右画面分别保存为 left.png 和 right.png，肉眼检查。"""
import cv2, time, sys

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out_prefix = sys.argv[2] if len(sys.argv) > 2 else f"cam{idx}"

cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Index {idx} 打开失败")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# 双目相机 SBS 整帧 = 3840x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
time.sleep(0.5)

# 采集 5 帧
import os
os.makedirs("diag_frames", exist_ok=True)
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    cv2.imwrite(f"diag_frames/{out_prefix}_full_{i}.jpg", frame)
    cv2.imwrite(f"diag_frames/{out_prefix}_left_{i}.jpg", frame[:, :w//2])
    cv2.imwrite(f"diag_frames/{out_prefix}_right_{i}.jpg", frame[:, w//2:])
    print(f"Frame {i}: saved full={w}x{h} left={w//2}x{h} right={w//2}x{h}")

cap.release()
print(f"\nFiles saved in diag_frames/ - check {out_prefix}_left_0.jpg vs {out_prefix}_right_0.jpg")
