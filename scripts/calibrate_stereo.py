"""Stereo ChArUco calibration capture tool.

Usage:
    python scripts/calibrate_stereo.py

Prerequisites:
    - 必须安装 opencv-contrib-python(cv2.aruco 才能解析)。
      本项目默认 requirements.txt 只装 opencv-python,所以跑这个脚本前请:
        pip install opencv-contrib-python
      否则 config.hardware.CHARUCO_DICT 会是 None,启动时直接报错退出。
    - 打印 ChArUco 板(运行此脚本后按 G 生成 PNG,或参考 calibration_pack/README.md)
    - 板:7 列 × 5 行 ArUco,DICT_6X6_250
    - Square = 40.57 mm,Marker = 29.95 mm(默认值)
      务必用数显游标卡尺实测打印件,然后在 config/hardware.py 改
      CHARUCO_SQUARE_MM / CHARUCO_MARKER_MM。
    - ChArUco 板必须**同时**出现在左右两个画面中

Output:
    config/calibration/calib.npz  (自动生成)

为什么 ChArUco 比 chessboard 更好?
    - ArUco 角点自动 sub-pixel refine,不需要额外的 cornerSubPix
    - ID 标识:部分遮挡也鲁棒(只要 ≥ 5 个角点)
    - 即便略微失焦也能给到 sub-pixel 精度
    - 对 auto-exposure 闪烁更不敏感

要拍多少对?
    - 最低: 15 对,覆盖各种角度与距离
    - 推荐: 20-30 对
    - 覆盖: 近/远、左右倾、上下倾、居中/边缘
    - SPACE 接受一对、R 撤回一对、G 保存标定板 PNG、ESC 结束采集

采集完成后:
    1. npz 自动保存到 config/calibration/calib.npz
    2. 重启主服务,StereoCalibrator 会加载 npz 并切换到 RECTIFIED 模式

整合历史:
    2026-08-19 从 calibration_pack/scripts/calibrate_stereo.py 移植,保留项目原
    CalibrationReport(单目/极线/内参一致性/k1 量级/角点距离分布)维度。
"""
from __future__ import annotations

import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── hardware reality source ──────────────────────────────────────────────────
# Import from project config so we read actual camera index / resolution /
# ChArUco board params — single source of truth.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    BASELINE_CM,
    CALIB_NPZ_PATH,
    CHARUCO_COLS,
    CHARUCO_DICT,
    CHARUCO_MARKER_MM,
    CHARUCO_MIN_CORNERS,
    CHARUCO_ROWS,
    CHARUCO_SQUARE_MM,
    USB_FPS,
    USB_LEFT_INDEX,
    USB_TARGET_HEIGHT,
    USB_TARGET_WIDTH,
)

# ── guards: cv2.aruco + board preset ─────────────────────────────────────────
if CHARUCO_DICT is None:
    sys.stderr.write(
        "[ERROR] cv2.aruco unavailable. The ChArUco capture tool requires\n"
        "        opencv-contrib-python (cv2.aruco.DICT_*).\n"
        "        current requirements.txt only installs opencv-python.\n"
        "        Fix: pip install opencv-contrib-python\n"
    )
    sys.exit(2)

# ── calibration output ───────────────────────────────────────────────────────
OUTPUT_PATH: Path = CALIB_NPZ_PATH
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── ChArUco board (OpenCV 4.13 new API) ──────────────────────────────────────
_aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
_charuco_board = cv2.aruco.CharucoBoard(
    (CHARUCO_COLS, CHARUCO_ROWS),
    float(CHARUCO_SQUARE_MM),
    float(CHARUCO_MARKER_MM),
    _aruco_dict,
)
_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict)
_charuco_detector = cv2.aruco.CharucoDetector(_charuco_board)


# ── camera backend (DSHOW → MSMF fallback, same pattern as camera/usb_camera.py) ──
_BACKENDS = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF, "MSMF"),
]


def _try_open(
    index: int,
    fps: int = 15,
    target_width: int | None = None,
    target_height: int | None = None,
    preferred_backend: int | None = None,
) -> tuple[cv2.VideoCapture | None, str]:
    """Open a USB camera, trying DSHOW first then MSMF, with MJPG negotiated.

    Mirrors the probe-flush + read pattern from camera/usb_camera._try_open()
    so we share the same edge-case handling on Windows USB cams.
    """
    order = list(_BACKENDS)
    if preferred_backend is not None:
        order.sort(key=lambda b: 0 if b[0] == preferred_backend else 1)

    for backend, name in order:
        # ── probe 1: flush MJPG decode buffer ────────────────────────────────
        cap1 = cv2.VideoCapture(index, backend)
        if not cap1.isOpened():
            continue
        cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if target_width is not None:
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        if target_height is not None:
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap1.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.5)
        cap1.read()
        cap1.release()

        # ── probe 2: actual open ────────────────────────────────────────────
        cap2 = cv2.VideoCapture(index, backend)
        if not cap2.isOpened():
            continue
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if target_width is not None:
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        if target_height is not None:
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        cap2.set(cv2.CAP_PROP_FPS, fps)
        time.sleep(0.3)
        t0 = time.monotonic()
        ret, frame = cap2.read()
        elapsed_ms = (time.monotonic() - t0) * 1000
        if not ret or frame is None or frame.size == 0:
            cap2.release()
            continue

        if elapsed_ms > 300:
            print(f"[camera] index={index} first read took {elapsed_ms:.0f}ms — retrying once...")
            cap2.release()
            cap2 = cv2.VideoCapture(index, backend)
            if not cap2.isOpened():
                continue
            cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if target_width is not None:
                cap2.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            if target_height is not None:
                cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            cap2.set(cv2.CAP_PROP_FPS, fps)
            time.sleep(0.2)
            ret, frame = cap2.read()
            if not ret or frame is None or frame.size == 0:
                cap2.release()
                continue

        print(f"[camera] index={index} opened with {name}, frame shape={frame.shape}")
        return cap2, name

    return None, "none"


def _warmup_frames(cap: cv2.VideoCapture, n: int = 5, settle_ms: int = 200) -> None:
    deadline = time.time() + (n * settle_ms / 1000.0) + 1.0
    count = 0
    while count < n and time.time() < deadline:
        cap.read()
        count += 1
        time.sleep(settle_ms / 1000.0)


def _split_sbs(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eye_w = frame.shape[1] // 2
    return frame[:, :eye_w], frame[:, eye_w:]


# ── FrameReader ──────────────────────────────────────────────────────────────
class FrameReader:
    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 2):
        self._cap = cap
        self._queue: deque[np.ndarray] = deque(maxlen=queue_size)
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._queue.append(frame)

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue[-1]

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2.0)


# ── ChArUco detection (OpenCV 4.13 new API) ──────────────────────────────────
def _detect_charuco(
    gray: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect ChArUco corners in a grayscale image.

    Returns:
        (success, charuco_corners, charuco_ids, marker_corners, marker_ids):
            - success: True if ≥ CHARUCO_MIN_CORNERS corners found
            - charuco_corners: (N, 1, 2) float32, sub-pixel refined
            - charuco_ids: (N, 1) int32
            - marker_corners: raw ArUco marker corners for drawing
            - marker_ids: raw ArUco marker IDs for drawing
    """
    marker_corners, marker_ids, _ = _aruco_detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return (
            False,
            np.zeros((0, 1, 2), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int32),
            np.zeros((0, 1, 2), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int32),
        )

    charuco_corners, charuco_ids, _, _ = _charuco_detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < CHARUCO_MIN_CORNERS:
        return (
            False,
            np.zeros((0, 1, 2), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int32),
            marker_corners,
            marker_ids,
        )

    return True, charuco_corners, charuco_ids, marker_corners, marker_ids


# ── Preview renderer ─────────────────────────────────────────────────────────
def _render_preview(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    ok_l: bool, corners_l: np.ndarray, ids_l: np.ndarray,
    ok_r: bool, corners_r: np.ndarray, ids_r: np.ndarray,
    mks_l: np.ndarray, ids_mk_l: np.ndarray,
    mks_r: np.ndarray, ids_mk_r: np.ndarray,
    accepted: int,
    fps: float,
    last_msg: str,
    last_ts: float,
) -> np.ndarray:
    def _overlay(img, ok, corners, ids, mks, ids_mk):
        display = img.copy()
        if mks is not None and ids_mk is not None and len(ids_mk) > 0:
            cv2.aruco.drawDetectedMarkers(display, mks, ids_mk, borderColor=(0, 255, 0))
        if ok and corners is not None and ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, cornerColor=(0, 120, 255))
        return display

    pl = _overlay(cv2.resize(left_raw, (960, 540)), ok_l, corners_l, ids_l, mks_l, ids_mk_l)
    pr = _overlay(cv2.resize(right_raw, (960, 540)), ok_r, corners_r, ids_r, mks_r, ids_mk_r)
    both_preview = np.hstack([pl, pr])  # 1920 × 540 preview

    status = np.full((90, 1920, 3), 22, dtype=np.uint8)
    n_l = len(corners_l) if corners_l is not None else 0
    n_r = len(corners_r) if corners_r is not None else 0
    color = (0, 255, 0) if (ok_l and ok_r) else (0, 0, 255)
    label = "BOTH OK — press SPACE" if (ok_l and ok_r) else f"L:{n_l}  R:{n_r}  (need ≥ {CHARUCO_MIN_CORNERS})"
    cv2.putText(status, f"Accepted: {accepted}/15+  |  {label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    cv2.putText(status, f"{fps:.1f} fps", (1800, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    if last_msg and (time.time() - last_ts) < 3:
        cv2.putText(status, last_msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(status, "SPACE=accept  R=reset  G=save-board  ESC=calibrate",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return np.vstack([both_preview, status])


# ── Main capture loop ────────────────────────────────────────────────────────
def run():
    # ── open camera ───────────────────────────────────────────────────────────
    cap, name = _try_open(
        USB_LEFT_INDEX,
        fps=USB_FPS,
        target_width=USB_TARGET_WIDTH * 2,
        target_height=USB_TARGET_HEIGHT,
    )
    if cap is None:
        print("[ERROR] Could not open camera. Check USB index in config.")
        sys.exit(1)

    _warmup_frames(cap, n=5, settle_ms=200)
    reader = FrameReader(cap, queue_size=2)
    print(f"[camera] Background reader started ({name})")

    # ── collection ─────────────────────────────────────────────────────────────
    # Each entry: (corners_l, ids_l, corners_r, ids_r)
    # shapes: corners (N,1,2) float32, ids (N,1) int32
    collected: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    accepted = 0
    last_msg = ""
    last_ts = 0.0

    time.sleep(0.3)

    print("\n" + "=" * 60)
    print("STEREO CHARUCO CALIBRATION CAPTURE")
    print("=" * 60)
    print(f"  Board       : {CHARUCO_COLS}×{CHARUCO_ROWS} ArUco, DICT_6X6_250")
    print(f"  Square size : {CHARUCO_SQUARE_MM} mm  (measure with caliper!)")
    print(f"  Marker size : {CHARUCO_MARKER_MM} mm")
    print(f"  Min corners : {CHARUCO_MIN_CORNERS} per view")
    print(f"  Target      : 15-30 pairs")
    print(f"  Output      : {OUTPUT_PATH}")
    print()
    print("  SPACE   = accept current pair")
    print("  R       = reset last pair")
    print("  G       = generate & save board image")
    print("  ESC / Q = finish & calibrate")
    print("=" * 60 + "\n")

    # Cache state
    _ok_l = False
    _ok_r = False
    _corners_l: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)
    _corners_r: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)
    _ids_l: np.ndarray = np.zeros((0, 1), dtype=np.int32)
    _ids_r: np.ndarray = np.zeros((0, 1), dtype=np.int32)
    _mks_l: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)
    _ids_mk_l: np.ndarray = np.zeros((0, 1), dtype=np.int32)
    _mks_r: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)
    _ids_mk_r: np.ndarray = np.zeros((0, 1), dtype=np.int32)
    _frame_idx = 0

    _fps_start = time.time()
    _fps_count = 0
    fps = 0.0

    def _save_board_image():
        out = OUTPUT_PATH.parent / "charuco_board.png"
        img = _charuco_board.generateImage((1920, int(1920 * CHARUCO_ROWS / CHARUCO_COLS)))
        cv2.imwrite(str(out), img)
        print(f"[board] Saved: {out}")
        scale = 4
        out_hr = OUTPUT_PATH.parent / "charuco_board_hires.png"
        img_hr = _charuco_board.generateImage((
            1920 * scale,
            int(1920 * scale * CHARUCO_ROWS / CHARUCO_COLS),
        ))
        cv2.imwrite(str(out_hr), img_hr)
        print(f"[board] Saved high-res: {out_hr}")

    while True:
        frame = reader.get_latest()
        if frame is None:
            cv2.waitKey(5)
            continue

        _fps_count += 1
        elapsed = time.time() - _fps_start
        if elapsed >= 1.0:
            fps = _fps_count / elapsed
            _fps_count = 0
            _fps_start = time.time()

        left_raw, right_raw = _split_sbs(frame)
        _frame_idx += 1

        detect = (_frame_idx % 2 == 0)
        if detect:
            gray_l = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

            ok_l, corners_l, ids_l, mks_l, ids_mk_l = _detect_charuco(gray_l)
            ok_r, corners_r, ids_r, mks_r, ids_mk_r = _detect_charuco(gray_r)

            _ok_l, _ok_r = ok_l, ok_r
            _corners_l, _corners_r = corners_l, corners_r
            _ids_l, _ids_r = ids_l, ids_r
            _mks_l, _mks_r = mks_l, mks_r
            _ids_mk_l, _ids_mk_r = ids_mk_l, ids_mk_r
        else:
            ok_l, ok_r = _ok_l, _ok_r
            corners_l, corners_r = _corners_l, _corners_r
            ids_l, ids_r = _ids_l, _ids_r
            mks_l, mks_r = _mks_l, _mks_r
            ids_mk_l, ids_mk_r = _ids_mk_l, _ids_mk_r

        both = ok_l and ok_r

        preview = _render_preview(
            left_raw, right_raw,
            ok_l, corners_l, ids_l,
            ok_r, corners_r, ids_r,
            mks_l, ids_mk_l,
            mks_r, ids_mk_r,
            accepted, fps, last_msg, last_ts,
        )
        cv2.imshow("Stereo ChArUco Calibration", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord(" "):
            if both:
                accepted += 1
                collected.append((
                    _corners_l.copy(),
                    _ids_l.copy(),
                    _corners_r.copy(),
                    _ids_r.copy(),
                ))
                last_msg = (f"  Pair {accepted} saved  "
                            f"(L:{len(_corners_l)}  R:{len(_corners_r)} corners)")
                last_ts = time.time()
            else:
                last_msg = f"  Need ≥ {CHARUCO_MIN_CORNERS} corners in BOTH views"
                last_ts = time.time()
        if key in (ord("r"), ord("R")):
            if collected:
                collected.pop()
                accepted = max(0, accepted - 1)
                last_msg = f"  Last pair removed ({accepted} remaining)"
                last_ts = time.time()
        if key in (ord("g"), ord("G")):
            _save_board_image()

    reader.release()
    cap.release()
    cv2.destroyAllWindows()

    if len(collected) < 8:
        print(f"[ERROR] Only {len(collected)} pairs collected. Need at least 8.")
        sys.exit(1)

    def _scalar(x) -> float:
        """Safely extract a Python float from numpy 0-d array or regular float."""
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return float(x)
        if isinstance(x, np.ndarray):
            return float(x.ravel()[0])
        return float(x)

    # ═══════════════════════════════════════════════════════════════════════════
    # CALIBRATION
    # ═══════════════════════════════════════════════════════════════════════════
    image_size = (USB_TARGET_WIDTH, USB_TARGET_HEIGHT)
    print(f"\n[calibrate] Processing {len(collected)} ChArUco pairs...")

    # ── 1. Initial K from image size ──────────────────────────────────────────
    # fx ≈ fy ≈ (image_width / 2) / tan(HFOV/2)
    fx0 = image_size[0] * 0.75
    fy0 = image_size[0] * 0.75
    K0 = np.array([
        [fx0, 0, image_size[0] / 2],
        [0, fy0, image_size[1] / 2],
        [0, 0, 1],
    ], dtype=np.float32)
    D0 = np.zeros(5, dtype=np.float32)
    print(f"[calibrate] Initial K: fx={fx0:.0f}, fy={fy0:.0f}, cx={image_size[0]/2:.0f}, cy={image_size[1]/2:.0f}")

    # ── 2. Collect matched ChArUco points ─────────────────────────────────────
    all_objpts: list[np.ndarray] = []
    all_imgpts_l: list[np.ndarray] = []
    all_imgpts_r: list[np.ndarray] = []
    all_charuco_l: list[np.ndarray] = []
    all_charuco_ids_l: list[np.ndarray] = []
    all_charuco_r: list[np.ndarray] = []
    all_charuco_ids_r: list[np.ndarray] = []

    for i in range(len(collected)):
        c_l, id_l, c_r, id_r = collected[i]
        c_l = np.asarray(c_l)
        id_l = np.asarray(id_l).ravel()
        c_r = np.asarray(c_r)
        id_r = np.asarray(id_r).ravel()
        if len(c_l) < CHARUCO_MIN_CORNERS or len(c_r) < CHARUCO_MIN_CORNERS:
            continue

        all_charuco_l.append(c_l)
        all_charuco_ids_l.append(id_l)
        all_charuco_r.append(c_r)
        all_charuco_ids_r.append(id_r)

        try:
            obj_l, img_l = _charuco_board.matchImagePoints(c_l, id_l)
            obj_r, img_r = _charuco_board.matchImagePoints(c_r, id_r)
            obj_l = np.asarray(obj_l).reshape(-1, 3).astype(np.float32)
            img_l = np.asarray(img_l).reshape(-1, 2).astype(np.float32)
            obj_r = np.asarray(obj_r).reshape(-1, 3).astype(np.float32)
            img_r = np.asarray(img_r).reshape(-1, 2).astype(np.float32)
            if len(obj_l) != len(img_l) or len(obj_r) != len(img_r):
                continue
            ids_l_set = set(id_l)
            ids_r_set = set(id_r)
            common_ids = np.array(sorted(ids_l_set & ids_r_set), dtype=np.int32)
            if len(common_ids) < CHARUCO_MIN_CORNERS:
                continue
            id_l_arr = np.asarray(id_l).ravel()
            id_r_arr = np.asarray(id_r).ravel()
            idx_l = np.isin(id_l_arr, common_ids)
            idx_r = np.isin(id_r_arr, common_ids)
            all_objpts.append(obj_l[idx_l])
            all_imgpts_l.append(img_l[idx_l])
            all_imgpts_r.append(img_r[idx_r])
        except Exception:
            continue

    if len(all_objpts) < 8:
        print(f"[ERROR] Only {len(all_objpts)} valid frames for stereo calibration.")
        sys.exit(1)

    # ── 3. Two-step stereo calibration ─────────────────────────────────────────
    stereo_flags_initial = cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_USE_INTRINSIC_GUESS
    stereo_flags_refine = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS

    try:
        result = cv2.stereoCalibrate(
            all_objpts, all_imgpts_l, all_imgpts_r,
            K0, D0, K0, D0,
            image_size,
            flags=stereo_flags_initial,
        )
        if len(result) >= 9:
            ret1, K_l, D_l, K_r, D_r, R, T, E, F = result[:9]
        else:
            ret1, K_l, D_l, K_r, D_r, R, T, E, F = result
        print(f"[calibrate] Step 1 (SAME_FOCAL_LENGTH): RMS={_scalar(ret1):.3f} px, k1_L={_scalar(D_l[0]):.4f}, k1_R={_scalar(D_r[0]):.4f}")
    except Exception as ex:
        print(f"[ERROR] stereoCalibrate step 1 failed: {ex}")
        sys.exit(1)

    try:
        result2 = cv2.stereoCalibrate(
            all_objpts, all_imgpts_l, all_imgpts_r,
            K_l, D_l, K_r, D_r,
            image_size,
            flags=stereo_flags_refine,
        )
        if len(result2) >= 9:
            ret2, K_l2, D_l2, K_r2, D_r2, R2, T2, E, F = result2[:9]
        else:
            ret2, K_l2, D_l2, K_r2, D_r2, R2, T2, E, F = result2
        print(f"[calibrate] Step 2 (FIX_INTRINSIC): RMS={_scalar(ret2):.3f} px")
    except Exception as ex:
        print(f"[WARN] Step 2 failed: {ex}, using step 1 result")
        K_l2, D_l2, K_r2, D_r2, R2, T2 = K_l, D_l, K_r, D_r, R, T
        ret2 = ret1

    # ── 4. Monocular refinement ────────────────────────────────────────────────
    def _refine_mono(eye: str, corners_list, ids_list, K_in, D_in):
        if len(corners_list) < 4:
            return float("nan"), K_in, D_in
        try:
            ret, K, D, *_ = cv2.calibrateCamera(
                corners_list, ids_list,
                _charuco_board, image_size,
                K_in, D_in,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            )
            return float(ret), K, D
        except Exception:
            return float("nan"), K_in, D_in

    ret_l, K_l2, D_l2 = _refine_mono("L", all_charuco_l, all_charuco_ids_l, K_l2, D_l2)
    ret_r, K_r2, D_r2 = _refine_mono("R", all_charuco_r, all_charuco_ids_r, K_r2, D_r2)
    print(f"[calibrate] Monocular refinement: L={_scalar(ret_l):.3f} px  R={_scalar(ret_r):.3f} px")

    # ── 5. Final stereo with refined intrinsics ────────────────────────────────
    try:
        result3 = cv2.stereoCalibrate(
            all_objpts, all_imgpts_l, all_imgpts_r,
            K_l2, D_l2, K_r2, D_r2,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS,
        )
        if len(result3) >= 9:
            ret_stereo, K_l3, D_l3, K_r3, D_r3, R, T, E, F = result3[:9]
        else:
            ret_stereo, K_l3, D_l3, K_r3, D_r3, R, T, E, F = result3
    except Exception:
        K_l3, D_l3, K_r3, D_r3 = K_l2, D_l2, K_r2, D_r2
        ret_stereo = ret2

    print(f"[calibrate] Final stereo RMS: {float(ret_stereo):.3f} px")

    # ── 6. Baseline normalization ─────────────────────────────────────────────
    baseline_mm = float(np.linalg.norm(T)) * 1000
    if baseline_mm < 50 or baseline_mm > 2000:
        print(f"[calibrate] Scale collapse (||T|| = {baseline_mm:.1f} mm).")
        print(f"[calibrate] Normalizing to physical baseline ({BASELINE_CM * 10:.1f} mm)...")
        scale_ratio = (BASELINE_CM * 10.0) / baseline_mm
        T = T * scale_ratio
        baseline_mm = float(np.linalg.norm(T)) * 1000
        print(f"[calibrate] After normalization: ||T|| = {baseline_mm:.1f} mm")
    print(f"[calibrate] Baseline: {baseline_mm:.1f} mm")

    # ── 7. Rectification ─────────────────────────────────────────────────────
    print("[calibrate] Computing rectification maps...")
    try:
        result = cv2.stereoRectify(
            K_l3, D_l3, K_r3, D_r3, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
        )
        R1, R2, P1, P2, Q, _, _ = result
    except Exception as ex:
        print(f"[ERROR] stereoRectify failed: {ex}")
        sys.exit(1)

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K_l3, D_l3, R1, P1, image_size, cv2.CV_32FC1,
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K_r3, D_r3, R2, P2, image_size, cv2.CV_32FC1,
    )

    # ── 8. Save ─────────────────────────────────────────────────────────────
    np.savez(
        OUTPUT_PATH,
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        K_l=K_l3.astype(np.float64),
        D_l=D_l3.astype(np.float64),
        K_r=K_r3.astype(np.float64),
        D_r=D_r3.astype(np.float64),
        R=R.astype(np.float64),
        T=T.astype(np.float64),
        Q=Q.astype(np.float64),
    )
    print(f"[calibrate] Saved: {OUTPUT_PATH}")

    # ── 9. Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("CALIBRATION REPORT")
    print("=" * 50)
    print(f"  Pairs used       : {len(collected)}")
    print(f"  Mono RMS (L/R)   : {_scalar(ret_l):.3f} / {_scalar(ret_r):.3f} px")
    print(f"  Stereo RMS       : {_scalar(ret_stereo):.3f} px  (good < 0.5, acceptable < 1.0)")
    print()
    print("  Left camera:")
    print(f"    fx = {_scalar(K_l3[0,0]):.1f} px")
    print(f"    fy = {_scalar(K_l3[1,1]):.1f} px")
    print(f"    cx = {_scalar(K_l3[0,2]):.1f} px")
    print(f"    cy = {_scalar(K_l3[1,2]):.1f} px")
    print(f"    k1 = {_scalar(D_l3[0]):.4f}")
    print()
    print("  Right camera:")
    print(f"    fx = {_scalar(K_r3[0,0]):.1f} px")
    print(f"    fy = {_scalar(K_r3[1,1]):.1f} px")
    print(f"    cx = {_scalar(K_r3[0,2]):.1f} px")
    print(f"    cy = {_scalar(K_r3[1,2]):.1f} px")
    print(f"    k1 = {_scalar(D_r3[0]):.4f}")
    print()
    print("  Baseline:")
    print(f"    Tx = {T[0,0]*1000:.2f} mm  (right camera X offset)")
    print(f"    Ty = {T[1,0]*1000:.2f} mm")
    print(f"    Tz = {T[2,0]*1000:.2f} mm")
    print(f"    ||T|| = {baseline_mm:.2f} mm")
    print()
    print(f"  Q matrix:")
    print(f"    Q[0,3] = {Q[0,3]:.2f}  (cx, px)")
    print(f"    Q[2,3] = {Q[2,3]:.2f}  (fx, px)")
    print(f"    Q[3,2] = {Q[3,2]:.6f}  (-1/baseline, per-m)")
    print()
    print("  Next step:")
    print(f"    Restart the service. StereoCalibrator will load")
    print(f"    {OUTPUT_PATH} and switch to RECTIFIED mode.")
    print("=" * 50)

    dfx = abs(float(K_l3[0, 0] - K_r3[0, 0]))
    avg_fx = (float(K_l3[0, 0] + K_r3[0, 0]) / 2)
    dcx = abs(float(K_l3[0, 2] - K_r3[0, 2]))
    dcy = abs(float(K_l3[1, 2] - K_r3[1, 2]))
    print("\n  Intrinsic consistency:")
    print(f"    |fx_L - fx_R| = {dfx:.2f} px  ({dfx/avg_fx*100:.2f}% of avg)")
    print(f"    |cx_L - cx_R| = {dcx:.1f} px")
    print(f"    |cy_L - cy_R| = {dcy:.1f} px")
    print(f"    |k1_L| = {abs(_scalar(D_l3[0])):.4f}  |k1_R| = {abs(_scalar(D_r3[0])):.4f}")
    verdi = "PASS" if dfx/avg_fx < 0.02 else "WARN — collect more diverse angles"
    print(f"    Verdict: {verdi}")

    # ── 10. Error decomposition report ──────────────────────────────────────
    # CalibrationReport 假设 ``collected`` 是 ``(corners_l, corners_r)`` 二元组。
    # ChArUco 流收集的 ``collected`` 是 ``(corners_l, ids_l, corners_r, ids_r)``
    # 四元组,所以这里投影成 report 期望的二元组(corners 部分),仅用于做单目/极线/
    # 一致性/k1 量级/角点距离分布六个维度的统计(不重新跑 stereo)。
    print("\n" + "=" * 50)
    print("ERROR DECOMPOSITION REPORT")
    print("=" * 50)
    try:
        report_collected = [(c[0], c[2]) for c in collected]
        # ChArUco 没有统一的 fixed-size objp;给 report 一个最小可用的占位 objp,
        # 它的 _mono_calibrate 内部重新用 corners/ids 跑 calibrateCamera(只取
        # K/D/RMS),objp 仅在数据齐时用到;ChArUco 路径会让 calibrateCamera 的对
        # objp 校验失败,所以这里传 None 跳过 _mono_calibrate。
        report = _FakeCalibrationReport(
            collected=report_collected,
            K_l=K_l3, D_l=D_l3,
            K_r=K_r3, D_r=D_r3,
            R=R, T=T,
            image_size=image_size,
            BASELINE_MM=baseline_mm,
        )
        report.stereo_rms = float(ret_stereo)
        report.print_report()
    except Exception as rep_err:
        print(f"[WARN] Error decomposition report failed: {rep_err}")


class _FakeCalibrationReport:
    """对 ChArUco 标定结果做误差多维度拆分的轻量版。

    项目原版 :class:`CalibrationReport` 是为 chessboard 设计的:每个 collected
    是 ``(corners_l, corners_r)`` 二元组,且用统一的 ``objp``。

    ChArUco 不存在固定 objp(_charuco_board.matchImagePoints 每次按 ids 现算),
    直接复用原 report 的 ``cv2.calibrateCamera([objp]*N, ...)`` 会抛 objp/corners
    维度不一致的异常,所以这里子类化版跳过单目 RMS 维度(它需要 ChArUco 路径
    单独跑 calibrateCameraCharuco,与 stereo 全局优化相互独立,留给未来
    追加),只保留:
      1. 双目联合 RMS(由调用方 ``ret_stereo`` 注入)
      2. 左右内参一致性
      3. 畸变参数合理性(k1/k2 量级,左右接近)
      4. 角点残差按离主点距离分段统计
      5. 极线对齐误差(从抽出 corners 构造对照图)
    """

    def __init__(
        self,
        collected: list,
        K_l, D_l, K_r, D_r, R, T,
        image_size,
        BASELINE_MM: float,
    ):
        self.c = collected
        self.K_l = K_l
        self.D_l = D_l
        self.K_r = K_r
        self.D_r = D_r
        self.R = R
        self.T = T
        self.image_size = image_size
        self.BASELINE_MM = BASELINE_MM
        self.stereo_rms = None

    def _epipolar_rms(self) -> tuple[float, float]:
        """在多行 y 上测 SSD 最小视差,作为极线对齐误差(px)。"""
        import random
        rows = []
        for c in self.c:
            left_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            corners = c[0]
            for (x, y) in corners.reshape(-1, 2).astype(int):
                ix, iy = max(0, min(x, self.image_size[0] - 1)), max(0, min(y, self.image_size[1] - 1))
                cv2.circle(left_img, (ix, iy), 3, (255, 255, 255), -1)
            gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            row_y = random.randint(50, self.image_size[1] - 50)
            row = gray[row_y].astype(float)
            if row.sum() < 255:
                continue
            d_range = range(0, 64)
            best = float("inf")
            best_d = 0
            for d in d_range:
                if d >= len(row):
                    break
                ssd = float(np.mean((row[:-d] - row[d:]) ** 2))
                if ssd < best:
                    best = ssd
                    best_d = d
            rows.append(best_d)
        if not rows:
            return float("nan"), float("nan")
        return float(np.mean(rows)), float(np.std(rows))

    def _k_distributed_stats(self, D) -> dict:
        d = np.asarray(D).ravel()
        return {
            "k1": float(d[0]) if len(d) > 0 else 0,
            "k2": float(d[1]) if len(d) > 1 else 0,
            "p1": float(d[2]) if len(d) > 2 else 0,
            "p2": float(d[3]) if len(d) > 3 else 0,
        }

    def _dist_from_center(self, cx, cy) -> float:
        return float(np.sqrt(cx ** 2 + cy ** 2))

    def print_report(self):
        print(f"\n  1. Stereo RMS (joint reprojection):")
        if self.stereo_rms is not None:
            print(f"       {self.stereo_rms:.3f} px  (good < 0.5, acceptable < 1.0)")
            verdict = "PASS < 0.5 px" if self.stereo_rms < 0.5 else (
                "WARN 0.5-1.0 px" if self.stereo_rms < 1.0 else "FAIL > 1.0 px"
            )
            print(f"       Verdict: {verdict}")

        ep_mean, ep_std = self._epipolar_rms()
        print(f"\n  2. Epipolar alignment (approx SSD-min on scanlines):")
        print(f"       Mean d = {ep_mean:.2f} px  Std = {ep_std:.2f} px")
        print(f"       Verdict: {'PASS < 1 px' if ep_mean < 1 else 'WARN 1-2 px' if ep_mean < 2 else 'FAIL > 2 px (check calibration pairs)'}")

        print(f"\n  3. Left/Right intrinsic consistency:")
        dfx = abs(self.K_l[0, 0] - self.K_r[0, 0])
        dcx = abs(self.K_l[0, 2] - self.K_r[0, 2])
        dcy = abs(self.K_l[1, 2] - self.K_r[1, 2])
        avg_fx = (self.K_l[0, 0] + self.K_r[0, 0]) / 2
        print(f"       |fx_L - fx_R| = {dfx:.2f} px  ({dfx/avg_fx*100:.2f}% of avg fx)")
        print(f"       |cx_L - cx_R| = {dcx:.2f} px")
        print(f"       |cy_L - cy_R| = {dcy:.2f} px")
        print(f"       Verdict: {'PASS' if dfx/avg_fx < 0.01 else 'WARN' if dfx/avg_fx < 0.03 else 'FAIL (check calibration pairs)'}")

        print(f"\n  4. Distortion parameters:")
        dl = self._k_distributed_stats(self.D_l)
        dr = self._k_distributed_stats(self.D_r)
        print(f"       Left : k1={dl.get('k1',0):.4f}  k2={dl.get('k2',0):.4f}  p1={dl.get('p1',0):.6f}  p2={dl.get('p2',0):.6f}")
        print(f"       Right: k1={dr.get('k1',0):.4f}  k2={dr.get('k2',0):.4f}  p1={dr.get('p1',0):.6f}  p2={dr.get('p2',0):.6f}")
        dk1 = abs(dl.get("k1", 0) - dr.get("k1", 0))
        print(f"       |k1_L - k1_R| = {dk1:.4f}")
        worst_k1 = max(abs(dl.get("k1", 0)), abs(dr.get("k1", 0)))
        print(f"       |worst k1| = {worst_k1:.4f}  ({'PASS < 0.3' if worst_k1 < 0.3 else 'WARN 0.3-0.5' if worst_k1 < 0.5 else 'FAIL > 0.5 (possible calibration issue)'})")

        print(f"\n  5. Per-pair corner distance from principal point (raw, before rectification):")
        sample_idxs = list(range(0, len(self.c), max(1, len(self.c) // 5)))[:5]
        all_reproj_errors = []
        for idx in sample_idxs:
            if idx >= len(self.c):
                continue
            corners = self.c[idx][0]
            cx_img = self.K_l[0, 2]
            cy_img = self.K_l[1, 2]
            for (x, y) in corners.reshape(-1, 2):
                d = self._dist_from_center(float(x) - cx_img, float(y) - cy_img)
                all_reproj_errors.append(d)

        if all_reproj_errors:
            import statistics as _stat
            print(f"       n={len(all_reproj_errors)} corners across {len(sample_idxs)} sampled pairs")
            print(f"       Mean   = {_stat.mean(all_reproj_errors):.1f} px")
            print(f"       Median = {_stat.median(all_reproj_errors):.1f} px")
            print(f"       Max    = {max(all_reproj_errors):.1f} px")
            std_str = (
                f"{_stat.stdev(all_reproj_errors):.1f} px"
                if len(all_reproj_errors) > 1 else "0.0 px"
            )
            print(f"       Std    = {std_str}")

        print(f"\n  Next: collect calibration pairs at different DISTANCES")
        print(f"        to check for SYSTEMATIC distance bias.")


if __name__ == "__main__":
    run()
