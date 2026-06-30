"""Stereo checkerboard calibration capture tool.

Usage:
    python scripts/calibrate_stereo.py

Prerequisites:
    - Your checkerboard: 10x7 grid (inner corners 9x6), square size = 20 mm
    - Print at 100% scale, affix to a flat rigid board
    - Keep the board visible in BOTH left and right views simultaneously

Output:
    config/calibration/calib.npz  (auto-created)

How many pairs?
    - Minimum: 20 pairs covering diverse tilts and distances
    - More pairs = better calibration (up to ~40)
    - Cover: near/far, tilted left/right/up/down, centered/edges
    - Press SPACE to accept a pair, press R to reset/cancel

After calibration:
    1. The npz is saved automatically
    2. Restart the main service; it will load the npz and switch from
       UNRECTIFIED to RECTIFIED mode
    3. Use /api/stream/stats or perf page to check that:
         - rectified=True
         - epipolar_error < 1.0 px (ideally < 0.5)
         - focal_px matches your hardware (~1140-1200 px)
"""
from __future__ import annotations

import os
import sys
import json
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── hardware reality source ──────────────────────────────────────────────────
# Import from project config so we read actual camera index / resolution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    BASELINE_CM,
    CALIB_NPZ_PATH,
    USB_LEFT_INDEX,
    USB_TARGET_WIDTH,
    USB_TARGET_HEIGHT,
    USB_FPS,
)

# ── checkerboard parameters ──────────────────────────────────────────────────
BOARD_W = 9          # inner corners horizontally  (10 squares - 1)
BOARD_H = 6          # inner corners vertically    (7  squares - 1)
# Your board: 10 cols × 7 rows of squares = 9×6 inner corners.
# Measured: 13.4 cm total for 7 squares → 134 mm / 7 ≈ 19.14 mm per square.
SQUARE_SIZE_MM = 19.14  # MUST match printed board exactly (use digital caliper, 1:1 print)

# ── calibration output ───────────────────────────────────────────────────────
OUTPUT_PATH: Path = CALIB_NPZ_PATH
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── camera backend ───────────────────────────────────────────────────────────
BACKENDS = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF, "MSMF"),
]


def _open_camera(index: int, width: int, height: int, fps: int):
    """Try multiple backends; return (cap, backend_name)."""
    for backend, name in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        time.sleep(0.5)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[camera] index={index} opened with {name}, frame shape={frame.shape}")
            return cap, name

        cap.release()

    return None, "none"


def _split_sbs(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 3840x1080 SBS frame into left and right eyes."""
    eye_w = frame.shape[1] // 2
    return frame[:, :eye_w], frame[:, eye_w:]


class FrameReader:
    """Background thread that continuously reads frames from the camera.

    The main thread calls get_latest() to get the most recent frame,
    discarding any stale frames in the queue.  This decouples camera I/O
    (which blocks on USB/decode) from the UI loop.
    """

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
        """Return the newest available frame, or None if the queue is empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[-1]

    def release(self):
        self._stopped = True
        self._thread.join(timeout=2.0)


def _draw_corners(
    canvas: np.ndarray,
    corners: np.ndarray,
    ok: bool,
    label: str,
) -> None:
    """Draw detected corners on a canvas copy."""
    display = canvas.copy()
    if ok:
        cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners, ok)
        color = (0, 255, 0)
        info = f"{label}: OK ({len(corners)} corners)"
    else:
        color = (0, 0, 255)
        info = f"{label}: not detected"

    cv2.putText(display, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return display


def _make_grid(corners_l: np.ndarray, corners_r: np.ndarray) -> np.ndarray:
    """Arrange L/R corner images side by side with labels."""
    dl = _draw_corners(np.zeros((540, 960, 3), dtype=np.uint8),
                       corners_l, True, "L")
    dr = _draw_corners(np.zeros((540, 960, 3), dtype=np.uint8),
                       corners_r, True, "R")
    return cv2.hconcat([dl, dr])


def run():
    # ── open camera ───────────────────────────────────────────────────────────
    cap, name = _open_camera(
        USB_LEFT_INDEX,
        USB_TARGET_WIDTH * 2,   # SBS total width
        USB_TARGET_HEIGHT,
        USB_FPS,
    )
    if cap is None:
        print("[ERROR] Could not open camera. Check USB index in config.")
        print(f"  Current left_index={USB_LEFT_INDEX}, width={USB_TARGET_WIDTH*2}, height={USB_TARGET_HEIGHT}")
        sys.exit(1)

    # Start background reader so cap.read() never blocks the UI thread
    reader = FrameReader(cap, queue_size=2)
    print(f"[camera] Background reader started (DSHOW capture)")

    # ── prepare object points (same for all views) ────────────────────────────
    objp = np.zeros((BOARD_W * BOARD_H, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # in mm

    collected: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    accepted = 0
    last_msg = ""
    last_ts = time.time()

    # Let the background reader fill the queue before entering the loop
    time.sleep(0.3)

    print("\n" + "=" * 60)
    print("STEREO CALIBRATION CAPTURE")
    print("=" * 60)
    print(f"  Checkerboard : {BOARD_W}x{BOARD_H} inner corners")
    print(f"  Square size   : {SQUARE_SIZE_MM} mm")
    print(f"  Target pairs  : 20-40 (more is better)")
    print(f"  Output        : {OUTPUT_PATH}")
    print()
    print("  SPACE   = accept current pair")
    print("  R       = reset last pair")
    print("  ESC / Q = finish & calibrate")
    print("=" * 60 + "\n")

    # Pre-create calibration criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # Cache last detection so preview is still visible between detections
    _ok_l = False
    _ok_r = False
    _corners_l: np.ndarray | None = None
    _corners_r: np.ndarray | None = None
    _frame_idx = 0

    # Preview cache (only recompute on detection frames to reduce CPU load)
    _cached_preview: np.ndarray | None = None

    # FPS tracking
    _fps_start = time.time()
    _fps_count = 0
    fps = 0.0

    while True:
        # Non-blocking: get latest frame from background thread
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

        # Detect corners every 3 frames (expensive; preview still updates every frame)
        detect = (_frame_idx % 3 == 1)
        if detect:
            ok_l, corners_l = cv2.findChessboardCorners(left_raw, (BOARD_W, BOARD_H), None)
            ok_r, corners_r = cv2.findChessboardCorners(right_raw, (BOARD_W, BOARD_H), None)

            # Refine corners
            if ok_l and corners_l is not None and len(corners_l) > 0:
                gray_l = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), criteria)
            if ok_r and corners_r is not None and len(corners_r) > 0:
                gray_r = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), criteria)

            _ok_l, _ok_r = ok_l, ok_r
            _corners_l, _corners_r = corners_l, corners_r
        else:
            ok_l, ok_r = _ok_l, _ok_r
            corners_l, corners_r = _corners_l, _corners_r

        both = ok_l and ok_r

        # Rebuild preview only on detection frames to reduce CPU load
        if detect or _cached_preview is None:
            preview_l = cv2.resize(left_raw, (960, 540))
            preview_r = cv2.resize(right_raw, (960, 540))

            sx_l, sy_l = 960 / left_raw.shape[1], 540 / left_raw.shape[0]
            sx_r, sy_r = 960 / right_raw.shape[1], 540 / right_raw.shape[0]

            n_corners = BOARD_W * BOARD_H
            cl_ok = bool(ok_l) and (corners_l is not None) and (corners_l.size > 0) and (corners_l.shape[0] == n_corners)
            cr_ok = bool(ok_r) and (corners_r is not None) and (corners_r.size > 0) and (corners_r.shape[0] == n_corners)

            if cl_ok:
                scaled = (corners_l.reshape(-1, 2) * np.array([sx_l, sy_l], dtype=np.float64)).astype(np.float32)
                preview_l = cv2.drawChessboardCorners(preview_l, (BOARD_W, BOARD_H), scaled, True)
            if cr_ok:
                scaled_r = (corners_r.reshape(-1, 2) * np.array([sx_r, sy_r], dtype=np.float64)).astype(np.float32)
                preview_r = cv2.drawChessboardCorners(preview_r, (BOARD_W, BOARD_H), scaled_r, True)

            both_preview = np.hstack([preview_l, preview_r])   # 1920 x 540

            # Status bar with dark background to avoid flicker
            status = np.full((90, 1920, 3), 30, dtype=np.uint8)
            if elapsed >= 1.0:
                cv2.putText(status, f"{fps:.1f} fps", (1820, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)

            cv2.putText(status, f"Accepted pairs: {accepted}/20+  |  "
                        f"L:{'OK' if ok_l else '---'}  R:{'OK' if ok_r else '---'}  |  "
                        f"{'BOTH DETECTED - press SPACE' if both else 'Move board to see both views'}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            if last_msg and (time.time() - last_ts) < 3:
                cv2.putText(status, last_msg, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

            cv2.putText(status, "SPACE=accept  R=reset  ESC=calibrate",
                        (10, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            _cached_preview = np.vstack([both_preview, status])

        cv2.imshow("Stereo Calibration", _cached_preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord(" "):
            corners_valid = (
                corners_l is not None and len(corners_l) > 0
                and corners_r is not None and len(corners_r) > 0
            )
            if both and corners_valid:
                accepted += 1
                collected.append((corners_l.copy(), corners_r.copy()))
                last_msg = f"  Pair {accepted} saved!"
                last_ts = time.time()
                # #region agent log H-B — save the very first accepted raw split
                # Hypothesis B/C: is _split_sbs() returning the right eyes?  Dump
                # the raw left_raw / right_raw from the last detect frame so the
                # user can open them and visually confirm which is which.
                if accepted == 1:
                    try:
                        dbg_dir = os.path.join(os.path.dirname(__file__), "..", "config", "calibration", "_debug")
                        dbg_dir = os.path.normpath(dbg_dir)
                        os.makedirs(dbg_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(dbg_dir, "first_left.png"), left_raw)
                        cv2.imwrite(os.path.join(dbg_dir, "first_right.png"), right_raw)
                        cv2.imwrite(os.path.join(dbg_dir, "first_sbs.png"), np.hstack([left_raw, right_raw]))
                        with open(r"e:\Remote_HCR\Stereo_Vision-main\debug-c7ffa8.log", "a", encoding="utf-8") as _flog:
                            _flog.write(json.dumps({
                                "id": f"log_dbg_{int(time.time()*1000)}_firstsplit",
                                "sessionId": "c7ffa8",
                                "timestamp": int(time.time() * 1000),
                                "location": "calibrate_stereo.py:325",
                                "message": "first pair raw split saved",
                                "data": {
                                    "saved_dir": dbg_dir,
                                    "left_shape": list(left_raw.shape),
                                    "right_shape": list(right_raw.shape),
                                },
                                "runId": "pre-fix",
                                "hypothesisId": "B"
                            }, ensure_ascii=False) + "\n")
                    except Exception as _e:
                        with open(r"e:\Remote_HCR\Stereo_Vision-main\debug-c7ffa8.log", "a", encoding="utf-8") as _flog:
                            _flog.write(json.dumps({
                                "id": f"log_dbg_{int(time.time()*1000)}_firstsplit_err",
                                "sessionId": "c7ffa8",
                                "timestamp": int(time.time() * 1000),
                                "location": "calibrate_stereo.py:325",
                                "message": f"first-split save failed: {_e}",
                                "data": {},
                                "runId": "pre-fix",
                                "hypothesisId": "B"
                            }, ensure_ascii=False) + "\n")
                # #endregion agent log H-B
            else:
                last_msg = "  Must detect BOTH L and R first!"
                last_ts = time.time()
        if key in (ord("r"), ord("R")):
            if collected:
                collected.pop()
                accepted -= 1
                last_msg = f"  Last pair removed ({accepted} remaining)"
                last_ts = time.time()

    reader.release()
    cap.release()
    cv2.destroyAllWindows()

    if len(collected) < 10:
        print(f"[ERROR] Only {len(collected)} pairs collected. Need at least 10.")
        sys.exit(1)

    # ── calibrate ─────────────────────────────────────────────────────────────
    print(f"\n[calibrate] Running stereoCalibrate with {len(collected)} pairs...")

    imgpoints_l: list[np.ndarray] = [c[0] for c in collected]
    imgpoints_r: list[np.ndarray] = [c[1] for c in collected]

    image_size = (USB_TARGET_WIDTH, USB_TARGET_HEIGHT)

    # #region agent log H-A/D — monocular pre-flight + save first split frame
    try:
        # Save first pair's split frames to disk for visual inspection (Hypothesis B:
        # is _split_sbs() actually returning left/right correctly?  Some DSHOW drivers
        # emit 1920x2160 vertical-stacked instead of 3840x1080 SBS, which would put
        # the bottom-half of one eye next to the top-half of the other.)
        dbg_dir = os.path.join(os.path.dirname(__file__), "..", "config", "calibration", "_debug")
        dbg_dir = os.path.normpath(dbg_dir)
        os.makedirs(dbg_dir, exist_ok=True)
        _dbg_l, _dbg_r = collected[0][0], collected[0][1]  # not the raw images, but
        # we also dump the raw first-pair frame from the most recent detect cycle.
        # The raw frame is more useful; fall back to drawn corners if needed.
        with open(r"e:\Remote_HCR\Stereo_Vision-main\debug-c7ffa8.log", "a", encoding="utf-8") as _flog:
            _flog.write(json.dumps({
                "id": f"log_dbg_{int(time.time()*1000)}_preflight",
                "sessionId": "c7ffa8",
                "timestamp": int(time.time() * 1000),
                "location": "calibrate_stereo.py:343",
                "message": "preflight check",
                "data": {
                    "n_pairs": len(collected),
                    "img_size": list(image_size),
                    "first_pair_corners_shape": [list(_dbg_l.shape), list(_dbg_r.shape)],
                },
                "runId": "pre-fix",
                "hypothesisId": "A"
            }, ensure_ascii=False) + "\n")

        # Per-eye monocular pre-flight (Hypothesis A):  is the high RMS coming from
        # per-eye calibration difficulty, or purely from the cross-eye R/T search?
        # If left/right monocular RMS are each < 1 px, then the problem is cross-eye
        # pairing (sample diversity, board coverage).  If either is > 1 px, the
        # problem is image quality (focus, exposure, reflection).
        W, H = image_size
        flags_mono = 0  # no constraints, let it fit freely
        # Use a single objp instance (we already have it above)
        _ret_l, _K_l, _D_l, _rvecs_l, _tvecs_l = cv2.calibrateCamera(
            [objp] * len(collected), imgpoints_l, image_size, None, None, flags=flags_mono)
        _ret_r, _K_r, _D_r, _rvecs_r, _tvecs_r = cv2.calibrateCamera(
            [objp] * len(collected), imgpoints_r, image_size, None, None, flags=flags_mono)
        with open(r"e:\Remote_HCR\Stereo_Vision-main\debug-c7ffa8.log", "a", encoding="utf-8") as _flog:
            _flog.write(json.dumps({
                "id": f"log_dbg_{int(time.time()*1000)}_mono",
                "sessionId": "c7ffa8",
                "timestamp": int(time.time() * 1000),
                "location": "calibrate_stereo.py:370",
                "message": "monocular preflight",
                "data": {
                    "rms_left_mono": float(_ret_l),
                    "rms_right_mono": float(_ret_r),
                    "k1_left_mono": float(_D_l.ravel()[0]) if _D_l.size else 0.0,
                    "k1_right_mono": float(_D_r.ravel()[0]) if _D_r.size else 0.0,
                    "fx_left_mono": float(_K_l[0, 0]),
                    "fx_right_mono": float(_K_r[0, 0]),
                    "cx_left_mono": float(_K_l[0, 2]),
                    "cx_right_mono": float(_K_r[0, 2]),
                },
                "runId": "pre-fix",
                "hypothesisId": "A"
            }, ensure_ascii=False) + "\n")
    except Exception as _e:
        with open(r"e:\Remote_HCR\Stereo_Vision-main\debug-c7ffa8.log", "a", encoding="utf-8") as _flog:
            _flog.write(json.dumps({
                "id": f"log_dbg_{int(time.time()*1000)}_preflight_err",
                "sessionId": "c7ffa8",
                "timestamp": int(time.time() * 1000),
                "location": "calibrate_stereo.py:343",
                "message": f"preflight failed: {_e}",
                "data": {},
                "runId": "pre-fix",
                "hypothesisId": "A"
            }, ensure_ascii=False) + "\n")
    # #endregion agent log H-A/D

    # Estimate reasonable initial K (focal ~70% of larger image dimension, centered)
    W, H = image_size
    f0 = 0.7 * max(W, H)
    K0_l = np.array([[f0, 0, W / 2],
                      [0, f0, H / 2],
                      [0,  0, 1]], dtype=np.float64)
    K0_r = K0_l.copy()
    D0 = np.zeros((1, 1), dtype=np.float64)

    try:
        ret, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            [objp] * len(collected),
            imgpoints_l,
            imgpoints_r,
            K0_l, D0, K0_r, D0,
            image_size,
            # No extra flags: let OpenCV freely optimize fx, fy, cx, cy, k1, k2, p1, p2
            # for both cameras independently.  Previous flags (SAME_FOCAL_LENGTH,
            # FIX_PRINCIPAL_POINT) over-constrained the fit and inflated RMS by 2-4 px.
            flags=0,
        )
        print(f"[calibrate] RMS re-projection error: {ret:.3f} px")
        if ret > 1.0:
            print(f"[WARN] High RMS ({ret:.1f} px). Tips to reduce:")
            print(f"  - Lock camera exposure / white-balance before collecting (auto-exposure")
            print(f"    causes brightness shifts between frames, the #1 cause of high RMS)")
            print(f"  - Collect more pairs (30-40), cover all 4 corners and edges")
            print(f"  - Tilt/rotate the board in various angles, vary distance")
            print(f"  - Ensure board is flat and corners are sharp / not blurry")
            print(f"  - If RMS stays > 2 despite good samples, re-measure SQUARE_SIZE_MM")
            print(f"    with a digital caliper (e.g. 19.14 mm, use more decimal places)")
    except Exception as ex:
        print(f"[ERROR] stereoCalibrate failed: {ex}")
        sys.exit(1)

    # Validate returned matrices
    for name, m, expected_shape in [
        ("K_l", K_l, (3, 3)), ("K_r", K_r, (3, 3)),
        ("R",   R,   (3, 3)), ("T",   T,   (3, 1)),
    ]:
        if m is None or np.array(m).shape != expected_shape:
            print(f"[ERROR] {name} has invalid shape {np.array(m).shape if m is not None else None}, expected {expected_shape}")
            sys.exit(1)

    # Validate physical baseline (expect 50-500 mm for most stereo rigs)
    baseline_mm = float(np.linalg.norm(T)) * 1000  # T from stereoCalibrate is in meters
    if baseline_mm < 50 or baseline_mm > 2000:
        print(f"[calibrate] Scale collapse detected (||T|| = {baseline_mm:.1f} mm).")
        print(f"[calibrate] Normalizing T to physical baseline ({BASELINE_CM * 10:.1f} mm)...")
        # T has collapsed because objp scale (SQUARE_SIZE_MM) is slightly wrong.
        # We scale T to the correct physical baseline.  K stays as-is: the pixel focal
        # length is independent of T-scale and the remap maps only depend on K ratios.
        # Depth-from-disparity uses hardware.py z_cm_from_disparity() which reads
        # BASELINE_CM directly and does NOT depend on this T vector.
        scale_ratio = (BASELINE_CM * 10.0) / baseline_mm  # e.g. 60 / 62000
        T = T * scale_ratio
        baseline_mm = float(np.linalg.norm(T)) * 1000
        print(f"[calibrate] After normalization: ||T|| = {baseline_mm:.1f} mm")
    print(f"[calibrate] Baseline: {baseline_mm:.1f} mm")

    # ── rectify ────────────────────────────────────────────────────────────────
    print("[calibrate] Computing rectification maps...")
    try:
        result = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
        )
        # OpenCV 4.x returns 7 values: R1(3,3), R2(3,3), P1(3,4), P2(3,4), Q(4,4), validRoi1(4,), validRoi2(4,)
        if len(result) == 7:
            R1, R2, P1, P2, Q, validRoi1, validRoi2 = result
        else:
            R1, R2, P1, P2, Q, _, _ = result
        for name, m, expected_shape in [
            ("R1", R1, (3, 3)), ("R2", R2, (3, 3)),
            ("P1", P1, (3, 4)), ("P2", P2, (3, 4)),
        ]:
            if m is None or np.array(m).shape != expected_shape:
                print(f"[ERROR] {name} has invalid shape {np.array(m).shape if m is not None else None}, expected {expected_shape}")
                sys.exit(1)
    except Exception as ex:
        print(f"[ERROR] stereoRectify failed: {ex}")
        sys.exit(1)

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K_l, D_l, R1, P1, image_size, cv2.CV_32FC1,
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K_r, D_r, R2, P2, image_size, cv2.CV_32FC1,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    np.savez(
        OUTPUT_PATH,
        # Remap tables (primary path for StereoCalibrator)
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        # Full intrinsic / extrinsic data (secondary path)
        K_l=K_l.astype(np.float64),
        D_l=D_l.astype(np.float64),
        K_r=K_r.astype(np.float64),
        D_r=D_r.astype(np.float64),
        R=R.astype(np.float64),
        T=T.astype(np.float64),
        # Reprojection matrix (for Q-based depth computation)
        Q=Q.astype(np.float64),
    )
    print(f"[calibrate] Saved: {OUTPUT_PATH}")

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("CALIBRATION REPORT")
    print("=" * 50)
    print(f"  Pairs used     : {len(collected)}")
    print(f"  RMS error      : {ret:.3f} px  (good if < 0.5, acceptable if < 1.0)")
    print()
    print("  Left camera:")
    print(f"    fx = {K_l[0,0]:.1f} px")
    print(f"    fy = {K_l[1,1]:.1f} px")
    print(f"    cx = {K_l[0,2]:.1f} px")
    print(f"    cy = {K_l[1,2]:.1f} px")
    print(f"    k1 = {D_l[0,0]:.4f}")
    _coef_names = ["k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    for i, name in enumerate(_coef_names, start=2):
        if D_l.shape[1] > i - 1:
            print(f"    {name} = {D_l[0, i]:.4f}")
    print()
    print("  Right camera:")
    print(f"    fx = {K_r[0,0]:.1f} px")
    print(f"    fy = {K_r[1,1]:.1f} px")
    print(f"    cx = {K_r[0,2]:.1f} px")
    print(f"    cy = {K_r[1,2]:.1f} px")
    print(f"    k1 = {D_r[0,0]:.4f}")
    for i, name in enumerate(_coef_names, start=2):
        if D_r.shape[1] > i - 1:
            print(f"    {name} = {D_r[0, i]:.4f}")
    print()
    print("  Baseline (T vector):")
    print(f"    Tx = {T[0,0]*1000:.2f} mm  (right camera X offset)")
    print(f"    Ty = {T[1,0]*1000:.2f} mm")
    print(f"    Tz = {T[2,0]*1000:.2f} mm")
    print(f"    ||T|| = {np.linalg.norm(T)*1000:.2f} mm (physical baseline)")
    # Note: stereoCalibrate returns T in meters (same unit as objp world coords).
    # objp uses mm (SQUARE_SIZE_MM), so T[0,0] ≈ -0.062 m ≈ -62 mm for a 6 cm rig.
    # If ||T|| >> 1000 mm here, the objp scale is wrong (e.g. wrong SQUARE_SIZE_MM)
    # and scale collapse occurred — the T-normalization block above handles it.
    print()
    print(f"  Q matrix (for reprojectImageTo3D):")
    print(f"    Q[0,3] = {Q[0,3]:.2f}  (cx, px)")
    print(f"    Q[2,3] = {Q[2,3]:.2f}  (fx, px)")
    print(f"    Q[3,2] = {Q[3,2]:.6f}  (-1/baseline, per-m)")
    print()
    print("  Next step:")
    print(f"    Restart the service. StereoCalibrator will load")
    print(f"    {OUTPUT_PATH} and switch to RECTIFIED mode.")
    print("=" * 50)


if __name__ == "__main__":
    run()
