"""Stereo display processing: crop, shift, undistort, separator.

All parameters are read from the global DisplayParamsStore in models/.
The control panel updates the store; this module reads from it on each frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from models import current_params_store


def _split_eyes(stereo: np.ndarray):
    h, w, _ = stereo.shape
    eye_w = w // 2
    left = stereo[:, :eye_w]
    right = stereo[:, eye_w:]
    return left, right, h, eye_w


def _crop_and_resize(left, right, eye_w, h, crop_ratio):
    if crop_ratio < 1.0:
        crop_w = int(eye_w * crop_ratio)
        x0 = (eye_w - crop_w) // 2
        x1 = x0 + crop_w
        left_crop = left[:, x0:x1]
        right_crop = right[:, x0:x1]
    else:
        left_crop = left
        right_crop = right
    interp = cv2.INTER_LINEAR if crop_ratio < 1.0 else cv2.INTER_AREA
    left_r = cv2.resize(left_crop, (eye_w, h), interpolation=interp)
    right_r = cv2.resize(right_crop, (eye_w, h), interpolation=interp)
    return left_r, right_r


def _apply_xshift(left_r, right_r, eye_w, xshift_px):
    shift = int(xshift_px)
    if shift == 0:
        return left_r, right_r
    left_s = np.zeros_like(left_r)
    right_s = np.zeros_like(right_r)
    if shift > 0:
        left_s[:, shift:] = left_r[:, : eye_w - shift]
        right_s[:, : eye_w - shift] = right_r[:, shift:]
    else:
        s = -shift
        left_s[:, : eye_w - s] = left_r[:, s:]
        right_s[:, s:] = right_r[:, : eye_w - s]
    return left_s, right_s


_WARP_CACHE: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}


def _get_warp_maps(h, eye_w, k1, k2):
    key = (float(k1), float(k2))
    if key in _WARP_CACHE:
        return _WARP_CACHE[key]
    cx = eye_w / 2.0
    cy = h / 2.0
    f = float(eye_w)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([float(k1), float(k2), 0, 0, 0], dtype=np.float32)
    w1, w2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (eye_w, h), cv2.CV_32FC1
    )
    _WARP_CACHE[key] = (w1, w2)
    return w1, w2


def _apply_global_shift(img, dx, dy):
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return img
    h_img, w_img, _ = img.shape
    dst = np.zeros_like(img)
    if dy > 0:
        dst[dy:, :, :] = img[: h_img - dy, :, :]
    elif dy < 0:
        s = -dy
        dst[: h_img - s, :, :] = img[s:, :, :]
    tmp = dst.copy()
    if dx > 0:
        dst[:, dx:, :] = tmp[:, : w_img - dx, :]
    elif dx < 0:
        s = -dx
        dst[:, : w_img - s, :] = tmp[:, s:, :]
    return dst


def _apply_separator(left_img, right_img, eye_w, sep_px):
    sep = max(0, min(int(sep_px), eye_w // 3))
    if sep <= 0:
        return left_img, right_img
    left = left_img.copy()
    right = right_img.copy()
    left[:, eye_w - sep :] = 0
    right[:, :sep] = 0
    return left, right


class DisplayProcessor:
    """Processes a raw stereo frame with current display params from the store."""

    def process(self, stereo: np.ndarray) -> np.ndarray | None:
        """Return processed SBS image (left+right) or None."""
        if stereo is None:
            return None
        params = current_params_store.get()
        left, right, h, eye_w = _split_eyes(stereo)

        left_r, right_r = _crop_and_resize(
            left, right, eye_w, h, params["crop"]
        )
        left_s, right_s = _apply_xshift(left_r, right_r, eye_w, params["xshift"])

        w1, w2 = _get_warp_maps(h, eye_w, params["k1"], params["k2"])
        left_w = cv2.remap(left_s, w1, w2, cv2.INTER_LINEAR)
        right_w = cv2.remap(right_s, w1, w2, cv2.INTER_LINEAR)

        gx = max(-eye_w // 2, min(int(params["gshift_x"]), eye_w // 2))
        gy = max(-h // 2, min(int(params["gshift_y"]), h // 2))
        left_w = _apply_global_shift(left_w, gx, gy)
        right_w = _apply_global_shift(right_w, gx, gy)

        left_f, right_f = _apply_separator(
            left_w, right_w, eye_w, params["sep"]
        )
        return cv2.hconcat([left_f, right_f])
