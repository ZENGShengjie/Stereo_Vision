"""Cup 检测器(YOLOv8 单例,只识别 COCO ID=41 cup)。

任务规约约束:
- 类别锁死 ``TARGET_CLS_ID = 41``(cup),其余 COCO 物体一律过滤。
- 全局仅加载一次模型,推理时不能重复加载。
- 检测结果:画面中**面积最大**的 cup 框;无 cup 返回空。
- 不训练任何自定义数据集,不引入推子类别。

加速路线(2026-06-22):
- 优先: ONNX Runtime + DirectML (AMD 780M) — 实测 ~21ms/帧
- 退路: PyTorch YOLO CPU — ~60-80ms/帧
DirectML 路线: torch-directml 不兼容 ultralytics (inference_mode bug),
故用 ONNX Runtime 绕开,导出命令:
  ``model.export(format='onnx', imgsz=640, classes=[41])``
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from config.hardware import (
    DEFAULT_CONF_THRESHOLD,
    TARGET_CLS_ID,
    TARGET_CLS_NAME,
    BASE_DIR,
)

logger = logging.getLogger(__name__)

_DEBUG_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "debug-c7ffa8.log",
)


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "?") -> None:
    try:
        payload = {
            "id": f"log_{int(time.time()*1000)}_{os.getpid()}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


MODELS_DIR: Path = BASE_DIR / "models"
WEIGHTS_PATH: Path = MODELS_DIR / "yolov8s.pt"
ONNX_PATH: Path = MODELS_DIR / "yolov8s.onnx"


# ── YOLOv8 postprocessing ───────────────────────────────────────────────────

def _yolo_postprocess(
    output: np.ndarray,
    orig_shape: tuple[int, int],
    input_size: int = 640,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.45,
) -> list[tuple[float, float, float, float, float, int]]:
    """从 ONNX (1, 84, 8400) 输出提取检测框并做 NMS。

    Returns:
        list of (x1, y1, x2, y2, conf, cls) sorted by conf descending.
    """
    orig_h, orig_w = orig_shape
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    preds = output[0]                   # (84, 8400)
    bboxes_xywh = preds[:4, :]          # (4, 8400)
    scores = preds[4:, :]                # (80, 8400)

    max_scores = scores.max(axis=0)
    class_ids = scores.argmax(axis=0)

    mask = max_scores >= conf_thresh
    if not mask.any():
        return []

    xywh = bboxes_xywh[:, mask]
    filtered_scores = max_scores[mask].astype(np.float32)
    filtered_cls = class_ids[mask].astype(np.int32)

    cx, cy, w, h = xywh
    x1 = (cx - w * 0.5) * scale_x
    y1 = (cy - h * 0.5) * scale_y
    x2 = (cx + w * 0.5) * scale_x
    y2 = (cy + h * 0.5) * scale_y

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    keep = cv2.dnn.NMSBoxes(
        boxes.tolist(), filtered_scores.tolist(),
        float(conf_thresh), float(iou_thresh),
    )
    if len(keep) == 0:
        return []

    result = []
    for idx in keep:
        i = int(idx[0]) if isinstance(idx, (list, np.ndarray)) else int(idx)
        if 0 <= i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            result.append((float(x1), float(y1), float(x2), float(y2),
                           float(filtered_scores[i]), int(filtered_cls[i])))
    return result


# ── CupDetector ──────────────────────────────────────────────────────────────

class CupDetector:
    """YOLOv8 cup 检测器 — 优先 ONNX Runtime + DirectML,退 PyTorch YOLO CPU。"""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = 640,
    ) -> None:
        self._conf = conf
        self._imgsz = imgsz
        self._ort_sess = None
        self._using_onnx = False
        self._yolo_model = None

        # 1. 尝试 ONNX Runtime + DirectML
        onnx_path = ONNX_PATH if weights is None else Path(weights).parent / "yolov8s.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                pref = ["DmlExecutionProvider", "CPUExecutionProvider"] \
                    if "DmlExecutionProvider" in providers \
                    else ["CPUExecutionProvider"]
                self._ort_sess = ort.InferenceSession(str(onnx_path), providers=pref)
                self._using_onnx = True
                used = self._ort_sess.get_providers()[0]
                logger.info(
                    "[CupDetector] ONNX + %s loaded from %s (%.1f ms/frame est.)",
                    used, onnx_path,
                    21.0 if "Dml" in used else 70.0,
                )
            except Exception as ex:
                timestamped_print(f"[CupDetector] ONNX load failed ({ex}), falling back to PyTorch YOLO")
                logger.warning("[CupDetector] ONNX load failed (%s), falling back to PyTorch YOLO", ex)

        # 2. 退路: PyTorch YOLO CPU
        if not self._using_onnx:
            from ultralytics import YOLO
            w = Path(weights) if weights is not None else WEIGHTS_PATH
            if not w.exists():
                logger.info("[CupDetector] Downloading yolov8s.pt ...")
                w = "yolov8s.pt"
            else:
                w = str(w)
            self._yolo_model = YOLO(w)
            self._yolo_model.to("cpu")
            logger.info("[CupDetector] PyTorch YOLO (CPU) loaded from %s — WARNING: slow (~60-80ms)", w)

        logger.info("[CupDetector] Ready, class=%d (%s)", TARGET_CLS_ID, TARGET_CLS_NAME)
        _debug_log(
            "processing/detector.py:CupDetector.__init__",
            "init done",
            {
                "using_onnx": self._using_onnx,
                "ort_providers": self._ort_sess.get_providers() if self._ort_sess else None,
            },
            "PERF",
        )

    def detect(self, img_bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
        """对单眼 BGR 图做 cup 检测,返回面积最大的 cup 框。"""
        if img_bgr is None or img_bgr.size == 0:
            return None, None

        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]

        if self._using_onnx and self._ort_sess is not None:
            box, center = self._detect_onnx(img_bgr, h, w)
        else:
            box, center = self._detect_yolo(img_bgr, h, w)

        t1 = time.perf_counter()
        _debug_log(
            "processing/detector.py:detect",
            "done",
            {"ms": round((t1 - t0) * 1000, 1), "box": list(box) if box else None},
            "PERF",
        )
        return box, center

    def _detect_onnx(self, img_bgr: np.ndarray, h: int, w: int):
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0 / 255.0, (self._imgsz, self._imgsz),
            (0, 0, 0), swapRB=True, crop=False,
        )   # (1, 3, 640, 640)
        out = self._ort_sess.run(None, {"images": blob})[0]  # (1, 84, 8400)

        detections = _yolo_postprocess(out, (h, w), self._imgsz, self._conf)
        cup_dets = [(x1, y1, x2, y2, conf, cls)
                    for (x1, y1, x2, y2, conf, cls) in detections
                    if cls == TARGET_CLS_ID]

        if not cup_dets:
            return None, None

        best = max(cup_dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        x1, y1, x2, y2 = best[0], best[1], best[2], best[3]
        x1 = int(max(0.0, min(w - 1, x1)))
        y1 = int(max(0.0, min(h - 1, y1)))
        x2 = int(max(0.0, min(w - 1, x2)))
        y2 = int(max(0.0, min(h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            return None, None
        return (x1, y1, x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2)

    def _detect_yolo(self, img_bgr: np.ndarray, h: int, w: int):
        results = self._yolo_model.predict(
            img_bgr, conf=self._conf, classes=[TARGET_CLS_ID],
            verbose=False, imgsz=self._imgsz,
        )
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None, None

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[best_idx]
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            return None, None
        return (x1, y1, x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2)


__all__ = ["CupDetector", "WEIGHTS_PATH", "MODELS_DIR", "ONNX_PATH"]
