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
        """对单眼 BGR 图做 cup 检测,返回选中的 cup 框。

        Selector(2026-07-03 统一):
        - 单检测 → 直接返回该 box(无歧义)
        - 多检测 → 调用 :meth:`select_nearest`;若 ``disparity_map`` 为 None,
          用 fallback 启发式(下边缘 y + 大面积加权)
        - 无检测 → None

        返回 ``(box, center)`` 兼容旧接口。
        """
        if img_bgr is None or img_bgr.size == 0:
            return None, None

        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]

        cands = self.detect_candidates(img_bgr, h, w)

        if not cands:
            box, center = None, None
        elif len(cands) == 1:
            box, center = cands[0]
        else:
            # 多目标 → 选最近(后面会在 sbs_pipeline 中传 disparity_map 时升级)
            box, center = self._fallback_heuristic_select([c[0] for c in cands], h, w)

        t1 = time.perf_counter()
        _debug_log(
            "processing/detector.py:detect",
            "done",
            {
                "ms": round((t1 - t0) * 1000, 1),
                "num_candidates": len(cands),
                "box": list(box) if box else None,
                "selector": "single" if len(cands) <= 1 else "fallback_heuristic",
            },
            "PERF",
        )
        return box, center

    def detect_candidates(
        self, img_bgr: np.ndarray, img_h: int | None = None, img_w: int | None = None,
    ) -> list[tuple[tuple[int, int, int, int], tuple[int, int]]]:
        """对单眼 BGR 图返回所有 cup 候选 ``[(box, center), ...]``。

        用于 :class:`SBSPipeline` 的"视差辅助选最近"流程:获取全部候选,
        后续用 SGBM 视差筛掉远的目标。
        """
        if img_bgr is None or img_bgr.size == 0:
            return []
        h = img_h if img_h is not None else img_bgr.shape[0]
        w = img_w if img_w is not None else img_bgr.shape[1]
        if self._using_onnx and self._ort_sess is not None:
            return self._detect_candidates_onnx(img_bgr, h, w)
        return self._detect_candidates_yolo(img_bgr, h, w)

    def select_nearest(
        self,
        candidates: list[tuple[int, int, int, int]],
        disparity_map: np.ndarray | None,
    ) -> tuple[int, int, int, int] | None:
        """从多个 cup 候选中选"最近"的一个(视差辅助方案 b,2026-07-03)。

        Args:
            candidates: list of (x1, y1, x2, y2) 候选框。
            disparity_map: 单眼视差图(像素, float32, 0=无效)。若为 None
                或 ROI 内有效视差不足,退化为 :meth:`_fallback_heuristic_select`。

        Returns:
            最近目标的 box(若无候选返回 None)。
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if disparity_map is None:
            return self._fallback_heuristic_select(
                candidates, 0, 0
            )[0]

        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for box in candidates:
            x1, y1, x2, y2 = box
            h, w = disparity_map.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            roi = disparity_map[y1c:y2c, x1c:x2c]
            valid = roi[roi > 0]
            if valid.size < 20:
                continue
            # 视差均值 → 越大表示越近
            scored.append((float(valid.mean()), box))

        if not scored:
            return self._fallback_heuristic_select(
                candidates, disparity_map.shape[0], disparity_map.shape[1]
            )[0]
        # 选视差最大的(最近)
        return max(scored, key=lambda s: s[0])[1]

    @staticmethod
    def _fallback_heuristic_select(
        candidates: list[tuple[int, int, int, int]],
        img_h: int,
        img_w: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        """无 disparity_map 时的兜底:画面下边缘 + 大面积加权。

        score = area * (1 + 0.5 * (y_center / img_h))
        y 越大(越靠下)权重越高 → 视觉上更近的目标更可能胜出。
        """
        def score(box):
            x1, y1, x2, y2 = box
            area = max(1, (x2 - x1) * (y2 - y1))
            y_center = (y1 + y2) / 2.0
            y_norm = y_center / max(1, img_h)
            return area * (1.0 + 0.5 * y_norm)
        best = max(candidates, key=score)
        return best, ((best[0] + best[2]) // 2, (best[1] + best[3]) // 2)

    def _detect_candidates_onnx(self, img_bgr: np.ndarray, h: int, w: int) -> list[tuple[tuple[int, int, int, int], tuple[int, int]]]:
        """ONNX 后端:返回所有 cup 候选 ``(box, center)`` 列表。"""
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0 / 255.0, (self._imgsz, self._imgsz),
            (0, 0, 0), swapRB=True, crop=False,
        )
        out = self._ort_sess.run(None, {"images": blob})[0]
        detections = _yolo_postprocess(out, (h, w), self._imgsz, self._conf)
        cup_dets = [(x1, y1, x2, y2, conf, cls)
                    for (x1, y1, x2, y2, conf, cls) in detections
                    if cls == TARGET_CLS_ID]
        results = []
        for (x1, y1, x2, y2, conf, cls) in cup_dets:
            x1i = int(max(0.0, min(w - 1, x1)))
            y1i = int(max(0.0, min(h - 1, y1)))
            x2i = int(max(0.0, min(w - 1, x2)))
            y2i = int(max(0.0, min(h - 1, y2)))
            if x2i <= x1i or y2i <= y1i:
                continue
            results.append(((x1i, y1i, x2i, y2i), ((x1i + x2i) // 2, (y1i + y2i) // 2)))
        return results

    def _detect_candidates_yolo(self, img_bgr: np.ndarray, h: int, w: int) -> list[tuple[tuple[int, int, int, int], tuple[int, int]]]:
        """PyTorch YOLO 后端:返回所有 cup 候选。"""
        results_obj = self._yolo_model.predict(
            img_bgr, conf=self._conf, classes=[TARGET_CLS_ID],
            verbose=False, imgsz=self._imgsz,
        )
        if not results_obj or not results_obj[0].boxes or len(results_obj[0].boxes) == 0:
            return []
        boxes = results_obj[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        results = []
        for row in xyxy:
            x1, y1, x2, y2 = row
            x1i = int(max(0, min(w - 1, x1)))
            y1i = int(max(0, min(h - 1, y1)))
            x2i = int(max(0, min(w - 1, x2)))
            y2i = int(max(0, min(h - 1, y2)))
            if x2i <= x1i or y2i <= y1i:
                continue
            results.append(((x1i, y1i, x2i, y2i), ((x1i + x2i) // 2, (y1i + y2i) // 2)))
        return results

    def _detect_onnx(self, img_bgr: np.ndarray, h: int, w: int):
        # 兼容旧调用:返回单 box(走 fallback heuristic)
        cands = self._detect_candidates_onnx(img_bgr, h, w)
        if not cands:
            return None, None
        return self._fallback_heuristic_select([c[0] for c in cands], h, w)

    def _detect_yolo(self, img_bgr: np.ndarray, h: int, w: int):
        cands = self._detect_candidates_yolo(img_bgr, h, w)
        if not cands:
            return None, None
        return self._fallback_heuristic_select([c[0] for c in cands], h, w)


__all__ = ["CupDetector", "WEIGHTS_PATH", "MODELS_DIR", "ONNX_PATH"]
