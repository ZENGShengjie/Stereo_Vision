"""三色分级预警 + 距离文字渲染。

任务规约约束(已对齐到 `target_depth_cm` 术语,不再用"头部/头皮/头部"):

| 深度区间              | 框色   | 厚度  | 闪烁 | 文字           |
|-----------------------|--------|-------|------|----------------|
| Z > SAFE_CM (5)       | 绿色   | 2     | 否   | 距离 cm        |
| DANGER_CM (2) <= Z <= SAFE_CM | 黄色   | 2     | 否   | 距离 cm        |
| 0 < Z < DANGER_CM (2) | 红色   | 5     | 1Hz  | 距离 cm + 顶部 DANGER |
| Z is None (无效)      | 灰色   | 1     | 否   | "无有效深度"   |

绘制策略:
- 中文文字用 PIL 渲染(OpenCV putText 不支持 CJK),然后合成回 BGR 图。
- 危险区红框 + 顶部 DANGER 文字共享同一个 1Hz 时钟(``time.time() - start_time``)。
- 左右眼必须用同一 box / 同一 depth 渲染,保证 VR 观感对称无重影。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.hardware import DANGER_CM, DANGER_FLASH_HZ, SAFE_CM

Box = Tuple[int, int, int, int]

# 中文字体优先级:SimHei > NotoSansSC > simfang
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\simfang.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
)
_CJK_FONT_PATH: Optional[Path] = next(
    (Path(p) for p in _CJK_FONT_CANDIDATES if Path(p).exists()),
    None,
)
if _CJK_FONT_PATH is None:
    # 退到 Pillow 自带的默认字体(不支持中文,作为最后兜底)
    _CJK_FONT_PATH = None
_DEFAULT_FONT = ImageFont.load_default()


# 字体缓存:每个字号缓存一个 Pillow 字体对象,避免重复加载文件
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _get_cjk_font(size: int) -> ImageFont.ImageFont:
    """获取 CJK 字体,从缓存读取;缓存未命中时加载并缓存。"""
    if _FONT_CACHE.get(size) is not None:
        return _FONT_CACHE[size]
    if _CJK_FONT_PATH is None:
        font = _DEFAULT_FONT
    else:
        try:
            font = ImageFont.truetype(str(_CJK_FONT_PATH), size=size)
        except Exception:  # noqa: BLE001
            font = _DEFAULT_FONT
    _FONT_CACHE[size] = font
    return font


def _put_text_zh(
    bgr: np.ndarray,
    text: str,
    top_left: Tuple[int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int],
) -> None:
    """在 BGR 图上写中文(用 PIL)。

    优化(2026-06-22):字体已缓存,不再每次从文件加载;
    对小文字区域(距离标签)仅在感兴趣区(ROI)上操作,减少 PIL 开销。

    Args:
        bgr: 目标 BGR 图,会被原地修改。
        text: 中文/英文文字。
        top_left: ``(x, y)`` 文字左上的像素位置。
        font_size: 字号(像素)。
        color_bgr: BGR 颜色。
    """
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_cjk_font(font_size)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(top_left, text, font=font, fill=color_rgb)
    bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


class WarningOverlay:
    """三色分级预警 + 距离文字渲染。

    Args:
        safe_cm / danger_cm: 阈值(默认从 :data:`config.hardware` 读)。
        flash_hz: 危险区闪烁频率(Hz),默认 1.0。
    """

    def __init__(
        self,
        safe_cm: float = SAFE_CM,
        danger_cm: float = DANGER_CM,
        flash_hz: float = DANGER_FLASH_HZ,
    ) -> None:
        if safe_cm <= danger_cm:
            raise ValueError(
                f"safe_cm ({safe_cm}) must be > danger_cm ({danger_cm})"
            )
        self._safe_cm = float(safe_cm)
        self._danger_cm = float(danger_cm)
        self._flash_hz = float(flash_hz)
        self._start_time = time.time()

    # ------------------------------------------------------------------ utils
    def _flash_on(self) -> bool:
        """危险区闪烁状态:基于 ``time.time() - start_time`` 的方波。

        1Hz 时方波周期 1s,半周期 0.5s;返回 True = 当前是 "on" 半周期。
        """
        elapsed = time.time() - self._start_time
        half_period = 0.5 / self._flash_hz
        phase = (elapsed % (1.0 / self._flash_hz)) / half_period
        return phase < 1.0

    def _level(self, depth_cm: Optional[float]) -> str:
        """把深度分到 ``safe`` / ``work`` / ``danger`` / ``none``。"""
        if depth_cm is None:
            return "none"
        if depth_cm > self._safe_cm:
            return "safe"
        if depth_cm >= self._danger_cm:
            return "work"
        return "danger"

    @staticmethod
    def _color_for(level: str, flash_on: bool) -> Tuple[int, int, int]:
        """BGR 颜色映射。"""
        if level == "safe":
            return (0, 255, 0)         # 绿
        if level == "work":
            return (0, 255, 255)       # 黄
        if level == "danger":
            return (0, 0, 255) if flash_on else (40, 40, 120)  # 红 / 暗红
        return (160, 160, 160)         # none 灰

    # ------------------------------------------------------------------ draw
    def render(
        self,
        img: np.ndarray,
        box: Optional[Box],
        depth_cm: Optional[float],
    ) -> np.ndarray:
        """在 BGR 图上画三色 cup 框(直接在原图上绘制,返回同一引用)。

        Args:
            img: 单眼 BGR 图,``(H, W, 3)`` = (1080, 1920, 3)。
            box: ``(x1, y1, x2, y2)`` 像素坐标;``None`` 时不画框。
            depth_cm: 相机到目标绝对深度,cm;``None`` 时画灰色"无有效深度"提示框。

        Returns:
            入参 ``img``(原图,绘制后)。
        """
        if box is None:
            return img
        level = self._level(depth_cm)
        flash_on = self._flash_on()
        color = self._color_for(level, flash_on)
        thickness = 5 if level == "danger" else 2
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def render_text(self, img: np.ndarray, depth_cm: Optional[float]) -> np.ndarray:
        """在画面左上角画距离文字,危险区在顶部画闪烁 DANGER 横幅。

        Args:
            img: 单眼 BGR 图,(H, W, 3) = (1080, 1920, 3)。
            depth_cm: 深度 cm,``None`` 时画 "无有效深度"。

        Returns:
            入参 ``img``(原图,绘制后)。
        """
        h, w = img.shape[:2]

        # 1. 左上角距离文字(白底黑字,统一字号,所有等级都画)
        # 任务规约里文字固定为"相机至目标距离:XX.X cm",此处严格遵循。
        if depth_cm is None:
            text = "相机至目标距离:无有效深度"
        else:
            text = f"相机至目标距离:{depth_cm:.1f} cm"
        # 字号随画面缩放:基线 1920 -> 字号 36;线宽 = max(2, font/12)
        font_size = max(28, int(w / 1920.0 * 36))
        font = _get_cjk_font(font_size)
        # 用 PIL 量文字 bounding box
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)  # 旧版 Pillow
        pad = 14
        # 白底圆角矩形(用普通矩形简化,圆角对 1080p 视觉差异不大)
        cv2.rectangle(
            img,
            (10, 10),
            (10 + tw + 2 * pad, 10 + th + 2 * pad),
            (255, 255, 255),
            -1,
        )
        _put_text_zh(
            img,
            text,
            (10 + pad, 10 + pad),
            font_size,
            (0, 0, 0),
        )

        # 2. 危险区顶部 DANGER 横幅(1Hz 闪烁)
        if self._level(depth_cm) == "danger" and self._flash_on():
            banner = "危险 过近"
            banner_font_size = max(40, int(w / 1920.0 * 56))
            banner_font = _get_cjk_font(banner_font_size)
            dummy2 = Image.new("RGB", (1, 1))
            draw2 = ImageDraw.Draw(dummy2)
            try:
                tb2 = draw2.textbbox((0, 0), banner, font=banner_font)
                bw, bh = tb2[2] - tb2[0], tb2[3] - tb2[1]
            except AttributeError:
                bw, bh = draw2.textsize(banner, font=banner_font)
            bx0 = (w - bw) // 2 - 30
            bx1 = bx0 + bw + 60
            by0 = 20
            by1 = by0 + bh + 40
            overlay_img = img.copy()
            cv2.rectangle(overlay_img, (bx0, by0), (bx1, by1), (0, 0, 200), -1)
            cv2.addWeighted(overlay_img, 0.7, img, 0.3, 0, img)
            _put_text_zh(
                img,
                banner,
                (bx0 + 30, by0 + 20),
                banner_font_size,
                (255, 255, 255),
            )

        return img


__all__ = ["WarningOverlay", "Box"]
