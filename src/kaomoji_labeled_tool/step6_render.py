from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/CJKSymbolsFallback.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def render_kaomoji_image_base64(kaomoji: str, *, font_path: str | None = None) -> str:
    resolved_font_path = _resolve_font_path(font_path)
    if resolved_font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(resolved_font_path, 28)
    dummy = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), kaomoji, font=font)
    width = max(128, bbox[2] - bbox[0] + 24)
    height = max(56, bbox[3] - bbox[1] + 20)
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    x = (width - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), kaomoji, font=font, fill=0)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resolve_font_path(custom_font_path: str | None = None) -> str | None:
    if custom_font_path is not None:
        return custom_font_path
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None
