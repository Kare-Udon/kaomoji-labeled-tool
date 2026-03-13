from __future__ import annotations

import base64


def test_render_kaomoji_image_base64_returns_png_bytes():
    from kaomoji_labeled_tool.step6_render import render_kaomoji_image_base64

    payload = render_kaomoji_image_base64("(╥﹏╥)")
    raw = base64.b64decode(payload)

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_kaomoji_image_base64_falls_back_without_system_font(monkeypatch):
    from kaomoji_labeled_tool import step6_render

    monkeypatch.setattr(step6_render, "_FONT_CANDIDATES", ())

    payload = step6_render.render_kaomoji_image_base64("(╥﹏╥)")

    assert base64.b64decode(payload).startswith(b"\x89PNG\r\n\x1a\n")
