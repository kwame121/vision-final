"""Horizontal strip: satellite | prediction | ground-truth (Pix2Pix-style)."""

from __future__ import annotations

from PIL import Image


def triptych(
    input_tile: Image.Image,
    pred_tile: Image.Image,
    gt_tile: Image.Image,
    size: int = 256,
    gap: int = 0,
) -> Image.Image:
    if gap <= 0:
        strip_w = size * 3
        strip = Image.new("RGB", (strip_w, size))
    else:
        strip_w = size * 3 + gap * 2
        strip = Image.new("RGB", (strip_w, size), (240, 240, 240))
    panels = []
    for t in (input_tile, pred_tile, gt_tile):
        panels.append(
            t if t.size == (size, size) else t.resize((size, size), Image.Resampling.BICUBIC)
        )
    x = 0
    for i, panel in enumerate(panels):
        strip.paste(panel, (x, 0))
        x += size + (gap if i < 2 else 0)
    return strip
