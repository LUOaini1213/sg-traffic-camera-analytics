"""Render pre-annotations for visual review: numbered boxes, colour by class, 100-px grid.

Writes {frame_id}_full.jpg (whole frame) and {frame_id}_z{k}.jpg (two half-height strips at
native resolution) to D:/.../archive/review/. Reviewers then record corrections in
data/review/{frame_id}.json (remove ids, relabel ids, add boxes in native pixels).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
COLORS = {0: (0, 255, 255), 1: (255, 160, 0), 2: (255, 0, 255), 3: (0, 255, 0)}
LETTER = {0: "c", 1: "B", 2: "T", 3: "m"}


def render(frame_id: str) -> list[Path]:
    pre = json.loads((ROOT / "data" / "pre" / f"{frame_id}.json").read_text())
    img = Image.open(DATA / "archive" / "images" / f"{frame_id}.jpg").convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, w // 120))
        small = ImageFont.truetype("arial.ttf", max(10, w // 160))
    except OSError:
        font = small = ImageFont.load_default()
    step = 100 if w >= 1280 else 50
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(90, 90, 90), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0), font=small)
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(90, 90, 90), width=1)
        d.text((2, y + 2), str(y), fill=(255, 255, 0), font=small)
    for b in pre["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        col = COLORS[b["cls"]]
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        d.text((x1, max(0, y1 - font.size - 1)), f"{b['id']}{LETTER[b['cls']]}", fill=col, font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    out = DATA / "archive" / "review"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{frame_id}_full.jpg"]
    img.save(paths[0], quality=88)
    if w >= 1280:
        for k, (y1, y2) in enumerate([(0, h // 2 + 40), (h // 2 - 40, h)]):
            p = out / f"{frame_id}_z{k}.jpg"
            img.crop((0, y1, w, y2)).save(p, quality=90)
            paths.append(p)
    return paths


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(*render(f))
