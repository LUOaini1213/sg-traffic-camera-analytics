"""Per-frame vehicle counts from detections."""
from __future__ import annotations

from collections.abc import Sequence

from .boxes import size_ok
from .classes import MIN_SIDE_FRAC, NAMES


def point_in_polygon(x: float, y: float, poly: Sequence[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def count(boxes, scores, classes, width: int, conf: float = 0.25, roi=None) -> dict[str, int]:
    """Count detections with score >= conf, size above the labelling floor, and (if an ROI
    polygon in pixels is given) bottom-centre inside the ROI. Returns per-class and total."""
    out = {n: 0 for n in NAMES}
    for b, s, c in zip(boxes, scores, classes):
        if s < conf or not size_ok(b, width, MIN_SIDE_FRAC):
            continue
        if roi is not None and not point_in_polygon((b[0] + b[2]) / 2, b[3], roi):
            continue
        out[NAMES[int(c)]] += 1
    out["total"] = sum(out[n] for n in NAMES)
    return out
