"""Small geometry helpers (WGS84, metres)."""
from __future__ import annotations

import math

EARTH_R = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def point_segment_m(lat: float, lon: float, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance from a point to a short segment, using a local equirectangular projection."""
    kx = math.cos(math.radians(lat)) * math.pi / 180 * EARTH_R
    ky = math.pi / 180 * EARTH_R
    px, py = 0.0, 0.0
    ax, ay = (lon1 - lon) * kx, (lat1 - lat) * ky
    bx, by = (lon2 - lon) * kx, (lat2 - lat) * ky
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(cx - px, cy - py)
