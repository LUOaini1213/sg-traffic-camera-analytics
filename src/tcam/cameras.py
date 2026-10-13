"""Camera scope.

Cameras west of 103.66 E (the far-western industrial area) are outside this project's
scope and are dropped everywhere: collection, labelling, training and evaluation.
"""
from __future__ import annotations

WEST_LIMIT_LON = 103.66


def in_scope(lon: float) -> bool:
    return float(lon) >= WEST_LIMIT_LON
