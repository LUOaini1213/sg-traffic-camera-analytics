"""Per-camera congestion flag from a stream of vehicle counts.

A camera is flagged when the median total count over the last `window` frames reaches its
threshold, and stays flagged until that median falls below `release` x threshold (hysteresis).
The threshold is the camera's own `quantile` of counts in a calibration period, never below
`floor` vehicles, so an empty road at night cannot be flagged by a single parked lorry.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def calibrate(counts, quantile: float = 0.85, floor: int = 8) -> float:
    counts = np.asarray(counts, float)
    if len(counts) == 0:
        return float(floor)
    return max(float(np.quantile(counts, quantile)), float(floor))


@dataclass
class CameraFlag:
    threshold: float
    window: int = 3
    release: float = 0.8
    history: deque = field(default_factory=deque)
    flagged: bool = False

    def update(self, total: int) -> dict:
        self.history.append(total)
        while len(self.history) > self.window:
            self.history.popleft()
        med = float(np.median(self.history))
        if not self.flagged and len(self.history) >= self.window and med >= self.threshold:
            self.flagged = True
        elif self.flagged and med < self.release * self.threshold:
            self.flagged = False
        return {"median": med, "threshold": self.threshold, "congested": self.flagged}


def flag_series(totals, threshold: float, window: int = 3, release: float = 0.8) -> list[bool]:
    f = CameraFlag(threshold, window, release)
    return [f.update(t)["congested"] for t in totals]
