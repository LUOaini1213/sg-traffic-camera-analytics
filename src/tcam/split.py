"""Camera-level split. Cameras closer than LINK_M are grouped so that two views of the
same stretch of road never land on different sides of the split."""
from __future__ import annotations

import random

from .geo import haversine_m

LINK_M = 600.0


def camera_groups(cams: dict[str, tuple[float, float]], link_m: float = LINK_M) -> list[list[str]]:
    ids = sorted(cams)
    parent = {c: c for c in ids}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if haversine_m(*cams[a], *cams[b]) < link_m:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for c in ids:
        groups.setdefault(find(c), []).append(c)
    return sorted(groups.values())


def split_cameras(cams: dict[str, tuple[float, float]], frames_per_cam: dict[str, int],
                  test_frac: float = 0.2, val_frac: float = 0.1, force_train: set[str] = frozenset(),
                  seed: int = 13) -> dict[str, str]:
    groups = camera_groups(cams)
    rng = random.Random(seed)
    rng.shuffle(groups)
    total = sum(frames_per_cam.get(c, 0) for c in cams)
    out: dict[str, str] = {}
    n = {"test": 0, "val": 0}
    for g in groups:
        k = sum(frames_per_cam.get(c, 0) for c in g)
        if any(c in force_train for c in g):
            part = "train"
        elif n["test"] + k <= test_frac * total:
            part = "test"
        elif n["val"] + k <= val_frac * total:
            part = "val"
        else:
            part = "train"
        if part in n:
            n[part] += k
        for c in g:
            out[c] = part
    return out
