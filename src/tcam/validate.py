"""Compare camera congestion flags with LTA speed bands and incident reports.

Rules fixed before any agreement number was computed (see README):
  * speed context of a camera = mean SpeedBand of road links with RoadCategory 1-4 whose
    segment passes within RADIUS_M of the camera;
  * "slow" = that mean <= SLOW_BAND (band 3 = 20-29 km/h);
  * an incident is "near" a camera if it lies within INCIDENT_M and appears in the incident
    snapshot closest in time to the frame (snapshots every 2 minutes).
"""
from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .geo import haversine_m, point_segment_m

RADIUS_M = 200.0
CATEGORIES = {"1", "2", "3", "4"}
SLOW_BAND = 3.0
INCIDENT_M = 1000.0


def links_near(cams: dict[str, tuple[float, float]], links: Iterable[dict], radius_m: float = RADIUS_M,
               categories=CATEGORIES) -> dict[str, set[str]]:
    out = {c: set() for c in cams}
    for l in links:
        if l["RoadCategory"] not in categories:
            continue
        seg = [float(l[k]) for k in ("StartLat", "StartLon", "EndLat", "EndLon")]
        for c, (lat, lon) in cams.items():
            if point_segment_m(lat, lon, *seg) <= radius_m:
                out[c].add(l["LinkID"])
    return out


def read_snapshot(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", newline="") as f:
        return {r["LinkID"]: int(r["SpeedBand"]) for r in csv.DictReader(f)}


def camera_speed(bands: dict[str, int], link_ids: set[str]) -> float:
    v = [bands[i] for i in link_ids if i in bands]
    return float(np.mean(v)) if v else float("nan")


def incidents_near(incidents: list[dict], lat: float, lon: float, radius_m: float = INCIDENT_M,
                   types: set[str] | None = None) -> list[dict]:
    return [i for i in incidents
            if (types is None or i["Type"] in types)
            and haversine_m(lat, lon, float(i["Latitude"]), float(i["Longitude"])) <= radius_m]


def agreement(pred: list[bool], truth: list[bool]) -> dict:
    p = np.asarray(pred, bool)
    t = np.asarray(truth, bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else float("nan")
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else float("nan")
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else float("nan")
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "accuracy": po, "kappa": kappa}


def spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")

    def rank(a):
        o = np.argsort(a, kind="stable")
        r = np.empty(len(a)); r[o] = np.arange(len(a))
        for v in np.unique(a):  # average ties
            m = a == v
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
