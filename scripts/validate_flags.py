"""Check camera counts and congestion flags against LTA speed bands and incident reports.

Input: a detector log (JSONL from tcam.stream.detector) for a set of frames, the frame manifest
that maps frame_id -> snapshot time, and the speed-band snapshots for those times.

week mode (2026-09-26 .. 10-03, archive frames at speed-band snapshot times):
  * thresholds are calibrated on frames before --split-at and applied after it;
  * thresholds are saved to data/thresholds.json for the live service.
live mode (frames from the live collector, speed bands every 5 min and incidents every 2 min):
  * thresholds come from data/thresholds.json (calibrated on the week data, never on live data).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.flags import calibrate, flag_series  # noqa: E402
from tcam.validate import (SLOW_BAND, agreement, camera_speed, incidents_near, links_near, read_snapshot,  # noqa: E402
                           spearman)


def nearest(times: list[datetime], t: datetime, max_s: float):
    i = bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times) and abs((times[j] - t).total_seconds()) <= max_s:
            if best is None or abs((times[j] - t).total_seconds()) < abs((times[best] - t).total_seconds()):
                best = j
    return best


def load_log(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            out[e["frame_id"]] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["week", "live"])
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--speed-dir", type=Path, required=True)
    ap.add_argument("--links", type=Path, required=True)
    ap.add_argument("--incident-dir", type=Path)
    ap.add_argument("--split-at", default="2026-09-30T00:00")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    log = load_log(a.log)
    frames = [r for r in csv.DictReader(a.manifest.open()) if r["frame_id"] in log]
    cams = {}
    for r in frames:
        cams[r["camera"]] = (float(r["lat"]), float(r["lon"]))
    link_sets = links_near(cams, csv.DictReader(a.links.open()))
    snaps = sorted(a.speed_dir.glob("sb_*.csv.gz"))
    snap_t = [datetime.strptime(p.name[3:16], "%Y%m%d_%H%M") for p in snaps]
    cache: dict[int, dict] = {}

    def frame_time(r):
        if a.mode == "week":
            return datetime.fromisoformat(r["snapshot"])
        return datetime.fromisoformat(r["capture"])

    inc_files = sorted(a.incident_dir.glob("inc_*.json")) if a.incident_dir else []
    inc_t = [datetime.strptime(p.name[4:19], "%Y%m%d_%H%M%S") for p in inc_files]

    series = defaultdict(list)  # cam -> list of dict(time, total, speed, incidents)
    for r in sorted(frames, key=frame_time):
        t = frame_time(r)
        k = nearest(snap_t, t, 120 if a.mode == "week" else 330)
        if k is None:
            continue
        if k not in cache:
            cache[k] = read_snapshot(snaps[k])
        e = log[r["frame_id"]]
        row = {"time": t.isoformat(timespec="minutes"), "total": e["counts"]["total"],
               "speed": camera_speed(cache[k], link_sets[r["camera"]])}
        if inc_files:
            j = nearest(inc_t, t, 180)
            near = [] if j is None else incidents_near(json.loads(inc_files[j].read_text()), *cams[r["camera"]])
            row["incident_types"] = sorted({i["Type"] for i in near})
        series[r["camera"]].append(row)

    th_path = ROOT / "data" / "thresholds.json"
    if a.mode == "week":
        thresholds = {c: calibrate([x["total"] for x in s if x["time"] < a.split_at]) for c, s in series.items()}
        th_path.write_text(json.dumps(thresholds, indent=1, sort_keys=True))
    else:
        thresholds = json.loads(th_path.read_text())

    result = {"mode": a.mode, "slow_rule": f"mean SpeedBand of category 1-4 links within 200 m <= {SLOW_BAND}",
              "cameras": {}}
    pooled_pred, pooled_true = [], []
    for cam, s in sorted(series.items()):
        ev = [x for x in s if a.mode == "live" or x["time"] >= a.split_at]
        tot = [x["total"] for x in s]
        flags = flag_series(tot, thresholds[cam])
        flags_ev = [f for x, f in zip(s, flags) if a.mode == "live" or x["time"] >= a.split_at]
        slow = [x["speed"] <= SLOW_BAND for x in ev]
        valid = [not np.isnan(x["speed"]) for x in ev]
        pred = [f for f, v in zip(flags_ev, valid) if v]
        truth = [t for t, v in zip(slow, valid) if v]
        pooled_pred += pred; pooled_true += truth
        c = {"frames": len(s), "frames_scored": len(pred), "links": len(link_sets[cam]), "threshold": thresholds[cam],
             "spearman_count_vs_speedband_all": spearman(tot, [x["speed"] for x in s]),
             "mean_count": float(np.mean(tot)), "flag_rate": float(np.mean(pred)) if pred else float("nan"),
             "slow_rate": float(np.mean(truth)) if truth else float("nan"),
             "flag_vs_slow": agreement(pred, truth) if pred else None}
        if inc_files:
            types = defaultdict(int)
            for x in ev:
                for ty in x.get("incident_types", []):
                    types[ty] += 1
            c["frames_with_incident_within_1km"] = dict(types)
            for ty in types:
                hit = [f for x, f in zip(ev, flags_ev) if ty in x.get("incident_types", [])]
                c.setdefault("flag_rate_when_incident", {})[ty] = float(np.mean(hit)) if hit else None
        result["cameras"][cam] = c
    result["pooled_flag_vs_slow"] = agreement(pooled_pred, pooled_true) if pooled_pred else None
    a.out.write_text(json.dumps(result, indent=1, default=float))
    (a.out.with_suffix(".series.json")).write_text(json.dumps(series))
    print(json.dumps({k: v for k, v in result.items() if k != "cameras"}, indent=1, default=float))
    for cam, c in result["cameras"].items():
        print(cam, {k: c[k] for k in ("frames", "frames_scored", "links", "threshold", "spearman_count_vs_speedband_all",
                                      "flag_rate", "slow_rate")})


if __name__ == "__main__":
    main()
