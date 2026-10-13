"""Live collector: camera frames, incidents and nearby speed bands from LTA DataMall.

Usage:  LTA_KEY=... python -m tcam.collect OUT_DIR --until 202610141100
Frames are deduplicated by MD5 per camera. Raw images stay in OUT_DIR and are never
committed (see README, "Data and licence").
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import datamall
from .cameras import in_scope
from .geo import point_segment_m

LINK_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/\d{2}-\d{2}/(\d+)_(\d{4})_(\d{14})_")
NEAR_M = 1500.0


def capture_time(link: str) -> str | None:
    """Local capture time 'YYYY-mm-ddTHH:MM' parsed from a DataMall ImageLink."""
    m = LINK_RE.search(link.split("?")[0])
    if not m:
        return None
    day, _cam, hhmm, _upload = m.groups()
    return f"{day}T{hhmm[:2]}:{hhmm[2:]}"


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


class Collector:
    def __init__(self, out: Path):
        self.out = out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / "incidents").mkdir(exist_ok=True)
        (out / "speedbands").mkdir(exist_ok=True)
        self.index = out / "frames.csv"
        self.seen: dict[str, set[str]] = {}
        if self.index.exists():
            with self.index.open(newline="") as f:
                for r in csv.DictReader(f):
                    self.seen.setdefault(r["camera"], set()).add(r["md5"])
        self.cams: dict[str, tuple[float, float]] = {}
        self.near_links: set[str] | None = None

    def frames(self) -> int:
        new = 0
        rows = []
        for v in datamall.traffic_images():
            lat, lon = float(v["Latitude"]), float(v["Longitude"])
            if not in_scope(lon):
                continue
            cam = str(v["CameraID"])
            self.cams[cam] = (lat, lon)
            try:
                b = datamall.download(v["ImageLink"])
            except Exception as e:  # one bad image must not stop the loop
                log(f"frame {cam} failed: {type(e).__name__}")
                continue
            md5 = hashlib.md5(b).hexdigest()
            if md5 in self.seen.setdefault(cam, set()):
                continue
            self.seen[cam].add(md5)
            ts = capture_time(v["ImageLink"]) or datetime.now().strftime("%Y-%m-%dT%H:%M")
            d = self.out / "frames" / cam
            d.mkdir(exist_ok=True)
            p = d / f"{ts.replace('-', '').replace(':', '').replace('T', '_')}_{md5[:8]}.jpg"
            p.write_bytes(b)
            rows.append({"camera": cam, "capture": ts, "fetched": datetime.now().isoformat(timespec="seconds"),
                         "md5": md5, "bytes": len(b), "lat": lat, "lon": lon, "path": str(p.relative_to(self.out))})
            new += 1
        if rows:
            head = not self.index.exists()
            with self.index.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if head:
                    w.writeheader()
                w.writerows(rows)
        return new

    def incidents(self) -> int:
        v = datamall.traffic_incidents()
        p = self.out / "incidents" / f"inc_{datetime.now():%Y%m%d_%H%M%S}.json"
        p.write_text(json.dumps(v, ensure_ascii=False))
        return len(v)

    def speedbands(self) -> int:
        recs = datamall.speed_bands()
        if self.near_links is None:
            near = []
            for r in recs:
                coords = [float(r[k]) for k in ("StartLat", "StartLon", "EndLat", "EndLon")]
                dmin = min(point_segment_m(lat, lon, *coords) for lat, lon in self.cams.values())
                if dmin <= NEAR_M:
                    near.append(r)
            self.near_links = {r["LinkID"] for r in near}
            with (self.out / "speedband_links_near.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["LinkID", "RoadName", "RoadCategory", "StartLat", "StartLon", "EndLat", "EndLon"])
                for r in near:
                    w.writerow([r["LinkID"], r["RoadName"], r["RoadCategory"], r["StartLat"], r["StartLon"], r["EndLat"], r["EndLon"]])
        p = self.out / "speedbands" / f"sb_{datetime.now():%Y%m%d_%H%M%S}.csv.gz"
        with gzip.open(p, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["LinkID", "SpeedBand", "MinimumSpeed", "MaximumSpeed"])
            n = 0
            for r in recs:
                if r["LinkID"] in self.near_links:
                    w.writerow([r["LinkID"], r["SpeedBand"], r["MinimumSpeed"], r["MaximumSpeed"]])
                    n += 1
        return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--until", required=True, help="local time YYYYmmddHHMM")
    ap.add_argument("--frame-every", type=int, default=60)
    ap.add_argument("--incident-every", type=int, default=120)
    ap.add_argument("--speed-every", type=int, default=300)
    a = ap.parse_args(argv)
    c = Collector(a.out)
    last = {"f": 0.0, "i": 0.0, "s": 0.0}
    log(f"collector start, until {a.until}")
    while datetime.now().strftime("%Y%m%d%H%M") < a.until:
        now = time.time()
        for k, every, fn in (("f", a.frame_every, c.frames), ("i", a.incident_every, c.incidents),
                             ("s", a.speed_every, c.speedbands)):
            if now - last[k] >= every and (k != "s" or c.cams):
                last[k] = now
                try:
                    n = fn()
                    log(f"{fn.__name__}: {n}")
                except Exception:
                    log(f"{fn.__name__} error: {traceback.format_exc(limit=1).splitlines()[-1]}")
        time.sleep(5)
    log("collector done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
