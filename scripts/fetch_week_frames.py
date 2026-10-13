"""Archive frames for the in-scope live cameras at the times of an existing speed-band log.

The speed-band snapshots (2026-09-26 .. 2026-10-03, about every 20 min) were collected by the
author's sg-bus-network-monitor sampler. For each snapshot time we take the data.gov.sg frame
current at that time for each in-scope camera, if it is no older than 6 minutes.
Writes data/week_frames_manifest.csv; images go to TCAM_DATA/week/images.
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.archive import cameras_at, fetch_image  # noqa: E402
from tcam.cameras import in_scope  # noqa: E402

DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
SB = Path(os.environ.get("SPEEDBAND_DIR", "C:/Users/LW/sg-bus-network-monitor/data/raw/datamall/speedbands"))
MAX_AGE_S = 360


def main() -> None:
    times = sorted(datetime.strptime(p.name[3:16], "%Y%m%d_%H%M") for p in SB.glob("sb_*.csv.gz"))
    out = DATA / "week" / "images"
    out.mkdir(parents=True, exist_ok=True)

    def per_time(t):
        dt = t.strftime("%Y-%m-%dT%H:%M:00")
        rows = []
        for c in cameras_at(dt):
            if not in_scope(c["location"]["longitude"]):
                continue
            img_t = datetime.fromisoformat(c["timestamp"]).replace(tzinfo=None)
            age = (t - img_t).total_seconds()
            if not 0 <= age <= MAX_AGE_S:
                continue
            fid = f"{c['camera_id']}_{t:%Y%m%d_%H%M}"
            p = out / f"{fid}.jpg"
            if not p.exists():
                p.write_bytes(fetch_image(c["image"], c["image_metadata"]["md5"]))
            rows.append({"frame_id": fid, "camera": c["camera_id"], "snapshot": t.strftime("%Y-%m-%dT%H:%M"),
                         "image_time": c["timestamp"], "age_s": int(age), "lat": c["location"]["latitude"],
                         "lon": c["location"]["longitude"], "url": c["image"], "md5": c["image_metadata"]["md5"]})
        return rows

    with ThreadPoolExecutor(6) as ex:
        rows = [r for rs in ex.map(per_time, times) for r in rs]
    with (ROOT / "data" / "week_frames_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(times)} snapshot times, {len(rows)} frames")


if __name__ == "__main__":
    main()
