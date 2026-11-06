"""Derive data/week_frames_manifest.csv from a speed-band log (one-off; already done).

For each speed-band snapshot time (2026-09-26 .. 2026-10-03, about every 20 min) take the
data.gov.sg frame current at that time for each in-scope camera, if it is no older than
6 minutes, and download it. The snapshots came from the author's own DataMall sampler and are
not published, so SPEEDBAND_DIR (or --speed-dir) must point at a folder of sb_YYYYmmdd_HHMM*.csv.gz
files. To re-download the committed frames use scripts/fetch_week_frames.py instead.

The manifest is written to a temporary file and only replaces the committed one when at least
one frame was found, so a missing or empty log cannot wipe it.
"""
from __future__ import annotations

import argparse
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

DATA = Path(os.environ.get("TCAM_DATA", ROOT / "data_local"))  # images, YOLO folders, MLflow store (not committed)
MAX_AGE_S = 360


def snapshot_times(speed_dir: Path) -> list[datetime]:
    return sorted(datetime.strptime(p.name[3:16], "%Y%m%d_%H%M") for p in speed_dir.glob("sb_*.csv.gz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    env = os.environ.get("SPEEDBAND_DIR")
    ap.add_argument("--speed-dir", type=Path, default=Path(env) if env else None, required=not env,
                    help="folder of sb_*.csv.gz speed-band snapshots (or set SPEEDBAND_DIR)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "week_frames_manifest.csv")
    a = ap.parse_args()

    times = snapshot_times(a.speed_dir)
    if not times:
        print(f"no sb_*.csv.gz snapshots in {a.speed_dir}; manifest left unchanged", file=sys.stderr)
        return 1
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
    if not rows:
        print(f"{len(times)} snapshot times but no frames; manifest left unchanged", file=sys.stderr)
        return 1
    tmp = a.out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(a.out)
    print(f"{len(times)} snapshot times, {len(rows)} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
