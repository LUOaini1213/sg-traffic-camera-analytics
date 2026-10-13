"""Turn the live collector's frames.csv into a replay/validation manifest (adds frame_id).
Writes data/live_frames_manifest.csv (no image content, no links: DataMall links expire)."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))


def main() -> None:
    rows = list(csv.DictReader((DATA / "live" / "frames.csv").open()))
    out = []
    for r in rows:
        stem = Path(r["path"]).stem
        out.append({"frame_id": f"{r['camera']}_{stem}", "camera": r["camera"], "capture": r["capture"],
                    "fetched": r["fetched"], "md5": r["md5"], "lat": r["lat"], "lon": r["lon"], "path": r["path"]})
    with (ROOT / "data" / "live_frames_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(len(out), "live frames")


if __name__ == "__main__":
    sys.exit(main())
