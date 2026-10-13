"""Download the frames listed in data/frames_manifest.csv from the public data.gov.sg archive.

Images are written outside the repo (default D:/sg-traffic-camera-data/archive/images,
override with TCAM_DATA) and checked against the MD5 published with each frame.
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.archive import fetch_image  # noqa: E402

DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))


def main() -> None:
    out = DATA / "archive" / "images"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / "data" / "frames_manifest.csv").open()))

    def one(r):
        p = out / f"{r['frame_id']}.jpg"
        if p.exists():
            return "cached"
        p.write_bytes(fetch_image(r["url"], r["md5"]))
        return "ok"

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, rows))
    print({k: res.count(k) for k in set(res)}, "->", out)


if __name__ == "__main__":
    main()
