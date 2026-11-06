"""Download the archive-week frames listed in data/week_frames_manifest.csv.

Reads the committed manifest (it never writes it), fetches each frame from its persistent
data.gov.sg URL and checks the MD5 published with it. Images go to TCAM_DATA/week/images
(default ./data_local/week/images, ignored by git). Needs no key and no speed-band log.

The manifest itself was derived once from a private speed-band log; that step is
scripts/derive_week_manifest.py and is not needed to re-download the frames.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.archive import fetch_image  # noqa: E402

DATA = Path(os.environ.get("TCAM_DATA", ROOT / "data_local"))  # images, YOLO folders, MLflow store (not committed)
MANIFEST = ROOT / "data" / "week_frames_manifest.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=DATA / "week" / "images")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (0 = all)")
    a = ap.parse_args()

    rows = list(csv.DictReader(a.manifest.open(newline="")))
    if not rows:
        print(f"no frames in {a.manifest}", file=sys.stderr)
        return 1
    if a.limit:
        rows = rows[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)

    def one(r):
        p = a.out / f"{r['frame_id']}.jpg"
        if p.exists():
            return "cached"
        p.write_bytes(fetch_image(r["url"], r["md5"]))  # raises on MD5 mismatch
        return "ok"

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, rows))
    print({k: res.count(k) for k in sorted(set(res))}, "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
