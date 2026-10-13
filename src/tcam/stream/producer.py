"""Publish camera frames to the frames topic.

live   : poll DataMall Traffic-Imagesv2 every --every seconds, publish frames not seen before.
replay : publish frames listed in a manifest (data.gov.sg archive) as fast as possible or at --rate.
Message key = camera id; value = JPEG bytes; headers carry metadata and the publish time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

from confluent_kafka import Producer

from . import BOOTSTRAP, FRAMES


def _headers(cam: str, capture: str, frame_id: str) -> list[tuple[str, bytes]]:
    return [("camera", cam.encode()), ("capture", capture.encode()), ("frame_id", frame_id.encode()),
            ("published_ns", str(time.time_ns()).encode())]


def make_producer(bootstrap: str = BOOTSTRAP) -> Producer:
    return Producer({"bootstrap.servers": bootstrap, "message.max.bytes": 4_000_000, "linger.ms": 5})


def live(p: Producer, every: int, until: str, scope) -> int:
    from datetime import datetime

    from .. import datamall
    from ..collect import capture_time
    seen: dict[str, str] = {}
    n = 0
    while datetime.now().strftime("%Y%m%d%H%M") < until:
        t0 = time.time()
        try:
            for v in datamall.traffic_images():
                if not scope(float(v["Longitude"])):
                    continue
                cam = str(v["CameraID"])
                b = datamall.download(v["ImageLink"])
                md5 = hashlib.md5(b).hexdigest()
                if seen.get(cam) == md5:
                    continue
                seen[cam] = md5
                cap = capture_time(v["ImageLink"]) or ""
                p.produce(FRAMES, key=cam, value=b, headers=_headers(cam, cap, f"{cam}_{md5[:8]}"))
                n += 1
            p.flush(10)
        except Exception as e:  # keep polling through transient API errors
            print("poll error", type(e).__name__, flush=True)
        time.sleep(max(0.0, every - (time.time() - t0)))
    return n


def replay(p: Producer, manifest: Path, images: Path, rate: float, limit: int | None) -> int:
    rows = list(csv.DictReader(manifest.open()))
    rows.sort(key=lambda r: (r.get("snapshot") or r.get("time") or r.get("capture"), r["camera"]))
    if limit:
        rows = rows[:limit]
    n = 0
    for r in rows:
        src = images / r["path"] if r.get("path") else images / f"{r['frame_id']}.jpg"
        b = src.read_bytes()
        p.produce(FRAMES, key=r["camera"], value=b,
                  headers=_headers(r["camera"], r.get("image_time") or r.get("capture", ""), r["frame_id"]))
        n += 1
        p.poll(0)
        if rate > 0:
            time.sleep(1.0 / rate)
    p.flush(30)
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "replay"])
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--until", default="209912312359")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    p = make_producer(a.bootstrap)
    if a.mode == "live":
        from ..cameras import in_scope
        n = live(p, a.every, a.until, in_scope)
    else:
        n = replay(p, a.manifest, a.images, a.rate, a.limit)
    print(f"published {n} frames", flush=True)


if __name__ == "__main__":
    main()
