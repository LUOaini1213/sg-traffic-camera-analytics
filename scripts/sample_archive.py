"""Pick the labelling frames from the data.gov.sg archive (deterministic, seed 13).

Writes data/frames_manifest.csv (camera, time, URL, MD5, rain, day/night). Run once;
the manifest is committed so the same frames can be re-downloaded with fetch_frames.py.
"""
from __future__ import annotations

import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.archive import cameras_at, rainfall_at  # noqa: E402
from tcam.cameras import in_scope  # noqa: E402
from tcam.geo import haversine_m  # noqa: E402

SEED = 13
START, END = datetime(2025, 7, 1), datetime(2026, 6, 29, 23, 0)
N_CANDIDATES, N_TIMES, MAX_RAIN_TIMES, PER_TIME = 140, 76, 22, 5


def main() -> None:
    rng = random.Random(SEED)
    span = int((END - START).total_seconds() // 60)
    cands = sorted(START + timedelta(minutes=rng.randrange(span)) for _ in range(N_CANDIDATES))
    info = []
    for t in cands:
        dt = t.strftime("%Y-%m-%dT%H:%M:00")
        stations, readings = rainfall_at(dt)
        wet = [r["station_id"] for r in readings if (r.get("value") or 0) > 0]
        info.append((dt, stations, {r["station_id"]: r.get("value", "") for r in readings}, len(wet) / max(1, len(readings))))
    rainy = [x for x in info if x[3] >= 0.10]
    dry = [x for x in info if x[3] == 0]
    rng.shuffle(rainy); rng.shuffle(dry)
    chosen = rainy[:MAX_RAIN_TIMES] + dry[: N_TIMES - min(len(rainy), MAX_RAIN_TIMES)]
    chosen.sort(key=lambda x: x[0])
    print(f"candidates {len(info)}, rainy {len(rainy)}, chosen {len(chosen)}")

    order: list[str] = []
    rows = []
    for dt, stations, rain, wet_frac in chosen:
        cams = [c for c in cameras_at(dt) if in_scope(c["location"]["longitude"])]
        by_id = {c["camera_id"]: c for c in cams}
        picked = 0
        while picked < PER_TIME:
            if not order:
                order = sorted(by_id) if not rows else sorted({r["camera"] for r in rows} | set(by_id))
                rng.shuffle(order)
            cam = order.pop()
            c = by_id.get(cam)
            if c is None or any(r["camera"] == cam and r["time"] == dt for r in rows):
                continue
            lat, lon = c["location"]["latitude"], c["location"]["longitude"]
            st = min(stations, key=lambda s: haversine_m(lat, lon, s["location"]["latitude"], s["location"]["longitude"]))
            hour = int(dt[11:13])
            rows.append({
                "frame_id": f"{cam}_{dt[:10].replace('-', '')}_{dt[11:16].replace(':', '')}",
                "camera": cam, "time": dt, "image_time": c["timestamp"], "lat": lat, "lon": lon,
                "url": c["image"], "md5": c["image_metadata"]["md5"],
                "width": c["image_metadata"]["width"], "height": c["image_metadata"]["height"],
                "rain_station": st["id"], "rain_mm_5min": rain.get(st["id"], ""),
                "network_wet_frac": round(wet_frac, 3),
                "period": "day" if 7 <= hour < 19 else "night",
            })
            picked += 1
    out = ROOT / "data" / "frames_manifest.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} frames, {len({r['camera'] for r in rows})} cameras -> {out}")


if __name__ == "__main__":
    main()
