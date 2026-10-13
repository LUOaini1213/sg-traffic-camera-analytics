"""Build final YOLO labels from pre-annotations + review corrections, split by camera.

Inputs (committed): data/frames_manifest.csv, data/pre/*.json, data/review/*.json
Outputs (committed): data/labels/*.txt, data/split.csv, results/label_stats.json
Outputs (local):     TCAM_DATA/yolo/{images,labels}/{train,val,test} + dataset.yaml
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.boxes import to_yolo  # noqa: E402
from tcam.classes import NAMES  # noqa: E402
from tcam.labels import apply_review, edit_stats  # noqa: E402
from tcam.split import camera_groups, split_cameras  # noqa: E402

DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
LIVE_CAMERAS = {"2701", "2702", "2704", "4798", "4799"}  # deployed cameras stay in train


def main() -> None:
    rows = [r for r in csv.DictReader((ROOT / "data" / "frames_manifest.csv").open()) if int(r["width"]) >= 640]
    reviewed = [r for r in rows if (ROOT / "data" / "review" / f"{r['frame_id']}.json").exists()]
    if len(reviewed) < len(rows):
        print(f"WARNING: {len(rows) - len(reviewed)} frames not reviewed yet; they are skipped")
    excluded = [r["frame_id"] for r in reviewed
                if json.loads((ROOT / "data" / "review" / f"{r['frame_id']}.json").read_text()).get("exclude")]
    rows = [r for r in reviewed if r["frame_id"] not in excluded]
    cams = {r["camera"]: (float(r["lat"]), float(r["lon"])) for r in rows}
    fpc = Counter(r["camera"] for r in rows)
    split = split_cameras(cams, fpc, test_frac=0.2, val_frac=0.1, force_train=LIVE_CAMERAS, seed=13)

    (ROOT / "data" / "labels").mkdir(parents=True, exist_ok=True)
    out = DATA / "yolo"
    if out.exists():
        shutil.rmtree(out)
    stats = {"edits": Counter(), "boxes": {p: Counter() for p in ("train", "val", "test")},
             "frames": Counter(), "cameras": Counter(), "period": {p: Counter() for p in ("train", "val", "test")}}
    split_rows = []
    for r in rows:
        fid = r["frame_id"]
        pre = json.loads((ROOT / "data" / "pre" / f"{fid}.json").read_text())
        rev = json.loads((ROOT / "data" / "review" / f"{fid}.json").read_text())
        labels = apply_review(pre, rev)
        stats["edits"].update(edit_stats(pre, rev))
        lines = [to_yolo(b, c, pre["width"], pre["height"]) for c, b in labels]
        (ROOT / "data" / "labels" / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        part = split[r["camera"]]
        for sub in ("images", "labels"):
            (out / sub / part).mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / "archive" / "images" / f"{fid}.jpg", out / "images" / part / f"{fid}.jpg")
        shutil.copy(ROOT / "data" / "labels" / f"{fid}.txt", out / "labels" / part / f"{fid}.txt")
        stats["frames"][part] += 1
        stats["period"][part][r["period"]] += 1
        stats["boxes"][part].update(NAMES[c] for c, _ in labels)
        split_rows.append({"frame_id": fid, "camera": fid.split("_")[0], "split": part, "period": r["period"],
                           "network_wet_frac": r["network_wet_frac"]})
    for c, p in split.items():
        stats["cameras"][p] += 1
    (out / "dataset.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    with (ROOT / "data" / "split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
        w.writeheader(); w.writerows(split_rows)
    stats["camera_groups"] = len(camera_groups(cams))
    stats["excluded_by_reviewer"] = excluded
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "label_stats.json").write_text(json.dumps(stats, indent=1, default=dict))
    print(json.dumps(stats, indent=1, default=dict))


if __name__ == "__main__":
    main()
