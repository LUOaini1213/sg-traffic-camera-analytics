"""Model pre-annotation for the labelling set.

Teacher: COCO-pretrained YOLO11x, run on the full frame (imgsz 1920) and on 2x2 overlapping
tiles (each at imgsz 1280), merged with class-agnostic NMS. Only COCO car/bus/truck/motorcycle
are kept. Output: data/pre/{frame_id}.json (committed) with numbered boxes for visual review.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.boxes import nms, size_ok  # noqa: E402
from tcam.classes import COCO_TO_OURS, MIN_SIDE_FRAC  # noqa: E402

DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
CONF, NMS_IOU = 0.20, 0.55


def tiles(w: int, h: int, overlap: float = 0.2):
    tw, th = int(w / (2 - overlap)), int(h / (2 - overlap))
    for y in (0, h - th):
        for x in (0, w - tw):
            yield x, y, x + tw, y + th


def detect(model, img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, h = img.size
    B, S, C = [], [], []

    def run(im, ox, oy, imgsz):
        r = model.predict(im, imgsz=imgsz, conf=CONF, iou=0.6, classes=list(COCO_TO_OURS), verbose=False, half=False)[0]  # fp16 returns no boxes on this GTX 1650
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            B.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]); S.append(float(s)); C.append(COCO_TO_OURS[int(c)])

    run(img, 0, 0, 1920 if w >= 1280 else 1280)
    if w >= 1280:
        for x1, y1, x2, y2 in tiles(w, h):
            run(img.crop((x1, y1, x2, y2)), x1, y1, 1280)
    B, S, C = np.array(B).reshape(-1, 4), np.array(S), np.array(C, int)
    keep = nms(B, S, NMS_IOU)
    keep = [i for i in keep if size_ok(B[i], w, MIN_SIDE_FRAC)]
    return B[keep], S[keep], C[keep]


def main() -> None:
    from ultralytics import YOLO
    model = YOLO("yolo11x.pt")
    rows = [r for r in csv.DictReader((ROOT / "data" / "frames_manifest.csv").open()) if int(r["width"]) >= 640]
    out = ROOT / "data" / "pre"
    out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        p = out / f"{r['frame_id']}.json"
        if p.exists():
            continue
        img = Image.open(DATA / "archive" / "images" / f"{r['frame_id']}.jpg").convert("RGB")
        B, S, C = detect(model, img)
        p.write_text(json.dumps({"frame_id": r["frame_id"], "width": img.size[0], "height": img.size[1],
                                 "boxes": [{"id": k, "cls": int(c), "conf": round(float(s), 3),
                                            "xyxy": [round(float(v), 1) for v in b]}
                                           for k, (b, s, c) in enumerate(zip(B, S, C))]}))
        if i % 25 == 0:
            print(i, r["frame_id"], len(B), flush=True)


if __name__ == "__main__":
    main()
