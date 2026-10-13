"""Consume frames, detect vehicles, publish per-camera counts and congestion flags.

Each output event (JSON) carries counts per class, the rolling median, the camera threshold,
the congestion flag and timings: queue wait (publish -> picked up), model time per batch,
and end-to-end (publish -> state produced). Events are also appended to --log (JSONL).
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from confluent_kafka import Consumer, Producer
from PIL import Image

from ..counting import count
from ..flags import CameraFlag
from . import BOOTSTRAP, FRAMES, STATE


class Detector:
    def __init__(self, weights: str, imgsz: int, conf: float, thresholds: dict[str, float], default_threshold: float,
                 device: str = "0", half: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device, self.half = imgsz, conf, device, half
        self.thresholds, self.default = thresholds, default_threshold
        self.flags: dict[str, CameraFlag] = {}

    def infer(self, imgs: list[Image.Image]):
        rs = self.model.predict(imgs, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                half=self.half, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)) for r in rs]

    def state(self, cam: str, width: int, det) -> dict:
        b, s, c = det
        cnt = count(b, s, c, width, conf=self.conf)
        f = self.flags.setdefault(cam, CameraFlag(self.thresholds.get(cam, self.default)))
        return {"camera": cam, "counts": cnt, **f.update(cnt["total"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--thresholds", type=Path, help="JSON {camera: threshold}")
    ap.add_argument("--default-threshold", type=float, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--group", default="tcam-detector")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--idle-exit", type=float, default=0, help="exit after this many idle seconds (0 = never)")
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    th = json.loads(a.thresholds.read_text()) if a.thresholds else {}
    det = Detector(a.weights, a.imgsz, a.conf, th, a.default_threshold, a.device, half=False)  # fp16 returns no boxes on this GTX 1650
    det.infer([Image.new("RGB", (1920, 1080))])  # warm-up
    cons = Consumer({"bootstrap.servers": a.bootstrap, "group.id": a.group, "auto.offset.reset": "earliest",
                     "enable.auto.commit": True, "fetch.message.max.bytes": 4_000_000})
    cons.subscribe([FRAMES])
    prod = Producer({"bootstrap.servers": a.bootstrap, "linger.ms": 5})
    a.log.parent.mkdir(parents=True, exist_ok=True)
    last = time.time()
    n = 0
    with a.log.open("a") as log:
        while True:
            msgs = [m for m in cons.consume(num_messages=a.batch, timeout=1.0) if m.error() is None]
            if not msgs:
                if a.idle_exit and time.time() - last > a.idle_exit:
                    break
                continue
            t_pick = time.time_ns()
            imgs, meta = [], []
            for m in msgs:
                h = {k: v.decode() for k, v in (m.headers() or [])}
                imgs.append(Image.open(io.BytesIO(m.value())).convert("RGB"))
                meta.append(h)
            t0 = time.perf_counter()
            dets = det.infer(imgs)
            model_ms = (time.perf_counter() - t0) * 1000
            for img, h, d in zip(imgs, meta, dets):
                st = det.state(h["camera"], img.size[0], d)
                t_out = time.time_ns()
                st.update({"frame_id": h.get("frame_id"), "capture": h.get("capture"), "batch": len(imgs),
                           "queue_ms": (t_pick - int(h["published_ns"])) / 1e6,
                           "model_ms_batch": model_ms,
                           "e2e_ms": (t_out - int(h["published_ns"])) / 1e6,
                           "emitted": t_out / 1e9})
                prod.produce(STATE, key=h["camera"], value=json.dumps(st).encode())
                log.write(json.dumps(st) + "\n")
                n += 1
            prod.poll(0)
            log.flush()
            last = time.time()
    prod.flush(10)
    cons.close()
    print(f"processed {n} frames", flush=True)


if __name__ == "__main__":
    main()
