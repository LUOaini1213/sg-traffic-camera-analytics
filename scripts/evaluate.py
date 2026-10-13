"""Evaluate detectors on the held-out test cameras with camera-level bootstrap intervals.

Models:
  pre-annotation : the YOLO11x tiled pre-labels (data/pre), i.e. what the labels looked like before review
  coco-*         : COCO-pretrained YOLO11 with car/bus/truck/motorcycle mapped to our classes
  ft-*           : fine-tuned weights (one entry per seed)
Writes results/detection_eval.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.boxes import from_yolo, size_ok  # noqa: E402
from tcam.classes import COCO_TO_OURS, MIN_SIDE_FRAC, NAMES  # noqa: E402
from tcam.metrics import cluster_bootstrap, match_image, mean_ap, per_class_ap, precision_recall_at  # noqa: E402

DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
CLASSES = list(range(len(NAMES)))
IOUS = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]


def load_gt(fid: str, w: int, h: int):
    lines = (ROOT / "data" / "labels" / f"{fid}.txt").read_text().split("\n")
    parsed = [from_yolo(l, w, h) for l in lines if l.strip()]
    return [b for _, b in parsed], [c for c, _ in parsed]


def predict(weights: str, imgsz: int, coco: bool, frames: list[str], split: str = "test"):
    from ultralytics import YOLO
    m = YOLO(weights)
    out = {}
    for fid in frames:
        img = Image.open(DATA / "yolo" / "images" / split / f"{fid}.jpg").convert("RGB")
        r = m.predict(img, imgsz=imgsz, conf=0.001, iou=0.6, max_det=600, verbose=False, half=False,  # fp16 returns no boxes on this GTX 1650
                      classes=list(COCO_TO_OURS) if coco else None)[0]
        b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int)
        if coco:
            c = np.array([COCO_TO_OURS[int(x)] for x in c], int)
        out[fid] = (b, s, c, img.size)
    return out


def pre_as_predictions(frames: list[str], sizes: dict):
    """Pre-annotation boxes (native resolution) rescaled to the evaluation image size."""
    out = {}
    for fid in frames:
        p = json.loads((ROOT / "data" / "pre" / f"{fid}.json").read_text())
        w, h = sizes[fid]
        b = np.array([x["xyxy"] for x in p["boxes"]], float).reshape(-1, 4) * (w / p["width"])
        out[fid] = (b, np.array([x["conf"] for x in p["boxes"]]), np.array([x["cls"] for x in p["boxes"]], int),
                    (w, h))
    return out


def score(preds, frames, gts, conf_count: float):
    per_iou = {}
    matches50 = {}
    for thr in IOUS:
        ms = {}
        for fid in frames:
            b, s, c, (w, h) = preds[fid]
            keep = [i for i in range(len(b)) if size_ok(b[i], w, MIN_SIDE_FRAC)]
            gb, gc = gts[fid]
            ms[fid] = match_image(b[keep], s[keep], c[keep], gb, gc, CLASSES, iou_thr=thr)
        per_iou[thr] = mean_ap(per_class_ap(list(ms.values()), CLASSES))
        if thr == 0.5:
            matches50 = ms
    aps = per_class_ap(list(matches50.values()), CLASSES)
    p, r = precision_recall_at(list(matches50.values()), CLASSES, conf_count)
    err = []
    for fid in frames:
        b, s, c, (w, h) = preds[fid]
        n = sum(1 for i in range(len(b)) if s[i] >= conf_count and size_ok(b[i], w, MIN_SIDE_FRAC))
        err.append(n - len(gts[fid][1]))
    err = np.array(err, float)
    agn = []
    for fid in frames:
        b, s_, c, (w, h) = preds[fid]
        keep = [i for i in range(len(b)) if size_ok(b[i], w, MIN_SIDE_FRAC)]
        gb, gc = gts[fid]
        agn.append(match_image(b[keep], s_[keep], np.zeros(len(keep), int), gb, [0] * len(gc), [0], iou_thr=0.5))
    ap_agn = per_class_ap(agn, [0])[0]
    return matches50, err, agn, {
        "mAP50": mean_ap(aps), "mAP50_95": float(np.mean(list(per_iou.values()))),
        "AP50": {NAMES[k]: v for k, v in aps.items()},
        f"precision@{conf_count}": p, f"recall@{conf_count}": r,
        "count_mae": float(np.abs(err).mean()), "count_bias": float(err.mean()),
        "AP50_class_agnostic": ap_agn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", nargs="*", default=[], help="name=weights.pt:imgsz")
    ap.add_argument("--coco", nargs="*", default=["yolo11n.pt:1280", "yolo11n.pt:640"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/detection_eval.json")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader((ROOT / "data" / "split.csv").open()) if r["split"] == a.split]
    frames = [r["frame_id"] for r in rows]
    period = {r["frame_id"]: r["period"] for r in rows}
    sizes = {}
    gts = {}
    for fid in frames:
        w, h = Image.open(DATA / "yolo" / "images" / a.split / f"{fid}.jpg").size
        sizes[fid] = (w, h)
        gts[fid] = load_gt(fid, w, h)
    models = {"pre-annotation (YOLO11x tiled)": pre_as_predictions(frames, sizes)}
    for spec in a.coco:
        w, sz = spec.rsplit(":", 1)
        models[f"coco-{Path(w).stem}@{sz}"] = predict(w, int(sz), True, frames, a.split)
    for spec in a.ft:
        name, rest = spec.split("=", 1)
        w, sz = rest.rsplit(":", 1)
        models[name] = predict(w, int(sz), False, frames, a.split)

    result = {"split": a.split, "frames": len(frames), "cameras": len({f.split("_")[0] for f in frames}),
              "labels": {NAMES[c]: sum(gc.count(c) for _, gc in gts.values()) for c in CLASSES},
              "models": {}, "bootstrap": {}}
    allm, errs, agns = {}, {}, {}
    for name, preds in models.items():
        ms, err, agn, summary = score(preds, frames, gts, a.conf)
        errs[name] = dict(zip(frames, err))
        agns[name] = dict(zip(frames, agn))
        for per in ("day", "night"):
            sub = [ms[f] for f in frames if period[f] == per]
            summary[f"mAP50_{per}"] = mean_ap(per_class_ap(sub, CLASSES))
        result["models"][name] = summary
        allm[name] = ms
        print(f"{name:40s} mAP50 {summary['mAP50']:.3f}  " +
              " ".join(f"{k} {v:.3f}" for k, v in summary["AP50"].items()), flush=True)
    names = list(models)
    groups: dict[str, list] = {}
    for fid in frames:
        groups.setdefault(fid.split("_")[0], [[] for _ in names])
        for k, n in enumerate(names):
            groups[fid.split("_")[0]][k].append(allm[n][fid])
    boot = cluster_bootstrap(groups, CLASSES, n_boot=a.boot, seed=0)
    for k, n in enumerate(names):
        result["bootstrap"][n] = {"mAP50_ci95": [float(np.nanpercentile(boot[:, k], 2.5)),
                                                 float(np.nanpercentile(boot[:, k], 97.5))]}
    # camera-level bootstrap of count MAE and class-agnostic AP50
    rng = np.random.default_rng(1)
    cams = sorted(groups)
    cam_frames = {c: [f for f in frames if f.split("_")[0] == c] for c in cams}
    mae_b = np.zeros((a.boot, len(names)))
    agn_b = np.zeros((a.boot, len(names)))
    for bi in range(a.boot):
        pick = [cams[i] for i in rng.integers(0, len(cams), len(cams))]
        fs = [f for c in pick for f in cam_frames[c]]
        for k, n in enumerate(names):
            mae_b[bi, k] = np.mean([abs(errs[n][f]) for f in fs])
            agn_b[bi, k] = per_class_ap([agns[n][f] for f in fs], [0])[0]
    for k, n in enumerate(names):
        result["bootstrap"][n]["count_mae_ci95"] = [float(np.percentile(mae_b[:, k], 2.5)), float(np.percentile(mae_b[:, k], 97.5))]
        result["bootstrap"][n]["AP50_class_agnostic_ci95"] = [float(np.percentile(agn_b[:, k], 2.5)), float(np.percentile(agn_b[:, k], 97.5))]
    base = [n for n in names if n.startswith("coco-") and n.endswith("@1280")]
    if base:
        b = names.index(base[0])
        for k, n in enumerate(names):
            if n.startswith("ft-"):
                d = boot[:, k] - boot[:, b]
                result["bootstrap"][n]["delta_vs_" + base[0]] = {
                    "mean": float(np.nanmean(d)), "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                    "share_of_resamples_above_0": float(np.mean(d > 0))}
                dm = mae_b[:, k] - mae_b[:, b]
                result["bootstrap"][n]["count_mae_delta_vs_" + base[0]] = {
                    "mean": float(dm.mean()), "ci95": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]}
                da = agn_b[:, k] - agn_b[:, b]
                result["bootstrap"][n]["AP50_class_agnostic_delta_vs_" + base[0]] = {
                    "mean": float(da.mean()), "ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]}
    Path(ROOT / a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["bootstrap"], indent=1))


if __name__ == "__main__":
    main()
