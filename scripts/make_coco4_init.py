"""Build a 4-class YOLO11 starting checkpoint whose class head is copied from the COCO head.

A plain fine-tune re-initialises the classification convolutions when the class count changes
(80 -> 4), so the model has to relearn 'bus' from 55 training boxes. Here the last 1x1 conv of
each class branch keeps the COCO rows for car, bus, truck and motorcycle (in our class order),
so training starts from the pretrained detector's own class scores.

Usage: python scripts/make_coco4_init.py yolo11n.pt yolo11n_coco4.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcam.classes import COCO_TO_OURS, NAMES  # noqa: E402


def main(src: str, dst: str) -> None:
    from ultralytics import YOLO
    m = YOLO(src)
    det = m.model.model[-1]
    order = [coco for coco, ours in sorted(COCO_TO_OURS.items(), key=lambda kv: kv[1])]  # [2, 5, 7, 3]
    idx = torch.tensor(order)
    for branch in det.cv3:
        conv = branch[-1]
        new = torch.nn.Conv2d(conv.in_channels, len(order), 1, bias=True).to(conv.weight.device)
        with torch.no_grad():
            new.weight.copy_(conv.weight[idx])
            new.bias.copy_(conv.bias[idx])
        branch[-1] = new
    det.nc = len(order)
    det.no = det.nc + det.reg_max * 4
    m.model.nc = len(order)
    m.model.yaml["nc"] = len(order)
    m.model.names = dict(enumerate(NAMES))
    m.save(dst)
    # sanity check: the sliced model must give the same class scores as the COCO model
    test = YOLO(dst)
    print("saved", dst, "nc", test.model.model[-1].nc, test.model.names)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
