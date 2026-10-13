"""Box utilities shared by labelling, evaluation and counting. Boxes are xyxy pixels."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy class-agnostic NMS; returns kept indices in descending score order."""
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep: list[int] = []
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order])[0]
        order = [j for j, v in zip(order, ious) if v <= iou_thr]
    return keep


def size_ok(box, width: int, min_side_frac: float) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    return min(w, h) >= min_side_frac * width


def to_yolo(box, cls: int, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def from_yolo(line: str, w: int, h: int) -> tuple[int, list[float]]:
    c, cx, cy, bw, bh = line.split()
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(c), [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
