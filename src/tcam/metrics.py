"""Detection metrics written out explicitly so they can be bootstrapped by camera.

Matching is done once per image (greedy by score, IoU >= thr, same class, each label used
once), so a bootstrap only has to resample images and re-integrate precision/recall.
AP uses the COCO 101-point definition (as in pycocotools).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boxes import iou_matrix


@dataclass
class ImageMatch:
    """Per-image, per-class detection scores, TP flags and label counts."""
    scores: dict[int, np.ndarray] = field(default_factory=dict)
    tp: dict[int, np.ndarray] = field(default_factory=dict)
    n_gt: dict[int, int] = field(default_factory=dict)


def match_image(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, classes, iou_thr: float = 0.5) -> ImageMatch:
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, float).reshape(-1)
    pred_cls = np.asarray(pred_cls, int).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls, int).reshape(-1)
    m = ImageMatch()
    for c in classes:
        pi = np.where(pred_cls == c)[0]
        gi = np.where(gt_cls == c)[0]
        order = pi[np.argsort(-pred_scores[pi], kind="stable")]
        ious = iou_matrix(pred_boxes[order], gt_boxes[gi])
        used = np.zeros(len(gi), bool)
        tp = np.zeros(len(order), bool)
        for k in range(len(order)):
            if len(gi) == 0:
                break
            cand = np.where(~used & (ious[k] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                used[j] = True
                tp[k] = True
        m.scores[c] = pred_scores[order]
        m.tp[c] = tp
        m.n_gt[c] = int(len(gi))
    return m


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: mean over recall r in {0, 0.01, ..., 1} of the best precision
    reached at recall >= r (0 where that recall is never reached)."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp, bool)[o].astype(float)
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    envelope = np.flip(np.maximum.accumulate(np.flip(precision)))
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs - 1e-12, side="left")
    p = np.where(idx < len(recall), envelope[np.minimum(idx, len(recall) - 1)], 0.0)
    return float(p.mean())


def per_class_ap(matches: list[ImageMatch], classes) -> dict[int, float]:
    out = {}
    for c in classes:
        s = np.concatenate([m.scores.get(c, np.zeros(0)) for m in matches]) if matches else np.zeros(0)
        t = np.concatenate([m.tp.get(c, np.zeros(0, bool)) for m in matches]) if matches else np.zeros(0, bool)
        n = sum(m.n_gt.get(c, 0) for m in matches)
        out[c] = average_precision(s, t, n)
    return out


def mean_ap(aps: dict[int, float]) -> float:
    v = [a for a in aps.values() if not np.isnan(a)]
    return float(np.mean(v)) if v else float("nan")


def precision_recall_at(matches: list[ImageMatch], classes, conf: float) -> tuple[float, float]:
    tp = fp = n = 0
    for m in matches:
        for c in classes:
            keep = m.scores.get(c, np.zeros(0)) >= conf
            t = m.tp.get(c, np.zeros(0, bool))[keep]
            tp += int(t.sum()); fp += int((~t).sum()); n += m.n_gt.get(c, 0)
    return tp / max(tp + fp, 1), tp / max(n, 1)


def cluster_bootstrap(matches_by_group: dict[str, list[list[ImageMatch]]], classes, n_boot: int = 2000,
                      seed: int = 0) -> np.ndarray:
    """Resample groups (cameras) with replacement. matches_by_group[g] is a list of
    per-model match lists for that group's images: [model0_matches, model1_matches, ...].
    Returns array (n_boot, n_models) of mAP values."""
    rng = np.random.default_rng(seed)
    groups = sorted(matches_by_group)
    n_models = len(next(iter(matches_by_group.values())))
    out = np.zeros((n_boot, n_models))
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        for k in range(n_models):
            ms = [m for i in pick for m in matches_by_group[groups[i]][k]]
            out[b, k] = mean_ap(per_class_ap(ms, classes))
    return out
