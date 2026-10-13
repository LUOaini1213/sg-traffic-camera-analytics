"""Drift check on per-camera count distributions (reference window vs current window)."""
from __future__ import annotations

import numpy as np


def psi(ref, cur, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with quantile bins taken from the reference window."""
    ref = np.asarray(ref, float); cur = np.asarray(cur, float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([ref.min() - 0.5, ref.max() + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, eps, None); c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_stat(ref, cur) -> float:
    ref = np.sort(np.asarray(ref, float)); cur = np.sort(np.asarray(cur, float))
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    grid = np.concatenate([ref, cur])
    fr = np.searchsorted(ref, grid, side="right") / len(ref)
    fc = np.searchsorted(cur, grid, side="right") / len(cur)
    return float(np.max(np.abs(fr - fc)))


def drift_report(ref: dict[str, list[float]], cur: dict[str, list[float]], psi_alert: float = 0.25) -> dict:
    out = {}
    for cam in sorted(set(ref) & set(cur)):
        p = psi(ref[cam], cur[cam])
        out[cam] = {"n_ref": len(ref[cam]), "n_cur": len(cur[cam]),
                    "mean_ref": float(np.mean(ref[cam])), "mean_cur": float(np.mean(cur[cam])),
                    "psi": p, "ks": ks_stat(ref[cam], cur[cam]), "drift": bool(p > psi_alert)}
    return out
