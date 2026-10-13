"""Turn model pre-annotations plus recorded review corrections into final labels."""
from __future__ import annotations

from .boxes import size_ok
from .classes import MIN_SIDE_FRAC


def apply_review(pre: dict, review: dict) -> list[tuple[int, list[float]]]:
    """pre: {"width", "boxes": [{"id", "cls", "xyxy"}]}.
    review: {"remove": [id], "relabel": {id: cls}, "adjust": {id: xyxy}, "add": [{"cls", "xyxy"}]}.
    Unknown ids in a review are an error (they mean the review was made on other boxes)."""
    by_id = {b["id"]: b for b in pre["boxes"]}
    remove = {int(i) for i in review.get("remove", [])}
    relabel = {int(k): int(v) for k, v in review.get("relabel", {}).items()}
    adjust = {int(k): v for k, v in review.get("adjust", {}).items()}
    unknown = (remove | set(relabel) | set(adjust)) - set(by_id)
    if unknown:
        raise KeyError(f"review refers to unknown box ids {sorted(unknown)}")
    if remove & (set(relabel) | set(adjust)):
        raise ValueError("a box cannot be both removed and edited")
    out = []
    for i, b in sorted(by_id.items()):
        if i in remove:
            continue
        out.append((relabel.get(i, int(b["cls"])), [float(v) for v in adjust.get(i, b["xyxy"])]))
    for a in review.get("add", []):
        x1, y1, x2, y2 = (float(v) for v in a["xyxy"])
        out.append((int(a["cls"]), [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]))
    w = pre["width"]
    return [(c, b) for c, b in out if size_ok(b, w, MIN_SIDE_FRAC)]


def edit_stats(pre: dict, review: dict) -> dict:
    return {"pre_boxes": len(pre["boxes"]), "removed": len(review.get("remove", [])),
            "relabelled": len(review.get("relabel", {})), "adjusted": len(review.get("adjust", {})),
            "added": len(review.get("add", []))}
