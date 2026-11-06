"""Rebuild committed results from committed inputs only (no images, no weights, no GPU, no keys).

Each test recomputes a results file, or part of one, with the same code the scripts use and
asserts it equals what is committed. Together with scripts/check_readme.py this ties the README
numbers back to data/ rather than only to results/*.json.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import tcam  # noqa: F401  (import first so TCAM_SRC from conftest decides which package is tested)
from tcam.boxes import to_yolo
from tcam.classes import NAMES
from tcam.flags import calibrate
from tcam.labels import apply_review, edit_stats
from tcam.metrics import cluster_bootstrap
from tcam.split import split_cameras

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def same(a, b, path="") -> list[str]:
    """Differences between two JSON-like values; floats equal to 1e-12 relative, NaN == NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one side" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += same(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if math.isnan(a) and math.isnan(b):
            return []
        return [] if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def normalise(obj):
    return json.loads(json.dumps(obj, default=float))


# --- labels, label statistics and split -------------------------------------------------------

def kept_frames():
    rows = [r for r in csv.DictReader((DATA / "frames_manifest.csv").open(newline="")) if int(r["width"]) >= 640]
    rows = [r for r in rows if (DATA / "review" / f"{r['frame_id']}.json").exists()]
    excluded = [r["frame_id"] for r in rows if read_json(DATA / "review" / f"{r['frame_id']}.json").get("exclude")]
    return [r for r in rows if r["frame_id"] not in excluded], excluded


def test_labels_and_label_stats_rebuild_from_pre_and_review():
    rows, excluded = kept_frames()
    split = {r["frame_id"]: r["split"] for r in csv.DictReader((DATA / "split.csv").open(newline=""))}
    stats = read_json(RESULTS / "label_stats.json")
    edits = Counter()
    boxes = {p: Counter() for p in ("train", "val", "test")}
    frames = Counter()
    for r in rows:
        fid = r["frame_id"]
        pre = read_json(DATA / "pre" / f"{fid}.json")
        rev = read_json(DATA / "review" / f"{fid}.json")
        labels = apply_review(pre, rev)
        edits.update(edit_stats(pre, rev))
        lines = [to_yolo(b, c, pre["width"], pre["height"]) for c, b in labels]
        assert (DATA / "labels" / f"{fid}.txt").read_text() == "\n".join(lines) + ("\n" if lines else ""), fid
        boxes[split[fid]].update(NAMES[c] for c, _ in labels)
        frames[split[fid]] += 1
    assert len(rows) == 330
    assert sorted(p.stem for p in (DATA / "labels").glob("*.txt")) == sorted(r["frame_id"] for r in rows)
    assert dict(edits) == stats["edits"]
    assert {p: dict(c) for p, c in boxes.items()} == stats["boxes"]
    assert dict(frames) == stats["frames"]
    assert excluded == stats["excluded_by_reviewer"]
    final = sum(sum(c.values()) for c in boxes.values())
    assert final == stats["edits"]["pre_boxes"] - stats["edits"]["removed"] + stats["edits"]["added"]


def test_camera_split_rebuilds():
    bd = load_script("build_dataset")
    rows, _ = kept_frames()
    cams = {r["camera"]: (float(r["lat"]), float(r["lon"])) for r in rows}
    fpc = Counter(r["camera"] for r in rows)
    split = split_cameras(cams, fpc, test_frac=0.2, val_frac=0.1, force_train=bd.LIVE_CAMERAS, seed=13)
    committed = {r["frame_id"]: r["split"] for r in csv.DictReader((DATA / "split.csv").open(newline=""))}
    assert {r["frame_id"]: split[r["camera"]] for r in rows} == committed


# --- flags vs speed bands / incidents, thresholds, drift ----------------------------------------

def test_manifests_match_the_frame_counts_quoted():
    archive = list(csv.DictReader((DATA / "frames_manifest.csv").open(newline="")))
    week = list(csv.DictReader((DATA / "week_frames_manifest.csv").open(newline="")))
    assert len(archive) == 380  # README: fetch_frames.py downloads 380 archive frames
    assert all(len(r["md5"]) == 32 and r["url"].startswith("https://images.data.gov.sg/") for r in archive + week)
    week_frames = sum(c["frames"] for c in read_json(RESULTS / "validation_week.json")["cameras"].values())
    assert len(week) == week_frames == read_json(RESULTS / "stream_week_replay.json")["frames"]


def test_week_thresholds_rebuild():
    vf = load_script("validate_flags")
    series = read_json(RESULTS / "validation_week.series.json")
    th = {c: calibrate([x["total"] for x in s if x["time"] < vf.WEEK_SPLIT_AT]) for c, s in series.items()}
    assert th == read_json(DATA / "thresholds.json")


@pytest.mark.parametrize("mode", ["week", "live"])
def test_validation_results_rebuild_from_series(mode):
    vf = load_script("validate_flags")
    committed = read_json(RESULTS / f"validation_{mode}.json")
    series = read_json(RESULTS / f"validation_{mode}.series.json")
    links = {c: v["links"] for c, v in committed["cameras"].items()}  # needs the road-link file; taken as given
    got = normalise(vf.summarise(series, read_json(DATA / "thresholds.json"), mode, vf.WEEK_SPLIT_AT, links,
                                 incidents=mode == "live"))
    if mode == "week":  # the week file predates the first/last fields the script now adds
        assert (got.pop("first"), got.pop("last")) == ("2026-09-26T19:28", "2026-10-03T00:06")
    assert same(got, committed) == []


def test_live_heavy_traffic_frames_none_flagged():
    cams = read_json(RESULTS / "validation_live.json")["cameras"]
    n = sum(c.get("frames_with_incident_within_1km", {}).get("Heavy Traffic", 0) for c in cams.values())
    flagged = sum(c["frames_with_incident_within_1km"]["Heavy Traffic"] * c["flag_rate_when_incident"]["Heavy Traffic"]
                  for c in cams.values() if "Heavy Traffic" in c.get("frames_with_incident_within_1km", {}))
    assert (n, flagged) == (20, 0)


def test_drift_rebuilds_from_series():
    dc = load_script("drift_check")
    got = dc.build(read_json(RESULTS / "validation_week.series.json"), read_json(RESULTS / "validation_live.series.json"))
    assert same(normalise(got), read_json(RESULTS / "drift.json")) == []


# --- detection: the pre-annotation row needs no model and no image --------------------------------

PRE = "pre-annotation (YOLO11x tiled)"


@pytest.fixture(scope="module")
def pre_row():
    ev = load_script("evaluate")
    bd = load_script("build_dataset")
    rows = [r for r in csv.DictReader((DATA / "split.csv").open(newline="")) if r["split"] == "test"]
    frames = [r["frame_id"] for r in rows]
    sizes, gts = {}, {}
    for fid in frames:
        p = read_json(DATA / "pre" / f"{fid}.json")
        w, h = p["width"], p["height"]
        if w > bd.TRAIN_WIDTH:  # evaluation reads the copies build_dataset.py resized
            w, h = bd.TRAIN_WIDTH, round(h * bd.TRAIN_WIDTH / w)
        sizes[fid] = (w, h)
        gts[fid] = ev.load_gt(fid, w, h)
    preds = ev.pre_as_predictions(frames, sizes)
    ms, _, _, summary = ev.score(preds, frames, gts, 0.25)
    return ev, frames, ms, summary, {r["frame_id"]: r["period"] for r in rows}


def test_pre_annotation_row_rebuilds(pre_row):
    ev, frames, ms, summary, period = pre_row
    for per in ("day", "night"):
        summary[f"mAP50_{per}"] = ev.mean_ap(ev.per_class_ap([ms[f] for f in frames if period[f] == per], ev.CLASSES))
    committed = read_json(RESULTS / "detection_eval.json")
    assert committed["frames"] == len(frames) == 65
    assert same(normalise(summary), committed["models"][PRE]) == []


def test_pre_annotation_bootstrap_interval_rebuilds(pre_row):
    ev, frames, ms, _, _ = pre_row
    groups: dict[str, list] = {}
    for fid in frames:
        groups.setdefault(fid.split("_")[0], [[]])[0].append(ms[fid])
    boot = cluster_bootstrap(groups, ev.CLASSES, n_boot=2000, seed=0)
    ci = [float(np.nanpercentile(boot[:, 0], 2.5)), float(np.nanpercentile(boot[:, 0], 97.5))]
    assert same(ci, read_json(RESULTS / "detection_eval.json")["bootstrap"][PRE]["mAP50_ci95"]) == []
