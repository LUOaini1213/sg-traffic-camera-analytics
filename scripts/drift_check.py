"""Drift check: per-camera vehicle-count distributions, reference week vs live window.

Reference = detector counts on the archive week (2026-09-26 .. 10-03); current = counts on the
live window. Because traffic depends strongly on time of day, the check is also run on the
hours that both windows cover. Logs a 'monitoring' run to MLflow and writes results/drift.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tcam.drift import drift_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=ROOT / "results" / "validation_week.series.json")
    ap.add_argument("--cur", type=Path, default=ROOT / "results" / "validation_live.series.json")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    ref = json.loads(a.ref.read_text())
    cur = json.loads(a.cur.read_text())
    cur_hours = {int(x["time"][11:13]) for s in cur.values() for x in s}
    all_ref = {c: [x["total"] for x in s] for c, s in ref.items()}
    all_cur = {c: [x["total"] for x in s] for c, s in cur.items()}
    hm_ref = {c: [x["total"] for x in s if int(x["time"][11:13]) in cur_hours] for c, s in ref.items()}
    out = {"live_hours": sorted(cur_hours),
           "all_hours": drift_report(all_ref, all_cur),
           "matched_hours": drift_report(hm_ref, all_cur)}
    (ROOT / "results" / "drift.json").write_text(json.dumps(out, indent=1))
    for k in ("all_hours", "matched_hours"):
        for cam, r in out[k].items():
            print(k, cam, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()})
    if not a.no_mlflow:
        from train import mlflow_setup
        mlflow = mlflow_setup()
        with mlflow.start_run(run_name="drift-check-week-vs-live"):
            mlflow.set_tag("kind", "monitoring")
            for cam, r in out["matched_hours"].items():
                mlflow.log_metric(f"psi_{cam}", r["psi"])
                mlflow.log_metric(f"ks_{cam}", r["ks"])
            mlflow.log_dict(out, "drift.json")


if __name__ == "__main__":
    main()
