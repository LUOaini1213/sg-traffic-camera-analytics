"""Attach held-out test metrics to the MLflow run and register the fine-tuned model.

Usage: python scripts/register_model.py RUN_DIR_NAME [RUN_DIR_NAME ...] --champion RUN_DIR_NAME
Reads results/detection_eval.json (keys "ft-<run dir name>"), logs the test metrics to each
run, registers every run as a version of MODEL_NAME, and sets the alias "champion" on the
chosen one. Writes results/mlflow_registry.json (a plain-text copy of what MLflow holds).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import DATA, EXPERIMENT, mlflow_setup  # noqa: E402

MODEL_NAME = "sg-traffic-vehicle-detector"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--champion", required=True)
    a = ap.parse_args()
    mlflow = mlflow_setup()
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    ev = json.loads((ROOT / "results" / "detection_eval.json").read_text())
    out = {"experiment": EXPERIMENT, "model": MODEL_NAME, "versions": []}
    try:
        client.create_registered_model(MODEL_NAME, description="YOLO11n fine-tuned on reviewed LTA camera frames")
    except Exception:
        pass
    for name in a.runs:
        run_id = (DATA / "runs" / name / "mlflow_run_id.txt").read_text().strip()
        m = ev["models"][f"ft-{name}"]
        metrics = {"test_mAP50": m["mAP50"], "test_mAP50_95": m["mAP50_95"], "test_count_mae": m["count_mae"],
                   **{f"test_AP50_{k}": v for k, v in m["AP50"].items()}}
        for k, v in metrics.items():
            client.log_metric(run_id, k, v)
        mv = client.create_model_version(MODEL_NAME, source=f"runs:/{run_id}/weights", run_id=run_id,
                                         tags={"run_name": name})
        if name == a.champion:
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        out["versions"].append({"version": mv.version, "run_name": name, "run_id": run_id,
                                "alias": "champion" if name == a.champion else None, **metrics})
    (ROOT / "results" / "mlflow_registry.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
