"""Fine-tune YOLO11 on the labelled frames and track the run in a local MLflow store.

Usage: python scripts/train.py --model yolo11n.pt --imgsz 1280 --epochs 60 --batch 4 --seed 0
MLflow: sqlite:///TCAM_DATA/mlflow/mlflow.db, artifacts in TCAM_DATA/mlflow/artifacts.
On CUDA out-of-memory the run is retried with half the batch size (GPU is shared).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
EXPERIMENT = "sg-traffic-camera-detection"


def mlflow_setup():
    import mlflow
    (DATA / "mlflow").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(DATA / 'mlflow' / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(DATA / "mlflow" / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def train_once(a, batch: int, name: str):
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False})
    model = YOLO(a.model)
    t0 = time.time()
    model.train(data=str(DATA / "yolo" / "dataset.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=batch,
                workers=2, seed=a.seed, deterministic=False, project=str(DATA / "runs"), name=name, exist_ok=True,
                plots=False, amp=False,  # fp16 is broken on this GTX 1650 (no detections)
                cos_lr=True, close_mosaic=10, patience=0, verbose=False)
    return model, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    import torch
    mlflow = mlflow_setup()
    name = f"{Path(a.model).stem}_{a.imgsz}_e{a.epochs}_s{a.seed}"
    batch = a.batch
    while True:
        try:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({"model": a.model, "imgsz": a.imgsz, "epochs": a.epochs, "batch": batch,
                                   "seed": a.seed, "gpu": torch.cuda.get_device_name(0)})
                model, secs = train_once(a, batch, name)
                rdir = DATA / "runs" / name
                with (rdir / "results.csv").open() as f:
                    for r in csv.DictReader(f):
                        r = {k.strip(): v for k, v in r.items()}
                        step = int(float(r["epoch"]))
                        mlflow.log_metrics({k.replace("(B)", "").replace("/", "_"): float(v) for k, v in r.items()
                                            if k != "epoch" and v.strip()}, step=step)
                mlflow.log_metric("train_seconds", secs)
                mlflow.log_artifact(str(rdir / "weights" / "best.pt"), artifact_path="weights")
                mlflow.log_artifact(str(rdir / "args.yaml"))
                print("run_id", run.info.run_id, "weights", rdir / "weights" / "best.pt", f"{secs:.0f}s")
                (DATA / "runs" / name / "mlflow_run_id.txt").write_text(run.info.run_id)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1:
                raise
            batch //= 2
            print(f"CUDA OOM, retrying with batch {batch}", flush=True)


if __name__ == "__main__":
    main()
