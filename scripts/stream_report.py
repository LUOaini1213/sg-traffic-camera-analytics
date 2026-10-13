"""Summarise a detector log (JSONL written by tcam.stream.detector) into throughput and latency.

  frames            events emitted
  wall_s            first publish -> last state emitted
  throughput_fps    frames / wall_s (meaningful for a replay at full speed)
  model_ms_per_frame  median of (model time per batch / batch size)
  e2e_ms p50/p95    publish -> state emitted (meaningful for the live run, where frames
                    arrive as the cameras refresh and there is no backlog)
  capture_to_state_s p50/p95 (live only): camera capture minute -> state emitted
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def summarise(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ev = [e for e in rows if "skipped_bad_frame" not in e]
    skipped = len(rows) - len(ev)
    pub = [e["emitted"] - e["e2e_ms"] / 1000 for e in ev]
    wall = max(e["emitted"] for e in ev) - min(pub)
    e2e = np.array([e["e2e_ms"] for e in ev])
    per = np.array([e["model_ms_batch"] / e["batch"] for e in ev])
    out = {"frames": len(ev), "skipped_unreadable": skipped, "wall_s": wall, "throughput_fps": len(ev) / wall,
           "model_ms_per_frame_p50": float(np.median(per)),
           "e2e_ms_p50": float(np.percentile(e2e, 50)), "e2e_ms_p95": float(np.percentile(e2e, 95)),
           "queue_ms_p50": float(np.median([e["queue_ms"] for e in ev])),
           "mean_batch": float(np.mean([e["batch"] for e in ev])),
           "cameras": sorted({e["camera"] for e in ev})}
    lag = []
    for e in ev:
        c = e.get("capture") or ""
        if len(c) == 16:  # YYYY-mm-ddTHH:MM (live DataMall capture minute)
            lag.append(e["emitted"] - datetime.fromisoformat(c).timestamp())
    if lag:
        out["capture_to_state_s_p50"] = float(np.percentile(lag, 50))
        out["capture_to_state_s_p95"] = float(np.percentile(lag, 95))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    s = summarise(a.log)
    s["note"] = a.note
    a.out.write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
