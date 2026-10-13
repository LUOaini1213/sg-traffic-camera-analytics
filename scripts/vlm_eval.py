"""Small local vision-language model as a scene tagger, scored against independent references.

Model: gemma3:4b through a local Ollama server (http://localhost:11434). For each frame the model
returns JSON {"time_of_day": "day|night", "rain": true|false, "traffic": "light|moderate|heavy"}.
References (none of them come from the VLM):
  * time_of_day : capture hour from the archive (07:00-18:59 = day)
  * rain        : the nearest NEA rain gauge reported > 0 mm in that 5-min reading
  * traffic     : reviewed label count in the frame (<=10 light, 11-30 moderate, >30 heavy)
Frames: the 65 held-out test frames (cameras never used in training).
Writes results/vlm_eval.json with per-frame outputs and accuracy / confusion per field.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TCAM_DATA", "D:/sg-traffic-camera-data"))
MODEL = os.environ.get("VLM_MODEL", "gemma3:4b")
PROMPT = ("This is a frame from a Singapore expressway traffic camera. Answer with JSON only, keys: "
          "\"time_of_day\" (\"day\" or \"night\"), \"rain\" (true if it is raining or the road is visibly wet "
          "from current rain, else false), \"traffic\" (\"light\", \"moderate\" or \"heavy\", judged by how many "
          "vehicles are visible on the road).")


def ask(img_path: Path) -> tuple[dict, float]:
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((896, 896))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    body = {"model": MODEL, "prompt": PROMPT, "images": [base64.b64encode(buf.getvalue()).decode()],
            "stream": False, "format": "json", "options": {"temperature": 0}}
    t0 = time.time()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return json.loads(out["response"]), time.time() - t0


def bucket(n: int) -> str:
    return "light" if n <= 10 else ("moderate" if n <= 30 else "heavy")


def main() -> None:
    man = {r["frame_id"]: r for r in csv.DictReader((ROOT / "data" / "frames_manifest.csv").open())}
    test = [r["frame_id"] for r in csv.DictReader((ROOT / "data" / "split.csv").open()) if r["split"] == "test"]
    rows = []
    for fid in test:
        m = man[fid]
        n = sum(1 for l in (ROOT / "data" / "labels" / f"{fid}.txt").read_text().splitlines() if l.strip())
        rain_val = m["rain_mm_5min"]
        ref = {"time_of_day": "day" if 7 <= int(m["time"][11:13]) < 19 else "night",
               "rain": (float(rain_val) > 0) if rain_val not in ("", None) else None,
               "traffic": bucket(n), "labels": n}
        try:
            got, secs = ask(DATA / "archive" / "images" / f"{fid}.jpg")
        except Exception as e:
            got, secs = {"error": type(e).__name__}, None
        rows.append({"frame_id": fid, "ref": ref, "vlm": got, "seconds": secs})
        print(fid, ref, got, f"{secs:.1f}s" if secs else "", flush=True)
    res = {"model": MODEL, "frames": len(rows), "fields": {}}
    for k in ("time_of_day", "rain", "traffic"):
        pairs = [(r["ref"][k], r["vlm"].get(k)) for r in rows if r["ref"][k] is not None and k in r["vlm"]]
        res["fields"][k] = {"n": len(pairs), "accuracy": sum(a == b for a, b in pairs) / max(len(pairs), 1),
                            "majority_baseline": Counter(a for a, _ in pairs).most_common(1)[0][1] / max(len(pairs), 1),
                            "confusion": {f"{a}->{b}": c for (a, b), c in Counter(pairs).items()}}
    secs = [r["seconds"] for r in rows if r["seconds"]]
    res["median_seconds_per_frame"] = sorted(secs)[len(secs) // 2] if secs else None
    res["rows"] = rows
    (ROOT / "results" / "vlm_eval.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
