"""Plant known bugs in the package one at a time and check that the test suite catches each.

Each mutation replaces one exact snippet (which must occur exactly once). A mutation
"escapes" if the tests still pass with it. Exit code 1 if any mutation escapes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    # counting
    ("counting.py", "if s < conf or", "if s <= conf or", "confidence threshold made exclusive"),
    ("counting.py", "if s < conf or not size_ok(b, width, MIN_SIDE_FRAC):", "if s < conf:", "size floor dropped"),
    ("counting.py", "point_in_polygon((b[0] + b[2]) / 2, b[3], roi)", "point_in_polygon((b[0] + b[2]) / 2, b[1], roi)", "ROI tested at box top"),
    ("counting.py", "xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)", "xin = x1", "edge crossing ignores slope"),
    ("counting.py", 'out["total"] = sum(out[n] for n in NAMES)', 'out["total"] = sum(out[n] for n in NAMES[:3])', "total omits motorcycles"),
    ("counting.py", "out[NAMES[int(c)]] += 1", "out[NAMES[int(c) - 1]] += 1", "class index shifted"),
    # flags
    ("flags.py", "med = float(np.median(self.history))", "med = float(np.mean(self.history))", "median replaced by mean"),
    ("flags.py", "len(self.history) >= self.window and", "len(self.history) >= 1 and", "flag before window is full"),
    ("flags.py", "elif self.flagged and med < self.release * self.threshold:", "elif self.flagged and med < self.threshold:", "hysteresis removed"),
    ("flags.py", "and med >= self.threshold:", "and med > self.threshold:", "threshold made exclusive"),
    ("flags.py", "while len(self.history) > self.window:", "while len(self.history) > self.window + 1:", "window one frame too long"),
    ("flags.py", "return max(float(np.quantile(counts, quantile)), float(floor))", "return min(float(np.quantile(counts, quantile)), float(floor))", "floor applied as ceiling"),
    ("flags.py", "np.quantile(counts, quantile)", "np.quantile(counts, 0.5)", "calibration quantile ignored"),
    # validation / drift
    ("validate.py", "pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n)", "pe = (((tp + fp) * (tp + fn))) / (n * n)", "kappa chance term wrong"),
    ("validate.py", "return float(np.mean(v)) if v else", "return float(np.min(v)) if v else", "speed context uses min"),
    ("validate.py", 'if l["RoadCategory"] not in categories:', "if False:", "road-category filter dropped"),
    ("drift.py", "return float(np.sum((c - r) * np.log(c / r)))", "return float(np.sum((r - c) * np.log(c / r)))", "PSI sign flipped"),
    # metrics
    ("metrics.py", "envelope = np.flip(np.maximum.accumulate(np.flip(precision)))", "envelope = precision", "precision envelope dropped"),
    ("metrics.py", "cand = np.where(~used & (ious[k] >= iou_thr))[0]", "cand = np.where(ious[k] >= iou_thr)[0]", "labels can be matched twice"),
    ("metrics.py", "order = pi[np.argsort(-pred_scores[pi], kind=\"stable\")]", "order = pi", "detections not sorted by score"),
    ("metrics.py", "p = np.where(idx < len(recall), envelope[np.minimum(idx, len(recall) - 1)], 0.0)", "p = envelope[np.minimum(idx, len(recall) - 1)]", "unreached recall not zeroed"),
    # labels
    ("labels.py", "if i in remove:\n            continue", "if False:\n            continue", "review removals ignored"),
    ("labels.py", "out.append((relabel.get(i, int(b[\"cls\"]))", "out.append((int(b[\"cls\"])", "review relabels ignored"),
]


def run_tests(src: Path) -> bool:
    env = dict(os.environ, TCAM_SRC=str(src))
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
                       env=env, capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 0


def main() -> int:
    if not run_tests(ROOT / "src"):
        print("baseline tests fail; fix them first")
        return 2
    escaped = []
    with tempfile.TemporaryDirectory() as td:
        for fname, old, new, desc in MUTATIONS:
            dst = Path(td) / "src"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(ROOT / "src", dst)
            f = dst / "tcam" / fname
            text = f.read_text()
            if text.count(old) != 1:
                print(f"BROKEN MUTATION ({text.count(old)} matches): {fname}: {desc}")
                escaped.append(desc)
                continue
            f.write_text(text.replace(old, new))
            caught = not run_tests(dst)
            print(f"{'caught ' if caught else 'ESCAPED'}  {fname:12s} {desc}")
            if not caught:
                escaped.append(desc)
    print(f"{len(MUTATIONS) - len(escaped)}/{len(MUTATIONS)} mutations caught")
    return 1 if escaped else 0


if __name__ == "__main__":
    sys.exit(main())
