"""Fail if a number quoted in the README does not match the committed result files.

Each claim is (result file, key path, format). The formatted value must appear verbatim in
README.md. Run in CI so the README cannot drift away from results/*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(obj, path: str):
    for part in path.split("|"):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def fmt(v, spec: str) -> str:
    if spec == "pct1":
        return f"{100 * v:.1f}%"
    if spec == "pct0":
        return f"{100 * v:.0f}%"
    return format(v, spec)


CLAIMS: list[tuple[str, str, str]] = []  # filled from results/readme_claims.json


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claims = json.loads((ROOT / "results" / "readme_claims.json").read_text())
    bad = 0
    for c in claims:
        data = json.loads((ROOT / "results" / c["file"]).read_text())
        s = fmt(get(data, c["path"]), c["fmt"])
        ok = s in readme
        bad += not ok
        print(("ok      " if ok else "MISSING ") + f"{c['file']}:{c['path']} -> {s}")
    print(f"{len(claims) - bad}/{len(claims)} README numbers match the result files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
