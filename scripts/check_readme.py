"""Fail if a number quoted in the README does not match the committed result files.

Each claim in results/readme_claims.json names a value and where in README.md it must appear:

  value   "file" + "path" into results/<file> ("|" separates keys, "*" fans out over every
          key or item), optionally reduced with "agg" (sum, len, count_true, max); or
          "source": "pytest" (tests collected) / "mutations" (planted bugs in
          scripts/mutation_check.py), which are counted from the committed code.
  format  "fmt": a format spec (".3f", "+.3f", ",") or pct0 / pct1.
  anchor  "table": {"header": <text in the table's header row>, "row": <first cell>,
          "col": <header cell>, optional "text" with "{}"} -> that exact cell must contain the
          value (or the text with the value filled in); or
          "context": text with "{}" -> the README must contain it with the value filled in.

A value counts as present only as a whole number (0.82 does not match inside 0.824). Every
claim must carry an anchor, so a value cannot pass just because the same number appears
somewhere else in the README. Run in CI so the README cannot drift away from results/*.json.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_all(obj, parts: list[str]) -> list:
    """Leaves reached by the path; "*" fans out, keys missing below a "*" are skipped."""
    if not parts:
        return [obj]
    head, rest = parts[0], parts[1:]
    if head == "*":
        items = obj if isinstance(obj, list) else list(obj.values())
        out = []
        for it in items:
            try:
                out += get_all(it, rest)
            except (KeyError, IndexError, TypeError):
                continue
        return out
    return get_all(obj[int(head)] if isinstance(obj, list) else obj[head], rest)


def get(obj, path: str):
    leaves = get_all(obj, path.split("|"))
    if "*" not in path.split("|"):
        return leaves[0]
    return leaves


AGG = {
    "sum": lambda v: sum(v),
    "len": lambda v: len(v),
    "count_true": lambda v: sum(1 for x in v if x is True),
    "max": lambda v: max(v),
}


def fmt(v, spec: str) -> str:
    if spec == "pct1":
        return f"{100 * v:.1f}%"
    if spec == "pct0":
        return f"{100 * v:.0f}%"
    return format(v, spec)


def collected_tests() -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"^(\d+) tests? collected", r.stdout, re.M)
    if r.returncode != 0 or not m:
        raise RuntimeError("pytest --collect-only failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return int(m.group(1))


def planted_mutations() -> int:
    tree = ast.parse((ROOT / "scripts" / "mutation_check.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MUTATIONS" for t in node.targets):
            return len(node.value.elts)
    raise RuntimeError("MUTATIONS not found in scripts/mutation_check.py")


def value_of(c: dict, cache: dict):
    src = c.get("source")
    if src == "pytest":
        if "pytest" not in cache:
            cache["pytest"] = collected_tests()
        return cache["pytest"]
    if src == "mutations":
        return planted_mutations()
    if c["file"] not in cache:
        cache[c["file"]] = json.loads((ROOT / "results" / c["file"]).read_text(encoding="utf-8"))
    v = get(cache[c["file"]], c["path"])
    return AGG[c["agg"]](v) if "agg" in c else v


def tables(lines: list[str]) -> list[list[list[str]]]:
    """Markdown tables as lists of rows of stripped cells (separator row dropped)."""
    out, cur = [], []
    for line in lines + [""]:
        if line.startswith("|"):
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", x) for x in cells):
                cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    return out


def has_number(text: str, s: str, template: str = "{}") -> bool:
    """template with s filled in occurs in text, and s is not part of a longer number there
    (0.82 is not found in 0.824 or 10.82)."""
    pre, post = template.split("{}", 1)
    post = post.replace("{}", s)
    pattern = ("" if pre else r"(?<![\d.])") + re.escape(pre) + re.escape(s) + ("" if post else r"(?![\d])") + re.escape(post)
    return re.search(pattern, text) is not None


def plain(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def check(c: dict, s: str, readme: str, tabs) -> str | None:
    """None if the claim holds, else the reason it does not."""
    if "table" in c:
        t = c["table"]
        hits = [tb for tb in tabs if t["header"] in " | ".join(tb[0])]
        if len(hits) != 1:
            return f"{len(hits)} tables have a header containing {t['header']!r}"
        tb = hits[0]
        cols = [plain(x) for x in tb[0]]
        if t["col"] not in cols:
            return f"no column {t['col']!r}"
        rows = [r for r in tb[1:] if plain(r[0]) == t["row"]]
        if len(rows) != 1:
            return f"{len(rows)} rows named {t['row']!r}"
        cell = rows[0][cols.index(t["col"])]
        return None if has_number(cell, s, t.get("text", "{}")) else f"cell is {cell!r}"
    if "context" in c:
        return None if has_number(readme, s, c["context"]) else f"{c['context'].replace('{}', s)!r} not found"
    return "claim has no anchor (add 'table' or 'context')"


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tabs = tables(readme.splitlines())
    claims = json.loads((ROOT / "results" / "readme_claims.json").read_text(encoding="utf-8"))
    cache: dict = {}
    bad = 0
    for c in claims:
        name = c.get("source") or f"{c['file']}:{c['path']}" + (f" ({c['agg']})" if "agg" in c else "")
        s = fmt(value_of(c, cache), c["fmt"])
        why = check(c, s, readme, tabs)
        bad += why is not None
        print(("ok      " if why is None else "MISSING ") + f"{name} -> {s}" + ("" if why is None else f"  [{why}]"))
    print(f"{len(claims) - bad}/{len(claims)} README numbers match the result files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
