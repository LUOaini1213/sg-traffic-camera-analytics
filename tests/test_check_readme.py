"""scripts/check_readme.py must reject a number that is only present somewhere else in the README."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_check_readme", ROOT / "scripts" / "check_readme.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

README = """Intro: champion 0.824 -> see table.

| Model | mAP50 | 95% CI | count |
|---|---|---|---|
| **seed 1** | **0.539** | **0.473-0.653** | 0 TP, 9 FP |
| seed 0 | 0.477 | 0.442-0.562 | 10 TP, 0 FP |
"""


def run(claim, value):
    return cr.check(claim, cr.fmt(value, claim["fmt"]), README, cr.tables(README.splitlines()))


def cell(row, col, **kw):
    return {"fmt": kw.pop("fmt", ".3f"), "table": {"header": "Model", "row": row, "col": col, **kw}}


def test_table_cell_must_hold_the_value_not_another_row():
    assert run(cell("seed 1", "mAP50"), 0.5393) is None
    assert run(cell("seed 1", "mAP50"), 0.477) is not None  # 0.477 is in the README, but in seed 0's row
    assert run(cell("seed 0", "95% CI", text="{}-"), 0.442) is None
    assert run(cell("seed 0", "95% CI", text="-{}"), 0.442) is not None


def test_numbers_do_not_match_inside_longer_numbers():
    assert not cr.has_number("0.824", "0.82")
    assert not cr.has_number("10.82", "0.82")
    assert cr.has_number("10 TP, 0 FP", "0", "{} FP")
    assert not cr.has_number("10 TP, 0 FP", "0", "{} TP")


def test_context_anchor_and_missing_anchor():
    assert run({"fmt": ".3f", "context": "champion {} ->"}, 0.8241) is None
    assert run({"fmt": ".3f", "context": "champion {} ->"}, 0.539) is not None
    assert run({"fmt": ".3f"}, 0.539) is not None  # a claim without an anchor is rejected


def test_aggregates_over_wildcards():
    data = {"cams": {"a": {"n": 4, "alert": True}, "b": {"n": 10, "alert": False}, "c": {"alert": True}}}
    assert cr.AGG["sum"](cr.get(data, "cams|*|n")) == 14
    assert cr.AGG["count_true"](cr.get(data, "cams|*|alert")) == 2
    assert cr.AGG["len"](cr.get(data, "cams")) == 3
