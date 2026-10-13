import pytest

from tcam.labels import apply_review, edit_stats

PRE = {"width": 1920, "boxes": [
    {"id": 0, "cls": 0, "xyxy": [0, 0, 50, 40]},
    {"id": 1, "cls": 2, "xyxy": [100, 100, 200, 180]},
    {"id": 2, "cls": 0, "xyxy": [300, 300, 340, 330]},
]}


def test_apply_review_all_edit_kinds():
    rev = {"remove": [0], "relabel": {"1": 1}, "adjust": {"2": [300, 300, 350, 335]},
           "add": [{"cls": 3, "xyxy": [520, 510, 500, 540]}, {"cls": 0, "xyxy": [0, 0, 5, 5]}]}
    out = apply_review(PRE, rev)
    assert out == [(1, [100.0, 100.0, 200.0, 180.0]), (0, [300.0, 300.0, 350.0, 335.0]),
                   (3, [500.0, 510.0, 520.0, 540.0])]  # added box normalised; 5-px box dropped by the floor
    assert edit_stats(PRE, rev) == {"pre_boxes": 3, "removed": 1, "relabelled": 1, "adjusted": 1, "added": 2}


def test_empty_review_keeps_pre():
    assert [c for c, _ in apply_review(PRE, {})] == [0, 2, 0]


def test_review_errors():
    with pytest.raises(KeyError):
        apply_review(PRE, {"remove": [7]})
    with pytest.raises(ValueError):
        apply_review(PRE, {"remove": [1], "relabel": {"1": 0}})
