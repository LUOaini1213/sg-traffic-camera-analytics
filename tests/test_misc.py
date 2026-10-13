import pytest

from tcam.boxes import from_yolo, iou_matrix, nms, size_ok, to_yolo
from tcam.cameras import in_scope
from tcam.collect import capture_time
from tcam.geo import haversine_m, point_segment_m
from tcam.split import camera_groups, split_cameras


def test_haversine_one_degree_latitude():
    assert haversine_m(1.0, 103.8, 2.0, 103.8) == pytest.approx(111195, rel=1e-3)


def test_point_segment_endpoint_and_perpendicular():
    d_perp = point_segment_m(1.0010, 103.8005, 1.0, 103.8, 1.0, 103.801)
    assert d_perp == pytest.approx(111.2, rel=0.01)
    d_end = point_segment_m(1.0, 103.803, 1.0, 103.8, 1.0, 103.801)  # 2 milli-degrees beyond the end
    assert d_end == pytest.approx(222.4, rel=0.01)


def test_iou_and_nms():
    a = [[0, 0, 10, 10]]
    b = [[5, 0, 15, 10], [0, 0, 10, 10], [20, 20, 30, 30]]
    assert iou_matrix(a, b)[0].tolist() == pytest.approx([50 / 150, 1.0, 0.0])
    keep = nms([[0, 0, 10, 10], [1, 0, 11, 10], [5, 0, 15, 10]], [0.8, 0.9, 0.7], iou_thr=0.5)
    # box1 (0.9) suppresses box0 (IoU .818); box2 vs box1 IoU 6/14=.43 survives
    assert keep == [1, 2]


def test_yolo_roundtrip_and_size():
    line = to_yolo([100, 50, 300, 150], 2, 1920, 1080)
    c, b = from_yolo(line, 1920, 1080)
    assert c == 2 and b == pytest.approx([100, 50, 300, 150], abs=0.01)
    assert size_ok([0, 0, 10, 40], 1920, 10 / 1920)
    assert not size_ok([0, 0, 9.9, 40], 1920, 10 / 1920)


def test_scope_and_capture_time():
    assert in_scope(103.77) and not in_scope(103.64)
    link = "https://x.s3.amazonaws.com/2026-10-13/21-45/2701_2144_20261013134522_F14CDA.jpg?X-Amz=1"
    assert capture_time(link) == "2026-10-13T21:44"
    assert capture_time("https://x/other.jpg") is None


def test_camera_groups_and_split():
    cams = {"a": (1.30, 103.80), "b": (1.3030, 103.80), "c": (1.40, 103.90), "d": (1.35, 103.70), "e": (1.45, 103.75)}
    g = camera_groups(cams, link_m=500)
    assert ["a", "b"] in g and len(g) == 4
    s = split_cameras(cams, {k: 10 for k in cams}, test_frac=0.2, val_frac=0.2, force_train={"c"}, seed=3)
    assert s["a"] == s["b"] and s["c"] == "train"
    assert sorted(s.values()).count("test") >= 1
