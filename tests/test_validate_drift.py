import gzip

import numpy as np
import pytest

from tcam.drift import drift_report, ks_stat, psi
from tcam.validate import agreement, camera_speed, incidents_near, links_near, read_snapshot, spearman


def test_agreement_confusion_and_kappa():
    pred = [True, True, True, False, False, False, False, False, True, False]
    true = [True, True, False, True, False, False, False, False, False, False]
    a = agreement(pred, true)
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (2, 2, 1, 5)
    assert a["precision"] == 0.5 and a["recall"] == pytest.approx(2 / 3)
    # po = .7 ; pe = (4*3 + 6*7)/100 = .54 ; kappa = .16/.46
    assert a["kappa"] == pytest.approx(0.16 / 0.46)


def test_spearman_with_ties_and_nan():
    x = [1, 2, 2, 3, np.nan]
    y = [10, 20, 30, 40, 5]
    # ranks x: 0, 1.5, 1.5, 3 ; y: 0, 1, 2, 3
    rx = np.array([0, 1.5, 1.5, 3]); ry = np.array([0, 1, 2, 3])
    assert spearman(x, y) == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert np.isnan(spearman([1, 2], [1, 2]))


def test_links_near_radius_and_category():
    cams = {"A": (1.3000, 103.8000)}
    links = [
        {"LinkID": "1", "RoadCategory": "1", "StartLat": "1.3010", "StartLon": "103.7990", "EndLat": "1.3010", "EndLon": "103.8010"},  # ~111 m
        {"LinkID": "2", "RoadCategory": "5", "StartLat": "1.3000", "StartLon": "103.7990", "EndLat": "1.3000", "EndLon": "103.8010"},  # through cam, small road
        {"LinkID": "3", "RoadCategory": "2", "StartLat": "1.3030", "StartLon": "103.7990", "EndLat": "1.3030", "EndLon": "103.8010"},  # ~334 m
    ]
    assert links_near(cams, links, radius_m=200) == {"A": {"1"}}
    assert links_near(cams, links, radius_m=400) == {"A": {"1", "3"}}


def test_camera_speed_and_snapshot(tmp_path):
    p = tmp_path / "sb.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("LinkID,SpeedBand,MinimumSpeed,MaximumSpeed\n1,2,10,19\n3,5,40,49\n4,8,70,999\n")
    b = read_snapshot(p)
    assert b == {"1": 2, "3": 5, "4": 8}
    assert camera_speed(b, {"1", "3", "missing"}) == 3.5
    assert np.isnan(camera_speed(b, {"x"}))


def test_incidents_near_type_filter():
    inc = [{"Type": "Accident", "Latitude": 1.3005, "Longitude": 103.8}, {"Type": "Roadwork", "Latitude": 1.3, "Longitude": 103.8},
           {"Type": "Accident", "Latitude": 1.32, "Longitude": 103.8}]
    assert len(incidents_near(inc, 1.3, 103.8)) == 2
    assert [i["Type"] for i in incidents_near(inc, 1.3, 103.8, types={"Accident"})] == ["Accident"]


def test_psi_hand_computed():
    ref = np.arange(10)
    cur = [1, 2, 7, 8, 9, 9, 9, 9]
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert psi(ref, cur, bins=2) == pytest.approx(expected)
    assert psi(ref, ref, bins=5) == pytest.approx(0)


def test_ks_hand_computed():
    assert ks_stat([1, 2, 3], [2, 3, 4, 5]) == pytest.approx(0.5)
    assert ks_stat([1, 2, 3], [1, 2, 3]) == 0


def test_drift_report_flags_shift_only():
    rng = np.random.default_rng(0)
    ref = {"a": list(rng.poisson(10, 300)), "b": list(rng.poisson(10, 300))}
    cur = {"a": list(rng.poisson(10, 300)), "b": list(rng.poisson(25, 300)), "c": [1, 2]}
    r = drift_report(ref, cur)
    assert set(r) == {"a", "b"}
    assert r["a"]["drift"] is False and r["b"]["drift"] is True
