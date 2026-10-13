import numpy as np

from tcam.flags import CameraFlag, calibrate, flag_series


def test_flag_sequence_median_window_and_hysteresis():
    totals = [12, 3, 11, 11, 9, 7, 12, 3, 9, 30]
    got = flag_series(totals, threshold=10, window=3, release=0.8)
    # medians: -, -, 11, 11, 11, 9, 9, 7, 9, 9
    assert got == [False, False, True, True, True, True, True, False, False, False]


def test_flag_needs_full_window():
    assert flag_series([50, 50], threshold=10, window=3) == [False, False]
    assert flag_series([50, 50, 50], threshold=10, window=3) == [False, False, True]


def test_single_spike_does_not_flag():
    assert not any(flag_series([2, 2, 40, 2, 2, 2], threshold=10, window=3))


def test_release_ratio_matters():
    seq = [20, 20, 20, 9, 9, 9]
    assert flag_series(seq, threshold=10, release=0.8)[-1] is True     # 9 >= 8 stays on
    assert flag_series(seq, threshold=10, release=0.95)[-1] is False   # 9 < 9.5 releases


def test_update_reports_median_and_threshold():
    f = CameraFlag(threshold=10, window=3)
    for t in (1, 20, 4):
        r = f.update(t)
    assert r == {"median": 4.0, "threshold": 10, "congested": False}


def test_calibrate_quantile_and_floor():
    assert calibrate(np.arange(20), quantile=0.85, floor=8) == np.quantile(np.arange(20), 0.85) == 16.15
    assert calibrate([1, 2, 3, 2, 1], quantile=0.85, floor=8) == 8.0
    assert calibrate([], floor=6) == 6.0
    assert calibrate(np.arange(20), quantile=0.5, floor=0) == 9.5


def test_median_equal_to_threshold_flags():
    assert flag_series([10, 10, 10], threshold=10) == [False, False, True]
    assert flag_series([10, 9, 9], threshold=10) == [False, False, False]
