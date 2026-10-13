import numpy as np
import pytest

from tcam.metrics import (ImageMatch, average_precision, cluster_bootstrap, match_image, mean_ap,
                          per_class_ap, precision_recall_at)


def test_ap_hand_computed():
    # scores .9 TP, .8 FP, .7 TP, .6 TP ; 3 labels
    # precision 1, .5, .667, .75 ; recall 1/3, 1/3, 2/3, 1 ; envelope 1, .75, .75, .75
    # r in {0..0.33}: 34 points at 1.0 ; r in {0.34..1}: 67 points at .75
    ap = average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 1], bool), 3)
    assert ap == pytest.approx((34 * 1 + 67 * 0.75) / 101)


def test_ap_unreached_recall_counts_zero():
    # one TP out of 4 labels: recall .25 -> 26 points at 1.0
    ap = average_precision(np.array([0.9, 0.1]), np.array([1, 0], bool), 4)
    assert ap == pytest.approx(26 / 101)


def test_ap_order_independent_of_input_order():
    s = np.array([0.6, 0.9, 0.7, 0.8]); t = np.array([1, 1, 1, 0], bool)
    assert average_precision(s, t, 3) == pytest.approx((34 + 67 * 0.75) / 101)


def test_ap_edge_cases():
    assert np.isnan(average_precision(np.array([0.5]), np.array([False]), 0))
    assert average_precision(np.zeros(0), np.zeros(0, bool), 2) == 0.0


def test_match_image_duplicates_class_and_iou():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    gc = [0, 1]
    pred = [[0, 0, 10, 10],     # TP class 0
            [0, 0, 10, 9],      # duplicate on the same label -> FP (IoU .9)
            [20, 20, 30, 30],   # right place, wrong class (0) -> FP
            [20, 20, 30, 26]]   # class 1, IoU .6 -> TP at .5, FP at .7
    ps = [0.9, 0.8, 0.95, 0.5]
    pc = [0, 0, 0, 1]
    m = match_image(pred, ps, pc, gt, gc, classes=[0, 1], iou_thr=0.5)
    assert m.n_gt == {0: 1, 1: 1}
    assert list(m.scores[0]) == [0.95, 0.9, 0.8]
    assert list(m.tp[0]) == [False, True, False]
    assert list(m.tp[1]) == [True]
    m7 = match_image(pred, ps, pc, gt, gc, classes=[0, 1], iou_thr=0.7)
    assert list(m7.tp[1]) == [False]


def test_match_prefers_highest_iou_label():
    gt = [[0, 0, 10, 10], [2, 0, 12, 10]]
    m = match_image([[2, 0, 12, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 0], gt, [0, 0], classes=[0])
    assert list(m.tp[0]) == [True, True]


def test_per_class_and_mean_ap_skip_classes_without_labels():
    m1 = match_image([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10]], [0], classes=[0, 1, 2])
    m2 = match_image([[50, 50, 60, 60]], [0.8], [1], [[0, 0, 10, 10]], [1], classes=[0, 1, 2])
    aps = per_class_ap([m1, m2], [0, 1, 2])
    assert aps[0] == pytest.approx(1.0) and aps[1] == 0.0 and np.isnan(aps[2])
    assert mean_ap(aps) == pytest.approx(0.5)


def test_precision_recall_at_conf():
    m = ImageMatch(scores={0: np.array([0.9, 0.5, 0.3])}, tp={0: np.array([True, False, True])}, n_gt={0: 4})
    assert precision_recall_at([m], [0], 0.4) == (0.5, 0.25)
    assert precision_recall_at([m], [0], 0.3) == (2 / 3, 0.5)


def test_cluster_bootstrap_resamples_groups():
    good = match_image([[0, 0, 10, 10]], [0.9], [0], [[0, 0, 10, 10]], [0], classes=[0])
    bad = match_image([[50, 50, 60, 60]], [0.8], [0], [[0, 0, 10, 10]], [0], classes=[0])
    same = {"a": [[good], [bad]], "b": [[good], [bad]]}
    b = cluster_bootstrap(same, [0], n_boot=50, seed=1)
    assert b.shape == (50, 2) and np.all(b[:, 0] == 1.0) and np.all(b[:, 1] == 0.0)
    mixed = {"a": [[good]], "b": [[bad]]}
    v = cluster_bootstrap(mixed, [0], n_boot=400, seed=2)[:, 0]
    # two groups resampled: all-good 1.0, all-bad 0.0, one of each: recall .5 at precision 1 -> 51/101
    assert set(np.round(v, 4)) == {0.0, 1.0, round(51 / 101, 4)}
