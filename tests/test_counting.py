from tcam.counting import count, point_in_polygon

W = 1920
# x1, y1, x2, y2 ; all at least 10 px on the short side unless noted
BOXES = [
    [100, 100, 160, 140],   # car, conf .90
    [300, 300, 400, 380],   # bus, conf .30
    [500, 500, 560, 512],   # truck, short side 12 px, conf .80
    [700, 700, 760, 708],   # car, short side 8 px -> below floor
    [900, 900, 940, 960],   # motorcycle, conf .24 (just under default conf .25)
    [1000, 100, 1100, 160], # car, conf .25 exactly
    [1500, 50, 1560, 90],   # truck conf .5
]
SCORES = [0.90, 0.30, 0.80, 0.95, 0.24, 0.25, 0.50]
CLASSES = [0, 1, 2, 0, 3, 0, 2]


def test_count_exact_per_class():
    c = count(BOXES, SCORES, CLASSES, W)
    assert c == {"car": 2, "bus": 1, "truck": 2, "motorcycle": 0, "total": 5}


def test_conf_threshold_is_inclusive_and_configurable():
    assert count(BOXES, SCORES, CLASSES, W, conf=0.26)["car"] == 1
    low = count(BOXES, SCORES, CLASSES, W, conf=0.2)
    assert low["motorcycle"] == 1 and low["total"] == 6
    assert count(BOXES, SCORES, CLASSES, W, conf=0.85)["total"] == 1


def test_size_floor_scales_with_width():
    # On a 960-px-wide image the floor is 5 px, so the 8-px car now counts.
    assert count(BOXES, SCORES, CLASSES, 960)["car"] == 3


def test_roi_uses_bottom_centre():
    roi = [(0, 0), (1200, 0), (1200, 450), (0, 450)]
    c = count(BOXES, SCORES, CLASSES, W, roi=roi)
    # car1 (bottom y 140), car at 1000..1100 (bottom 160), truck 1500 is outside x; bus bottom 380 inside
    assert c == {"car": 2, "bus": 1, "truck": 0, "motorcycle": 0, "total": 3}
    # A box whose top is inside but whose bottom-centre is outside must not count.
    assert count([[100, 400, 200, 500]], [0.9], [0], W, roi=roi)["total"] == 0
    assert count([[100, 300, 200, 449]], [0.9], [0], W, roi=roi)["total"] == 1


def test_point_in_polygon_concave():
    L = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    assert point_in_polygon(0.5, 3, L)
    assert not point_in_polygon(3, 3, L)
    assert point_in_polygon(3, 0.5, L)
    assert not point_in_polygon(5, 0.5, L)


def test_point_in_polygon_slanted_edge():
    tri = [(0, 0), (10, 0), (0, 10)]
    assert point_in_polygon(6, 3, tri)
    assert not point_in_polygon(8, 3, tri)
