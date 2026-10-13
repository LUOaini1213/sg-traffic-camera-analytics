"""Class scheme for the four vehicle classes, and the mapping from COCO ids."""
NAMES = ["car", "bus", "truck", "motorcycle"]
COCO_TO_OURS = {2: 0, 5: 1, 7: 2, 3: 3}  # COCO car, bus, truck, motorcycle
# Boxes whose shorter side is below this fraction of image width are neither labelled
# nor scored (10 px on a 1920-px frame). Applied identically to labels and predictions.
MIN_SIDE_FRAC = 10 / 1920
