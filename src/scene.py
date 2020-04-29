import numba
import numpy as np
from typing import List
from primitives import *


@numba.experimental.jitclass([
    ('boxes', numba.types.ListType(Box.class_type.instance_type))
])
class SimpleScene:
    def __init__(self, boxes):
        self.boxes = boxes

    def hit(self, ray: Ray):
        best_t = np.inf
        best_box = None
        for box in self.boxes:
            hit, tmin, tmax = box.collide(ray)
            if hit and tmin < best_t:
                best_box = box
                best_t = tmin
        return best_box


def dummy_scene():
    l = numba.typed.List()
    l.append(Box(point(0, 0, 4), point(1, 1, 5), color=RED))
    l.append(Box(point(-1, 0, 4), point(0, 1, 5), color=GREEN))
    l.append(Box(point(-1, -1, 4), point(0, 0, 5), color=BLUE))
    l.append(Box(point(0, -1, 4), point(1, 0, 5), color=WHITE))
    return SimpleScene(l)
