import cv2
import numpy as np
from camera import Camera, capture
from scene import dummy_scene
from primitives import point, Box
from constants import ZEROS, ONES
from utils import timed
from datetime import datetime
from bvh import BoundingVolumeHierarchy, triangles_for_box
from load import load_obj

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

@timed
def render_something():
    s = datetime.now()
    c = Camera(center=point(0, 0, 10), direction=point(0, 0, -1), pixel_height=WINDOW_HEIGHT, pixel_width=WINDOW_WIDTH,
               phys_width=1., phys_height=1.)
    box = Box(ONES * -5, ONES * 5)
    bvh = BoundingVolumeHierarchy(triangles_for_box(box))
    print(datetime.now() - s, 'loading/compiling')
    return capture(c, bvh.root.box)

# todo: Feature Schedule
#  - multiple bounces, paths
#  - BRDFs, importance sampling
#  - unidirectional path tracing
#  - Bidirectional path tracing

# todo: Tech Debt
#  - Automated tests
#  - jit OBJ loading and bvh construction, eliminate TreeBox class


if __name__ == '__main__':
    cv2.imshow('render', render_something())
    cv2.waitKey(0)