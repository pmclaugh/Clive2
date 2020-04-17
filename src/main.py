import cv2
import numpy as np
from camera import Camera, screen_sample, parallel_capture, parallel_pixel_capture, single_threaded_capture, tone_map
from scene import dummy_scene
from primitives import point, Box
from constants import ZEROS, ONES
from utils import timed
from datetime import datetime
from bvh import BoundingVolumeHierarchy, triangles_for_box
from load import load_obj

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360
SAMPLE_COUNT = 500

@timed
def render_something():
    s = datetime.now()
    camera = Camera(center=point(0, 2, 4), direction=point(0, 0, -1), pixel_height=WINDOW_HEIGHT, pixel_width=WINDOW_WIDTH,
               phys_width=WINDOW_WIDTH/WINDOW_HEIGHT, phys_height=1.)
    box = Box(point(-5, -1, -5), point(5, 9, 5))
    bvh = BoundingVolumeHierarchy(load_obj('../resources/teapot.obj') + triangles_for_box(box))
    print(datetime.now() - s, 'loading/compiling')
    try:
        parallel_pixel_capture(camera, bvh.root.box, samples=SAMPLE_COUNT)
    except KeyboardInterrupt:
        pass
    return tone_map(camera)


if __name__ == '__main__':
    render = render_something()
    cv2.imwrite('../renders/%s.jpg' % datetime.now(), render)
    cv2.imshow('render', render)
    cv2.waitKey(0)

# performance test 200x200 10 samples
# parallel_pixel_capture - 33.9686
# parallel_capture - 33.8465
# single_threaded_capture - 44.5569

# todo: Feature Schedule
#  - multiple bounces, paths
#  - BRDFs, importance sampling
#  - unidirectional path tracing
#  - Bidirectional path tracing

# todo: Tech Debt
#  - Automated tests
#  - jit OBJ loading and bvh construction, eliminate TreeBox class

# todo: Known Bugs
