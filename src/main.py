import cv2
import numpy as np
from camera import Camera, capture
from scene import dummy_scene
from primitives import point
from utils import timed
from datetime import datetime
from bvh import BoundingVolumeHierarchy
from load import load_obj

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360

@timed
def render_something():
    s = datetime.now()
    c = Camera(center=point(0, 1.5, -10), direction=point(0, 0, 1), pixel_height=WINDOW_HEIGHT, pixel_width=WINDOW_WIDTH,
               phys_width=16., phys_height=9.)
    bvh = BoundingVolumeHierarchy(load_obj('../resources/teapot.obj'))
    print(datetime.now() - s, 'loading/compiling')
    return capture(c, bvh)


if __name__ == '__main__':
    cv2.imshow('render', render_something())
    cv2.waitKey(0)