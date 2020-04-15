import cv2
import numpy as np
from camera import Camera
from scene import dummy_scene
from primitives import point
from utils import timed
from datetime import datetime

WINDOW_WIDTH = 320
WINDOW_HEIGHT = 180

@timed
def render_something():
    s = datetime.now()
    c = Camera(center=point(0, 0, -2), direction=point(0, 0, 1), pixel_height=WINDOW_HEIGHT, pixel_width=WINDOW_WIDTH,
               phys_width=16., phys_height=9.)
    scene = dummy_scene()
    print(datetime.now() - s, 'loading/compiling')
    return c.capture(scene)


if __name__ == '__main__':
    cv2.imshow('render', render_something())
    cv2.waitKey(0)