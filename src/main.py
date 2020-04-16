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
    return capture(c, bvh.root.box)

# todo: this is really fun. so much more to do before it's actually a thing though. here's a rough order
#  - displaying images - done
#  - camera - done
#  - basic ray casting - done
#  - BVH - done
#  - loading models -done
#  - performance work - done (camera.py still needs work)
#  - multiple bounces, paths
#  - automated tests
#  - BRDFs, importance sampling
#  - unidirectional path tracing
#  - Bidirectional path tracing
#  Can safely copy a lot of basic routines from rtv2 but I want to redesign a lot of the non-gpu code

if __name__ == '__main__':
    cv2.imshow('render', render_something())
    cv2.waitKey(0)