import cv2
from camera import Camera, tone_map
from primitives import point, Box
from utils import timed
from datetime import datetime
from bvh import BoundingVolumeHierarchy, triangles_for_box
from load import load_obj
from bidirectional import bidirectional_screen_sample
from unidirectional import unidirectional_screen_sample
from constants import Material

WINDOW_WIDTH = 200
WINDOW_HEIGHT = 200
SAMPLE_COUNT = 30


@timed
def render_something():
    s = datetime.now()
    camera = Camera(center=point(0, 2, 7), direction=point(0, 0, -1), pixel_height=WINDOW_HEIGHT,
                    pixel_width=WINDOW_WIDTH, phys_width=WINDOW_WIDTH/WINDOW_HEIGHT, phys_height=1.)
    box = Box(point(-10, -3, -10), point(10, 17, 10))
    bvh = BoundingVolumeHierarchy(triangles_for_box(box) + load_obj('../resources/teapot.obj', material=Material.SPECULAR.value))
    print(datetime.now() - s, 'loading/compiling')
    try:
        unidirectional_screen_sample(camera, bvh.root.box, SAMPLE_COUNT)
    except KeyboardInterrupt:
        pass
    return tone_map(camera)


if __name__ == '__main__':
    render = render_something()
    cv2.imwrite('../renders/%s.jpg' % datetime.now(), render)
    cv2.imshow('render', render)
    cv2.waitKey(0)

# performance test unidirectional 200x200 10 samples
# parallel_pixel_capture - 33.9686
# parallel_capture - 33.8465
# single_threaded_capture - 44.5569

# todo: Feature Schedule
#  - Bidirectional is functionally in place but needs all the probability details implemented

# todo: Tech Debt
#  - Automated tests
#  - jit OBJ loading and bvh construction, eliminate TreeBox class

# todo: Known Bugs
