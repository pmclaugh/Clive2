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
import threading

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
SAMPLE_COUNT = 100


class RenderThread(threading.Thread):
    def __init__(self, camera, scene):
        threading.Thread.__init__(self)
        self.camera = camera
        self.scene = scene
        self.done_event = threading.Event()

    def run(self) -> None:
        try:
            for n in range(SAMPLE_COUNT):
                sample = unidirectional_screen_sample(self.camera, self.scene, 1)
                camera.image += sample
                camera.samples += 1
                print('sample', n, 'done')
        except KeyboardInterrupt:
            pass
        self.done_event.set()


if __name__ == '__main__':
    camera = Camera(center=point(0, 2, 7), direction=point(0, 0, -1), pixel_height=WINDOW_HEIGHT,
                    pixel_width=WINDOW_WIDTH, phys_width=WINDOW_WIDTH / WINDOW_HEIGHT, phys_height=1.)
    bvh = BoundingVolumeHierarchy(
        triangles_for_box(Box(point(-10, -3, -10), point(10, 17, 10))) + load_obj('../resources/teapot.obj', material=Material.SPECULAR.value))

    renderer = RenderThread(camera, bvh.root.box)
    renderer.start()
    while not renderer.done_event.isSet():
        cv2.imshow('render', tone_map(camera))
        cv2.waitKey(100)
    cv2.imwrite('../renders/%s.jpg' % datetime.now(), tone_map(camera))

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
