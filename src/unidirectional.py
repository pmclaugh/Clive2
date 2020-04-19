from camera import Camera
from primitives import Box, Path
from routines import generate_path
from constants import *
import numba
from utils import timed


@timed
# @numba.jit(nogil=True)
def unidirectional_screen_sample(camera: Camera, root: Box, samples=5):
    for _ in range(samples):
        for i in range(camera.pixel_height):
            for j in range(camera.pixel_width):
                camera_path = generate_path(root, camera.make_ray(i, j), Direction.FROM_CAMERA.value, stop_for_light=True)
                camera.image[i][j] += unidirectional_sample(camera_path)
    camera.samples += samples


# @numba.jit(nogil=True)
def unidirectional_sample(camera_path: Path):
    if not camera_path.hit_light:
        return BLACK
    else:
        # todo this is no longer correct, camera_path.ray.p is missing one last multiplier
        return camera_path.ray.color / camera_path.ray.p
