from camera import Camera
from primitives import FastBox, Path
from routines import generate_path
from constants import *
import numba
from utils import timed


@timed
@numba.njit
def unidirectional_screen_sample(camera: Camera, boxes, triangles, emitters):
    for i in range(camera.pixel_height):
        for j in range(camera.pixel_width):
            camera_path = generate_path(boxes, triangles, camera.make_ray(i, j), Direction.FROM_CAMERA.value, stop_for_light=True)
            camera.image[i][j] += unidirectional_sample(camera_path)
    camera.sample_counts += 1
    camera.samples += 1


@numba.njit
def unidirectional_sample(camera_path: Path):
    if not camera_path.hit_light:
        return BLACK
    else:
        return camera_path.ray.color / camera_path.ray.p
