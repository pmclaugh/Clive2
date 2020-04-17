import numpy as np
import numba
from constants import *
from primitives import unit, point, vec, Ray, Box
from bvh import BoundingVolumeHierarchy, traverse_bvh
from utils import timed

@numba.jitclass([
    ('center', numba.float32[3]),
    ('direction', numba.float32[3]),
    ('phys_width', numba.float32),
    ('phys_height', numba.float32),
    ('focal_dist', numba.float32),
    ('focal_point', numba.float32[3]),
    ('pixel_width', numba.int32),
    ('pixel_height', numba.int32),
    ('dx', numba.float32[3]),
    ('dy', numba.float32[3]),
    ('origin', numba.float32[3]),
    ('image', numba.float32[:, :, :]),
])
class Camera:
    def __init__(self, center=point(0, 0, 0), direction=vec(1, 0, 0), phys_width=1.0, phys_height=1.0,
                 pixel_width=1280, pixel_height=720):
        self.center = center
        self.direction = direction
        self.phys_width = phys_width
        self.phys_height = phys_height
        self.focal_dist = (phys_width / 2.) / np.tan(H_FOV / 2.0)
        self.focal_point = self.center + self.focal_dist * direction
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

        if abs(self.direction[0]) < FLOAT_TOLERANCE:
            self.dx = UNIT_X if direction[2] > 0 else UNIT_X * -1
        else:
            self.dx = unit(np.cross(direction * (UNIT_X + UNIT_Z), UNIT_Y * -1))

        if abs(self.direction[1]) < FLOAT_TOLERANCE:
            self.dy = UNIT_Y
        else:
            self.dy = unit(np.cross(direction, self.dx))

        self.origin = (center - self.dx * phys_width / 2 - self.dy * phys_height / 2).astype(np.float32)

        self.image = np.zeros((pixel_height, pixel_width, 3), dtype=np.float32)


@numba.jit(nogil=True)
def make_rays(camera: Camera):
    # this is so painful. this is the fastest i've been able to make it,
    # but it feels like there should be a much cleaner/faster way.
    rays = []
    dx_dp = camera.dx * camera.phys_width / camera.pixel_width
    dy_dp = camera.dy * camera.phys_height / camera.pixel_height
    for i in range(camera.pixel_height):
        for j in range(camera.pixel_width):
            origin = camera.origin + i * dy_dp + j * dx_dp
            ray = Ray(origin, unit(camera.focal_point - origin))
            ray.i = i
            ray.j = j
            rays.append(ray)
    return rays


def capture(camera: Camera, root: Box):
    rays = make_rays(camera)
    for ray in rays:
        camera.image[ray.i][ray.j] = sample(root, ray)
    return camera.image


@numba.jit(nogil=True)
def sample(root: Box, ray: Ray):
    while ray.bounces <= MAX_BOUNCES:
        result = traverse_bvh(root, ray)
        if result is not None:
            return result.color
        else:
            return BLACK


# todo: traverse_bvh must return t so sample can advance the path.
#  need random on unit hemisphere, a box around the teapot, and a light source.

if __name__ == '__main__':
    c = Camera()
    make_rays(c)
