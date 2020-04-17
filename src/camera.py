import numpy as np
import numba
from constants import *
from primitives import unit, point, vec, Ray, Box
from bvh import BoundingVolumeHierarchy, traverse_bvh
from utils import timed
from multiprocessing.dummy import Pool
from functools import partial

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
    ('dx_dp', numba.float32[3]),
    ('dy_dp', numba.float32[3]),
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

        self.dx_dp = self.dx * self.phys_width / self.pixel_width
        self.dy_dp = self.dy * self.phys_height / self.pixel_height

        self.origin = (center - self.dx * phys_width / 2 - self.dy * phys_height / 2).astype(np.float32)

        self.image = np.zeros((pixel_height, pixel_width, 3), dtype=np.float32)

    def make_ray(self, i, j):
        # was having difficulty making a good mass-ray-generation routine, settled on on-demand
        # speed is fine and it'll be good for future adaptive sampling stuff
        n1 = np.random.normal()
        n2 = np.random.normal()
        origin = self.origin + self.dx_dp * (j + n1) + self.dy_dp * (i + n2)
        ray = Ray(origin.astype(np.float32), unit(self.focal_point - origin).astype(np.float32))
        ray.i = i
        ray.j = j
        return ray


@numba.jit(nogil=True)
def screen_sample(camera: Camera, root: Box, sample_number):
    image = camera.image * 0
    for i in range(camera.pixel_height):
        for j in range(camera.pixel_width):
            ray = camera.make_ray(i, j)
            image[i][j] = sample(root, ray)
    return image


@numba.jit(nogil=True)
def pixel_multi_sample(camera: Camera, root: Box, samples, index):
    value = vec(0, 0, 0)
    i = index / camera.pixel_width
    j = index % camera.pixel_width
    for _ in range(samples):
        ray = camera.make_ray(i, j)
        value += sample(root, ray)
    return value


def single_threaded_capture(camera: Camera, root: Box, samples=10):
    for n in range(samples):
        camera.image += screen_sample(camera, root, n)


def parallel_capture(camera: Camera, root: Box, samples=10):
    pool = Pool()
    camera.image = np.sum(pool.map(partial(screen_sample, camera, root), range(samples)), axis=0)


def parallel_pixel_capture(camera: Camera, root: Box, samples=10):
    pool = Pool()
    pixels = np.array(pool.map(partial(pixel_multi_sample, camera, root, samples), range(camera.pixel_height * camera.pixel_width)))
    image = pixels.reshape(camera.pixel_height, camera.pixel_width, 3)
    camera.image = image


def tone_map(camera: Camera):
    tone_vector = point(0.0722, 0.7152, 0.2126)
    Lw = np.exp(np.sum(np.log(0.1 + np.sum(camera.image * tone_vector, axis=2))) / np.product(camera.image.shape))
    result = camera.image * 0.64 / Lw
    result = result / (result + 1)
    return (result * 255).astype(np.uint8)


@numba.jit(nogil=True)
def local_orthonormal_system(z):
    if np.abs(z[0]) > np.abs(z[1]):
        axis = UNIT_Y
    else:
        axis = UNIT_X
    x = np.cross(axis, z)
    y = np.cross(z, x)
    return x, y, z


@numba.jit(nogil=True)
def random_hemisphere_cosine_weighted(x_axis, y_axis, z_axis):
    u1 = np.random.random()
    u2 = np.random.random()
    r = np.sqrt(u1)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x * x_axis + y * y_axis + z_axis * np.sqrt(np.maximum(0., 1. - u1))


def specular_reflection(direction, normal):
    return 2 * np.dot(direction, normal) * normal - direction


@numba.jit(nogil=True)
def dir_to_color(direction):
    return (.5 + unit(direction) / 2).astype(np.float32)


@numba.jit(nogil=True)
def sample(root: Box, ray: Ray):
    while ray.bounces <= 3:
        triangle, t = traverse_bvh(root, ray)
        if triangle is not None:
            if triangle.emitter:
                return ray.color * triangle.color
            else:
                ray.color *= triangle.color
                x, y, z = local_orthonormal_system(triangle.n)
                new_dir = random_hemisphere_cosine_weighted(x, y, z)
                ray.update(t, new_dir, triangle.n)
        else:
            return BLACK # exited the scene
    return BLACK


@numba.jit(nogil=True)
def simple_sample(root: Box, ray: Ray):
    triangle, t = traverse_bvh(root, ray)
    if triangle is not None:
        return triangle.color
    else:
        return BLACK  # exited the scene


if __name__ == '__main__':
    c = Camera()
