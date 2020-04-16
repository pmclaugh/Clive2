import numba
import numpy as np
from constants import *
from datetime import datetime
from typing import List
from utils import timed

# sugar


@numba.jit(nogil=True, cache=True)
def point(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


@numba.jit(nogil=True, cache=True)
def vec(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


@numba.jit(nogil=True, cache=True)
def unit(v):
    return v / np.linalg.norm(v)

# fast primitives


@numba.jitclass([
    ('origin', numba.float32[3::1]),
    ('direction', numba.float32[3::1]),
    ('inv_direction', numba.float32[3::1]),
    ('sign', numba.uint8[3::1]),
    ('color', numba.float32[3::1]),
    ('i', numba.int32),
    ('j', numba.int32),
])
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.inv_direction = 1 / direction
        self.sign = (self.inv_direction < 0).astype(np.uint8)
        self.color = ZEROS
        self.i = 0
        self.j = 0


@numba.jitclass([
    ('v0', numba.float32[3::1]),
    ('v1', numba.float32[3::1]),
    ('v2', numba.float32[3::1]),
    ('e1', numba.float32[3::1]),
    ('e2', numba.float32[3::1]),
    ('n', numba.float32[3::1]),
    ('mins', numba.float32[3::1]),
    ('maxes', numba.float32[3::1]),
    ('color', numba.uint8[3::1]),
])
class Triangle:
    def __init__(self, v0, v1, v2, color=WHITE):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.e1 = v1 - v0
        self.e2 = v2 - v0
        self.n = unit(np.cross(self.e1, self.e2))
        self.mins = np.minimum(np.minimum(v0, v1), v2)
        self.maxes = np.maximum(np.maximum(v0, v1), v2)
        self.color = color


@numba.jit(nogil=True, fastmath=True, cache=True)
def ray_triangle_intersect(ray: Ray, triangle: Triangle):
    h = np.cross(ray.direction, triangle.e2)
    a = np.dot(h, triangle.e1)

    if a < 0:
        return None

    f = 1. / a
    s = ray.origin - triangle.v0
    u = f * np.dot(s, h)
    if u < 0. or u > 1.:
        return None
    q = np.cross(s, triangle.e1)
    v = f * np.dot(q, ray.direction)
    if v < 0. or v > 1.:
        return None

    t = f * np.dot(triangle.e2, q)
    if t > 0:
        return t
    else:
        return None


@numba.jitclass([
    ('min', numba.float32[3::1]),
    ('max', numba.float32[3::1]),
    ('bounds', numba.float32[:, ::1]),
    ('span', numba.float32[3::1]),
    ('color', numba.uint8[3::1]),
])
class Box:
    def __init__(self, least_corner, most_corner, color=WHITE):
        self.min = least_corner
        self.max = most_corner
        self.bounds = np.stack((least_corner, most_corner))
        self.span = self.max - self.min
        self.color = color

    def contains(self, point: numba.float32[3]):
        return (point >= self.min).all() and (point <= self.max).all()

    def extend(self, triangle: Triangle):
        self.min = np.minimum(triangle.mins, self.min)
        self.max = np.maximum(triangle.maxes, self.max)
        self.span = self.max - self.min
        self.bounds = np.stack((self.min, self.max))

    def surface_area(self):
        return 2 * (self.span[0] * self.span[1] + self.span[1] * self.span[2] + self.span[0] * self.span[2])

@numba.jit(nogil=True, fastmath=True, cache=True)
def ray_box_intersect(ray: Ray, box: Box):
    txmin = (box.bounds[ray.sign[0]][0] - ray.origin[0]) * ray.inv_direction[0]
    txmax = (box.bounds[1 - ray.sign[0]][0] - ray.origin[0]) * ray.inv_direction[0]
    tymin = (box.bounds[ray.sign[1]][1] - ray.origin[1]) * ray.inv_direction[1]
    tymax = (box.bounds[1 - ray.sign[1]][1] - ray.origin[1]) * ray.inv_direction[1]

    if txmin > tymax or tymin > txmax:
        return False, 0., 0.
    tmin = max(txmin, tymin)
    tmax = min(txmax, tymax)

    tzmin = (box.bounds[ray.sign[2]][2] - ray.origin[2]) * ray.inv_direction[2]
    tzmax = (box.bounds[1 - ray.sign[2]][2] - ray.origin[2]) * ray.inv_direction[2]

    if tmin > tzmax or tzmin > tmax:
        return False, 0., 0.
    tmin = max(tmin, tzmin)
    tmax = min(tmax, tzmax)
    if tmax > 0:
        return True, tmin, tmax
    else:
        return False, 0., 0.

# todo: this is really fun. so much more to do before it's actually a thing though. here's a rough order
#  - displaying images - done
#  - camera - done
#  - basic ray casting - done
#  - BVH - done
#  - loading models -done
#  - performance work - good progress, need a TreeBox jitclass to really get there
#  - multiple bounces, paths
#  - automated tests
#  - BRDFs, importance sampling
#  - unidirectional path tracing
#  - Bidirectional path tracing
#  Can safely copy a lot of basic routines from rtv2 but I want to redesign a lot of the non-gpu code


if __name__ == '__main__':
    r = Ray(ZEROS, UNIT_X)
    t = Triangle(ZEROS, UNIT_Y, UNIT_X)