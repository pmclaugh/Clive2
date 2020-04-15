import numba
import numpy as np
from constants import *
from datetime import datetime
from typing import List

# sugar


@numba.jit(nogil=True)
def point(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


@numba.jit(nogil=True)
def vec(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


@numba.jit(nogil=True)
def unit(v):
    return v / np.linalg.norm(v)

# fast primitives


@numba.jitclass([
    ('origin', numba.float32[3]),
    ('direction', numba.float32[3]),
    ('inv_direction', numba.float32[3]),
    ('sign', numba.uint8[3])])
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.inv_direction = 1 / direction
        self.sign = (self.inv_direction < 0).astype(np.uint8)


@numba.jitclass([
    ('least_corner', numba.float32[3]),
    ('most_corner', numba.float32[3]),
    ('bounds', numba.float32[:, :]),
    ('color', numba.uint8[3]),
])
class Box:
    def __init__(self, least_corner, most_corner, color=WHITE):
        self.least_corner = least_corner
        self.most_corner = most_corner
        self.bounds = np.stack((least_corner, most_corner))
        self.color = color

    def contains(self, point: numba.float32[3]):
        return (point >= self.least_corner).all() and (point <= self.most_corner).all()

    def collide(self, ray: Ray, t_limit=np.inf):
        txmin = (self.bounds[ray.sign[0]][0] - ray.origin[0]) * ray.inv_direction[0]
        txmax = (self.bounds[1 - ray.sign[0]][0] - ray.origin[0]) * ray.inv_direction[0]
        tymin = (self.bounds[ray.sign[1]][1] - ray.origin[1]) * ray.inv_direction[1]
        tymax = (self.bounds[1 - ray.sign[1]][1] - ray.origin[1]) * ray.inv_direction[1]
        tzmin = (self.bounds[ray.sign[2]][2] - ray.origin[2]) * ray.inv_direction[2]
        tzmax = (self.bounds[1 - ray.sign[2]][2] - ray.origin[2]) * ray.inv_direction[2]

        if txmin > tymax or tymin > txmax:
            return False, 0., 0.
        tmin = max(txmin, tymin)
        tmax = min(txmax, tymax)

        if tmin > tzmax or tzmin > tmax:
            return False, 0., 0.
        tmin = max(tmin, tzmin)
        tmax = min(tmax, tzmax)
        if t_limit > tmax > 0:
            return True, tmin, tmax
        else:
            return False, 0., 0.

    def simple_collide(self, ray: Ray):
        hit, tmin, tmax = self.collide(ray)
        return hit


@numba.jitclass([
    ('v0', numba.float32[3]),
    ('v1', numba.float32[3]),
    ('v2', numba.float32[3]),
    ('color', numba.uint8[3]),
])
class Triangle:
    def __init__(self, v0, v1, v2, color=WHITE):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.color = color

    def collide(self, ray: Ray, t_limit=np.inf):
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0

        h = np.cross(ray.direction, e2)
        a = np.dot(h, e1)

        if a < FLOAT_TOLERANCE:
            return None

        f = 1. / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)
        if u < 0. or u > 1.:
            return None
        q = np.cross(s, e1)
        v = f * np.dot(ray.direction, q)
        if v < 0. or v > 1.:
            return None

        t = f * np.dot(e2, q)
        if t_limit > t > 0:
            return t
        else:
            return None


# 25M ray/box intersects/sec single threaded
@numba.jit(nogil=True)
def collide_test(box: Box, ray: Ray, n):
    for _ in range(n):
        box.collide(ray)


# 17M ray/tri intersects/sec single threaded
@numba.jit(nogil=True)
def tri_collide_test(tri: Triangle, ray: Ray, n):
    for _ in range(n):
        tri.collide(ray)

# todo: this is really fun. so much more to do before it's actually a thing though. here's a rough order
#  - displaying images - done
#  - camera - done
#  - basic ray casting - done
#  - BVH
#  - loading models
#  - BRDFs, importance sampling
#  - unidirectional path tracing
#  - Bidirectional path tracing
#  Can safely copy a lot of basic routines from rtv2 but I want to redesign a lot of the non-gpu code


class TreeBox:
    def __init__(self, low_corner, high_corner, parent=None, children=None, members=None):
        self.low = low_corner
        self.high = high_corner
        self.parent = parent
        if children is not None:
            self.children = children
        else:
            self.children = []
        if members is not None:
            self.members = members
        else:
            self.members = []
        self.box = Box(low_corner, high_corner)

    # @numba.jit(nogil=True)
    def collide(self, ray: Ray, t_max=np.inf):
        return self.box.collide(ray, t_limit=t_max)


def AABB(triangles: List[Triangle]):
    minimum = np.ones(3, dtype=np.float32) * np.inf
    maximum = -1 * minimum
    for triangle in triangles:
        minimum = np.minimum(triangle.v0, triangle.v1, triangle.v2, minimum)
        maximum = np.minimum(triangle.v0, triangle.v1, triangle.v2, minimum)
    return TreeBox(minimum, maximum)


class BoundingVolumeHierarchy:
    def __init__(self, triangles):
        self.root = None
        self.triangles = triangles
        self.build()

    def build(self, max_members=2, max_depth=10):
        self.root = AABB(self.triangles)
        self.root.members = list(self.triangles)
        stack = [self.root]
        while stack:
            node = stack.pop()
            if len(node.members) <= max_members or len(stack) > max_depth:
                # leaf node
                continue
            # find best split, do it
            # put results on stack

    def hit(self, ray: Ray):
        least_t = np.inf
        least_hit = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            hit, t_low, t_high = node.collide(ray, t_limit=least_t)
            if hit and t_low <= least_t:
                if node.children:
                    stack += node.children
                else:
                    for member in node.members:
                        t = member.collide(ray)
                        if t is not None and t < least_t:
                            least_t = t
                            least_hit = member
        return least_hit, least_t


if __name__ == '__main__':
    a = point(2, 2, 2)
    b = point(4, 4, 4)
    box = Box(a, b)
    tri = Triangle(point(1, 0, 0), point(1, 1, 0), point(1, 1, 1))
    r = Ray(point(2, 2, 0), point(0, 0, 1))
    tri_collide_test(tri, r, 1)
    s = datetime.now()
    tri_collide_test(tri, r, 100000)
    print(datetime.now() - s)