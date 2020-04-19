import numba
import numpy as np
from constants import *

# sugar


@numba.jit(nogil=True)
def point(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


@numba.jit(nogil=True)
def vec(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


@numba.jit(nogil=True)
def unit(v):
    return v / np.linalg.norm(v)

# fast primitives


ray_type = numba.deferred_type()
node_type = numba.deferred_type()
box_type = numba.deferred_type()

@numba.jitclass([
    ('origin', numba.float64[3::1]),
    ('direction', numba.float64[3::1]),
    ('inv_direction', numba.float64[3::1]),
    ('sign', numba.uint8[3::1]),
    ('color', numba.float64[3::1]),
    ('i', numba.int32),
    ('j', numba.int32),
    ('bounces', numba.int32),
    ('p', numba.float64),
    ('next', numba.optional(ray_type)),
    ('prev', numba.optional(ray_type)),
])
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin.copy()
        self.direction = direction.copy()
        self.inv_direction = 1 / self.direction
        self.sign = (self.inv_direction < 0).astype(np.uint8)
        self.color = WHITE.copy()
        self.i = 0
        self.j = 0
        self.bounces = 0
        self.p = 1
        self.next = None
        self.prev = None

    def update(self, t, new_direction, incident_normal):
        self.origin = (self.origin + t * self.direction).astype(np.float64)
        self.direction = new_direction.astype(np.float64)
        self.origin += incident_normal * COLLISION_OFFSET
        self.inv_direction = 1 / self.direction
        self.sign = (self.inv_direction < 0).astype(np.uint8)
        self.bounces += 1


@numba.jitclass([
    ('v0', numba.float64[3::1]),
    ('v1', numba.float64[3::1]),
    ('v2', numba.float64[3::1]),
    ('e1', numba.float64[3::1]),
    ('e2', numba.float64[3::1]),
    ('normal', numba.float64[3::1]),
    ('mins', numba.float64[3::1]),
    ('maxes', numba.float64[3::1]),
    ('color', numba.float64[3::1]),
    ('emitter', numba.boolean),
    ('material', numba.int64)
])
class Triangle:
    def __init__(self, v0, v1, v2, color=WHITE, emitter=False, material=Material.DIFFUSE.value):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.e1 = v1 - v0
        self.e2 = v2 - v0
        self.normal = unit(np.cross(self.e1, self.e2))
        self.mins = np.minimum(np.minimum(v0, v1), v2)
        self.maxes = np.maximum(np.maximum(v0, v1), v2)
        self.color = color
        self.emitter = emitter
        self.material = material

    def sample_surface(self):
        r1 = np.random.random()
        r2 = np.random.random()
        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1) * (1 - r2)
        w = r2 * np.sqrt(r1)
        return self.v0 * u + self.v1 * v + self.v2 * w + COLLISION_OFFSET * self.normal


@numba.jitclass([
    ('min', numba.float64[3::1]),
    ('max', numba.float64[3::1]),
    ('bounds', numba.float64[:, ::1]),
    ('span', numba.float64[3::1]),
    ('left', numba.optional(box_type)),
    ('right', numba.optional(box_type)),
    ('triangles', numba.optional(numba.types.ListType(Triangle.class_type.instance_type))),
    ('lights', numba.optional(numba.types.ListType(Triangle.class_type.instance_type))),
])
class Box:
    def __init__(self, least_corner, most_corner, color=WHITE):
        self.min = least_corner
        self.max = most_corner
        self.bounds = np.stack((least_corner, most_corner))
        self.span = self.max - self.min
        self.left = None
        self.right = None
        self.triangles = None
        self.lights = None

    def contains(self, point: numba.float64[3]):
        return (point >= self.min).all() and (point <= self.max).all()

    def extend(self, triangle: Triangle):
        self.min = np.minimum(triangle.mins, self.min)
        self.max = np.maximum(triangle.maxes, self.max)
        self.span = self.max - self.min
        self.bounds = np.stack((self.min, self.max))

    def surface_area(self):
        return 2 * (self.span[0] * self.span[1] + self.span[1] * self.span[2] + self.span[0] * self.span[2])


@numba.jitclass([
    ('next', numba.optional(node_type)),
    ('data', box_type),
])
class BoxStackNode:
    def __init__(self, data):
        self.data = data
        self.next = None


@numba.jitclass([
    ('head', numba.optional(node_type)),
    ('size', numba.uint32)
])
class BoxStack:
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, data):
        node = BoxStackNode(data)
        node.next = self.head
        self.head = node
        self.size += 1

    def pop(self):
        old = self.head
        if old is None:
            return None
        self.head = old.next
        self.size -= 1
        return old.data


@numba.jitclass([
    ('ray', Ray.class_type.instance_type),
    ('hit_light', numba.boolean),
    ('direction', numba.int64),
])
class Path:
    # path is a stack of rays and (will have) some methods on them
    def __init__(self, ray, direction):
        self.ray = ray
        self.hit_light = False
        self.direction = direction


node_type.define(BoxStackNode.class_type.instance_type)
box_type.define(Box.class_type.instance_type)
ray_type.define(Ray.class_type.instance_type)


@numba.jit(nogil=True)
def jit_path_build(ray: Ray):
    return Path(ray, Direction.FROM_CAMERA.value)


if __name__ == '__main__':
    r = Ray(ZEROS, UNIT_Z)
    jit_path_build(r)