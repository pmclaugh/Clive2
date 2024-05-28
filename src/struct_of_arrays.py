import numpy as np
from camera import Camera
import numba
import objloader
from constants import *
from bvh import construct_BVH, np_flatten_bvh, BoxGroup, TriangleGroup
import cv2
from utils import timed, dir_to_color
import metalcompute as mc
from struct_types import Ray


class RayGroup:
    def __init__(self,
                 ray_origins,
                 ray_directions):
        self.origin = ray_origins
        self.direction = ray_directions
        self.inv_direction = 1 / self.direction
        self.sign = (self.inv_direction < 0).astype(np.uint8)
        self.color = np.ones_like(ray_origins)
        self.importance = np.ones(ray_origins.shape[0])

    @classmethod
    def from_camera(cls, camera: Camera):
        return cls(*camera.ray_batch())

    @classmethod
    def from_raygroup(cls, raygroup):
        return cls(raygroup.origin, raygroup.direction)


class Triangle:
    v0 = np.zeros(3, dtype=np.float64)
    v1 = np.zeros(3, dtype=np.float64)
    v2 = np.zeros(3, dtype=np.float64)
    n0 = np.zeros(3, dtype=np.float64)
    n1 = np.zeros(3, dtype=np.float64)
    n2 = np.zeros(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    t2 = np.zeros(3, dtype=np.float64)

    @property
    def min(self):
        return np.minimum(self.v0, np.minimum(self.v1, self.v2))

    @property
    def max(self):
        return np.maximum(self.v0, np.maximum(self.v1, self.v2))

    @property
    def n(self):
        a = cross(self.v1 - self.v0, self.v2 - self.v0)
        return a / np.linalg.norm(a)


def load_obj(obj_path):
    obj = objloader.Obj.open(obj_path)
    triangles = []
    for i, ((v0, n0, t0), (v1, n1, t1), (v2, n2, t2)) in enumerate(zip(*[iter(obj.face)] * 3)):
        triangle = Triangle()
        # vertices
        triangle.v0 = np.array(obj.vert[v0 - 1])
        triangle.v1 = np.array(obj.vert[v1 - 1])
        triangle.v2 = np.array(obj.vert[v2 - 1])

        # normals
        triangle.n0 = np.array(obj.norm[n0 - 1]) if n0 is not None else INVALID
        triangle.n1 = np.array(obj.norm[n1 - 1]) if n1 is not None else INVALID
        triangle.n2 = np.array(obj.norm[n2 - 1]) if n2 is not None else INVALID

        # texture UVs
        triangle.t0 = np.array(obj.text[t0 - 1]) if t0 is not None else INVALID
        triangle.t1 = np.array(obj.text[t1 - 1]) if t1 is not None else INVALID
        triangle.t2 = np.array(obj.text[t2 - 1]) if t2 is not None else INVALID

        triangles.append(triangle)
    return triangles


@numba.jit(nogil=True, fastmath=True)
def ray_box_intersect(box_min, box_max, ray_origin, ray_inv_direction):
    min_minus = (box_min - ray_origin) * ray_inv_direction
    max_minus = (box_max - ray_origin) * ray_inv_direction
    mins = np.minimum(min_minus, max_minus)
    maxes = np.maximum(min_minus, max_minus)

    if mins[0] > maxes[1] or mins[1] > maxes[0]:
        return False, 0., 0.

    tmin = max(mins[0], mins[1])
    tmax = min(maxes[0], maxes[1])

    if tmin > maxes[2] or mins[2] > tmax:
        return False, 0., 0.

    tmin = max(tmin, mins[2])
    tmax = min(tmax, maxes[2])

    if tmax > 0:
        return True, tmin, tmax
    else:
        return False, 0., 0.


@numba.jit(nogil=True, fastmath=True)
def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


@numba.jit(nogil=True, fastmath=True)
def ray_triangle_intersect(ray_origin, ray_direction, triangle_n, triangle_v0, triangle_e1, triangle_e2):

    if np.dot(ray_direction, triangle_n) >= 0:
        return -1.

    h = cross(ray_direction, triangle_e2)
    a = np.dot(h, triangle_e1)
    if a <= 0:
        return -1.

    f = 1. / a
    s = ray_origin - triangle_v0
    u = f * np.dot(s, h)
    if u < 0. or u > 1.:
        return -1.
    q = cross(s, triangle_e1)
    v = f * np.dot(q, ray_direction)
    if v < 0. or v > 1.:
        return -1.

    if (1 - u - v) < 0. or (1 - u - v) > 1.:
        return -1.

    t = f * np.dot(triangle_e2, q)
    if t > COLLISION_SHIFT:
        return float(t)
    else:
        return -1.


@numba.njit(nogil=True, fastmath=True)
def hit_bvh(origin, direction, inv_direction, box_mins, box_maxes, box_lefts, box_rights, n, v0, v1, v2):
    best_t = np.inf
    best_i = -1
    stack = [0]
    while stack:
        box_index = stack.pop()
        hit, t_min, t_max = ray_box_intersect(box_mins[box_index], box_maxes[box_index], origin, inv_direction)
        if not hit or t_min > best_t:
            continue
        if box_rights[box_index] == 0:
            # inner node
            stack.append(box_lefts[box_index])
            stack.append(box_lefts[box_index] + 1)
        else:
            for i in range(box_lefts[box_index], box_rights[box_index]):
                t = ray_triangle_intersect(origin, direction, n[i], v0[i], v1[i] - v0[i], v2[i] - v0[i])
                if 0 < t < best_t:
                    best_t = t
                    best_i = i
    return best_t, best_i


@numba.njit(nogil=True, fastmath=True)
def hit_basic(origin, direction, v0, v1, v2):
    best_t = np.inf
    best_i = -1
    for i in range(v0.shape[0]):
        t = ray_triangle_intersect(origin, direction, v0[i], v1[i] - v0[i], v2[i] - v0[i])
        if 0 < t < best_t:
            best_t = t
            best_i = i
    return best_t, best_i


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _bounce(ray_origins, ray_directions, ray_inv_directions, box_mins, box_maxes, box_lefts, box_rights, v0, v1, v2, n, n0, n1, n2, t0, t1, t2):
    output_color = np.zeros_like(ray_origins)
    for i in numba.prange(ray_origins.shape[0]):
        for j in range(ray_origins.shape[1]):
            best_t, best_i = hit_bvh(ray_origins[i][j], ray_directions[i][j], ray_inv_directions[i][j], box_mins, box_maxes, box_lefts, box_rights, n, v0, v1, v2)
            if best_i != -1:
                output_color[i][j] = n[best_i] / 2 + .5

    return output_color


@timed
def bounce(raygroup: RayGroup, bvh: BoxGroup, triangles: TriangleGroup):
    colors = _bounce(raygroup.origin, raygroup.direction, raygroup.inv_direction, bvh.min, bvh.max, bvh.left, bvh.right, triangles.v0, triangles.v1, triangles.v2, triangles.n, triangles.n0, triangles.n1, triangles.n2, triangles.t0, triangles.t1, triangles.t2)
    return colors


if __name__ == '__main__':
    tris = load_obj('../resources/teapot.obj')
    bvh = construct_BVH(tris)
    c = Camera(
        center=np.array([0, 0, -5]),
        direction=np.array([0, 0, 1]),
    )
    rays = c.ray_batch_numpy()

    boxes, triangles = np_flatten_bvh(bvh)

    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()

    kernel_fn = dev.kernel(kernel).function("bounce")
    buf_0 = rays.flatten()
    buf_1 = boxes.flatten()
    buf_2 = triangles.flatten()
    buf_3 = dev.buffer(np.size(rays) * 16)
    buf_4 = dev.buffer(16)

    kernel_fn(rays.size, buf_0, buf_1, buf_2, buf_3, buf_4)

    retrieved_image = np.frombuffer(buf_3, dtype=np.float32).reshape(rays.shape[0], rays.shape[1], 4)
    retrieved_values = np.frombuffer(buf_4, dtype=np.int32)

    print("ok")
    print(retrieved_image.shape)
    print(retrieved_values)

    # open a window to display the image
    cv2.imshow('image', retrieved_image[:, :, :3])
    cv2.waitKey(0)
