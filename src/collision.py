from primitives import Ray, Triangle, FastBox
import numpy as np
import numba
from constants import COLLISION_SHIFT


@numba.jit(nogil=True, fastmath=True)
def ray_triangle_intersect(ray: Ray, triangle: Triangle):
    if np.dot(ray.direction, triangle.normal) >= 0:
        return None
    h = np.cross(ray.direction, triangle.e2)
    a = np.dot(h, triangle.e1)

    if a <= 0:
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

    if (1 - u - v) < 0. or (1 - u - v) > 1.:
        return None

    t = f * np.dot(triangle.e2, q)
    if t > COLLISION_SHIFT:
        return t
    else:
        return None


@numba.jit(nogil=True, fastmath=True)
def ray_box_intersect(ray: Ray, box: FastBox):
    min_minus = (box.min - ray.origin) * ray.inv_direction
    max_minus = (box.max - ray.origin) * ray.inv_direction
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
def bvh_hit_inner(ray: Ray, box: FastBox, least_t: float):
    hit, t_low, t_high = ray_box_intersect(ray, box)
    return hit and t_low <= least_t


@numba.jit(nogil=True, fastmath=True)
def bvh_hit_leaf(ray: Ray, box: FastBox, triangles, least_t):
    hit, t_low, t_high = ray_box_intersect(ray, box)
    if not hit:
        return None, least_t
    least_hit = None
    for triangle in triangles:
        t = ray_triangle_intersect(ray, triangle)
        if t is not None and 0 < t < least_t:
            least_t = t
            least_hit = triangle
    return least_hit, least_t


@numba.njit
def visibility_test(boxes, triangles, ray_a: Ray, ray_b: Ray):
    delta = ray_b.origin - ray_a.origin
    least_t = np.linalg.norm(delta)
    direction = delta / least_t
    if np.dot(ray_a.normal, direction) <= 0 or np.dot(ray_b.normal, -1 * direction) <= 0:
        return False
    test_ray = Ray(ray_a.origin, direction)
    stack = [boxes[0]]
    while stack:
        box = stack.pop()
        if box.right == 0:
            if bvh_hit_inner(test_ray, box, least_t):
                stack.append(boxes[box.left])
                stack.append(boxes[box.left + 1])
        else:
            hit, t = bvh_hit_leaf(test_ray, box, triangles[box.left:box.right], least_t)
            if hit is not None and t < least_t:
                return False
    return True


@numba.njit
def traverse_bvh(boxes, triangles, ray: Ray):
    least_t = np.inf
    least_hit = None
    stack = [boxes[0]]
    while stack:
        box = stack.pop()
        if box.right == 0:
            if bvh_hit_inner(ray, box, least_t):
                stack.append(boxes[box.left])
                stack.append(boxes[box.left + 1])
        else:
            hit, t = bvh_hit_leaf(ray, box, triangles[box.left:box.right], least_t)
            if hit is not None and t < least_t:
                least_hit = hit
                least_t = t

    return least_hit, least_t
