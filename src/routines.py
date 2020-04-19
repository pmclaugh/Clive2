import numba
import numpy as np
from constants import *
from primitives import Ray, Path, Triangle, Box, BoxStack, unit


@numba.jit(nogil=True, fastmath=True)
def ray_triangle_intersect(ray: Ray, triangle: Triangle):
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
    if t > FLOAT_TOLERANCE:
        return t
    else:
        return None


@numba.jit(nogil=True, fastmath=True)
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


@numba.jit(nogil=True)
def visibility_test(root: Box, ray_a: Ray, ray_b: Ray):
    delta = ray_b.origin - ray_a.origin
    least_t = np.linalg.norm(delta)
    test_ray = Ray(ray_a.origin, delta / least_t)
    stack = BoxStack()
    stack.push(root)
    while stack.size:
        box = stack.pop()
        if box.left is not None or box.right is not None:
            if bvh_hit_inner(test_ray, box, least_t):
                stack.push(box.left)
                stack.push(box.right)
        else:
            hit, t = bvh_hit_leaf(test_ray, box, least_t)
            if hit is not None and t < least_t:
                return False
    return True


@numba.jit(nogil=True)
def traverse_bvh(root: Box, ray: Ray):
    least_t = np.inf
    least_hit = None
    stack = BoxStack()
    stack.push(root)
    while stack.size:
        box = stack.pop()
        if box.left is not None or box.right is not None:
            if bvh_hit_inner(ray, box, least_t):
                stack.push(box.left)
                stack.push(box.right)
        else:
            hit, t = bvh_hit_leaf(ray, box, least_t)
            if hit is not None and t < least_t:
                least_hit = hit
                least_t = t

    return least_hit, least_t


@numba.jit(nogil=True, fastmath=True)
def bvh_hit_inner(ray: Ray, box: Box, least_t: float):
    hit, t_low, t_high = ray_box_intersect(ray, box)
    return hit and t_low <= least_t


@numba.jit(nogil=True, fastmath=True)
def bvh_hit_leaf(ray: Ray, box: Box, least_t):
    hit, t_low, t_high = ray_box_intersect(ray, box)
    if not hit:
        return None, least_t
    least_hit = None
    for triangle in box.triangles:
        t = ray_triangle_intersect(ray, triangle)
        if t is not None and 0 < t < least_t:
            least_t = t
            least_hit = triangle
    return least_hit, least_t


@numba.jit(nogil=True)
def generate_path(root: Box, ray: Ray, direction, max_bounces=4, rr_chance=0.1, stop_for_light=False):
    path = Path(ray, direction)
    while path.ray.bounces < max_bounces or np.random.random() < rr_chance:
        extend_path(path, root)
        if path.ray.bounces >= max_bounces:
            path.ray.p *= rr_chance
        if stop_for_light and path.hit_light:
            return path
    return path


@numba.jit(nogil=True)
def extend_path(path: Path, root: Box):
    triangle, t = traverse_bvh(root, path.ray)
    if triangle is not None:
        # generate new ray
        new_origin = (path.ray.origin + path.ray.direction * t + triangle.normal * COLLISION_OFFSET).astype(np.float32)
        new_direction = sample_BRDF(triangle.material, path.ray.direction, triangle.normal, path.direction)
        new_ray = Ray(new_origin, new_direction)

        # transfer ray attributes and shade
        cos_theta = evaluate_BRDF(triangle.material, path.ray.direction, triangle.normal, new_direction, path.direction)
        new_ray.color = (path.ray.color * triangle.color * cos_theta).astype(np.float32)
        new_ray.p = path.ray.p * cos_theta
        new_ray.bounces = path.ray.bounces + 1

        # store new ray
        path.ray.next = new_ray
        new_ray.prev = path.ray
        path.ray = new_ray
        if triangle.emitter:
            path.hit_light = True


@numba.jit(nogil=True)
def local_orthonormal_system(z):
    # todo: this should take triangle, point and do normal smoothing if available
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
    return (x * x_axis + y * y_axis + z_axis * np.sqrt(np.maximum(0., 1. - u1))).astype(np.float32)


@numba.jit(nogil=True)
def random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis):
    u1 = np.random.random()
    u2 = np.random.random()
    r = np.sqrt(1 - u1 * u1)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x_axis * x + y_axis * y + z_axis * u1).astype(np.float32)


@numba.jit(nogil=True)
def specular_reflection(direction, normal):
    return (2 * np.dot(-1 * direction, normal) * normal + direction).astype(np.float32)


@numba.jit(nogil=True)
def sample_BRDF(material, incident_direction, incident_normal, path_direction):
    x, y, z = local_orthonormal_system(incident_normal)
    if material == Material.DIFFUSE.value:
        if path_direction == Direction.FROM_CAMERA.value:
            return random_hemisphere_cosine_weighted(x, y, z)
        else:
            return random_hemisphere_uniform_weighted(x, y, z)
    elif material == Material.SPECULAR.value:
        return specular_reflection(incident_direction, incident_normal)
    else:
        return incident_normal


@numba.jit(nogil=True)
def evaluate_BRDF(material, incident_direction, incident_normal, exitant_direction, path_direction):
    if material == Material.DIFFUSE.value:
        if path_direction == Direction.FROM_CAMERA.value:
            return 1 / np.pi
        else:
            return np.dot(-1 * incident_direction, incident_normal)
    elif material == Material.SPECULAR.value:
        return 1
    else:
        return 0


@numba.jit(nogil=True)
def generate_light_ray(box: Box):
    light_index = np.random.randint(0, len(box.lights))
    light = box.lights[light_index]
    light_origin = light.sample_surface().astype(np.float32)
    x, y, z = local_orthonormal_system(light.normal)
    light_direction = random_hemisphere_cosine_weighted(x, y, z).astype(np.float32)
    return Ray(light_origin, light_direction)


if __name__ == '__main__':
    print(specular_reflection(-1 * UNIT_Z, UNIT_Z))
