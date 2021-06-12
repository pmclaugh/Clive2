import numba
import numpy as np
from constants import *
from primitives import Ray, Path, Triangle, FastBox, unit
from utils import timed
from collision import traverse_bvh


# todo these 3 don't need to be split up, and need to be homogenized (ie used the same in bd and ud)
@numba.njit
def generate_path(boxes, triangles, ray: Ray, direction, max_bounces=4, rr_chance=0.1, stop_for_light=False):
    path = Path(ray, direction)
    while path.ray.bounces < max_bounces: # or np.random.random() < rr_chance:
        hit = extend_path(path, boxes, triangles)
        if not hit:
            break
        if path.ray.prev.bounces >= max_bounces:
            path.ray.p *= rr_chance
        if stop_for_light and path.hit_light:
            return path
    return path


@numba.njit
def extend_path(path: Path, boxes, triangles):
    triangle, t = traverse_bvh(boxes, triangles, path.ray)
    if triangle is not None:
        # generate new ray
        new_origin = path.ray.origin + path.ray.direction * t
        new_direction = brdf_sample(triangle.material, -1 * path.ray.direction, triangle.normal, path.direction)
        new_ray = Ray(new_origin, new_direction)
        new_ray.normal = triangle.normal
        new_ray.material = triangle.material
        new_ray.local_color = triangle.color
        if triangle.emitter:
            path.hit_light = True

        path_push(path, new_ray)
        return True
    else:
        return False


@numba.njit
def path_push(path: Path, ray: Ray):
    # update stuff appropriately
    G = geometry_term(path.ray, ray)
    path.ray.direction = unit(ray.origin - path.ray.origin)
    if path.ray.prev is None:
        # pushing onto stack of 1, just propagate p and color (?)
        ray.color = path.ray.color * G
        ray.p = path.ray.p * G
    else:
        # pushing onto stack of 2 or more, do some work
        brdf = brdf_function(path.ray.material, -1 * path.ray.prev.direction, path.ray.normal, path.ray.direction,
                             path.direction)
        ray.color = path.ray.local_color * path.ray.color * brdf * G
        ray.p = path.ray.p * G * brdf_pdf(path.ray.material, -1 * path.ray.prev.direction, path.ray.normal, path.ray.direction,
                                          path.direction)
    ray.bounces = path.ray.bounces + 1

    # store new ray
    ray.prev = path.ray
    path.ray = ray


@numba.njit
def local_orthonormal_system(z):
    # todo: this should take triangle, point and do normal smoothing if available
    if np.abs(z[0]) > np.abs(z[1]):
        axis = UNIT_Y
    else:
        axis = UNIT_X
    x = np.cross(axis, z)
    y = np.cross(z, x)
    return x, y, z


@numba.njit
def random_hemisphere_cosine_weighted(x_axis, y_axis, z_axis):
    u1 = np.random.random()
    u2 = np.random.random()
    r = np.sqrt(u1)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x * x_axis + y * y_axis + z_axis * np.sqrt(np.maximum(0., 1. - u1))


@numba.njit
def random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis):
    u1 = np.random.random()
    u2 = np.random.random()
    r = np.sqrt(1 - u1 * u1)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x_axis * x + y_axis * y + z_axis * u1


@numba.njit
def specular_reflection(direction, normal):
    return 2 * np.dot(direction, normal) * normal - direction


@numba.njit
def brdf_sample(material, input_direction, normal, path_direction):
    # in all BRDF routines, all directions must point away from the point being sampled
    # returns a new direction
    x, y, z = local_orthonormal_system(normal)
    if material == Material.DIFFUSE.value:
        if path_direction == Direction.FROM_CAMERA.value:
            return random_hemisphere_cosine_weighted(x, y, z)
        else:
            return random_hemisphere_uniform_weighted(x, y, z)
    elif material == Material.SPECULAR.value:
        return specular_reflection(input_direction, normal)
    else:
        return normal


@numba.njit
def brdf_function(material, incident_direction, incident_normal, exitant_direction, path_direction):
    # in all BRDF routines, all directions must point away from the point being sampled
    # returns albedo mask of this bounce
    if material == Material.DIFFUSE.value:
        if path_direction == Direction.FROM_CAMERA.value:
            return np.dot(exitant_direction, incident_normal) / np.pi
        else:
            return np.dot(incident_direction, incident_normal) / np.pi
    else:
        return 1


@numba.njit
def brdf_pdf(material, incident_direction, incident_normal, exitant_direction, path_direction):
    # in all BRDF routines, all directions must point away from the point being sampled
    # returns probability density of choosing exitant direction
    if material == Material.DIFFUSE.value:
        if path_direction == Direction.FROM_CAMERA.value:
            return np.dot(exitant_direction, incident_normal) / np.pi
        else:
            return 1 / (2 * np.pi)
    elif material == Material.SPECULAR.value:
        return 1.
    else:
        return 0.


@numba.njit
def geometry_term(a: Ray, b: Ray):
    # quantifies the probability of connecting two specific vertices.
    # used when joining paths in bidirectional
    delta = b.origin - a.origin
    t = np.linalg.norm(delta)
    direction = delta / t

    camera_cos = np.dot(a.normal, direction)
    light_cos = np.dot(b.normal, -1 * direction)

    return np.abs(camera_cos * light_cos) / (t * t)


@numba.njit
def generate_light_ray(emitters):

    light = emitters[np.random.randint(0, len(emitters))]
    light_origin = light.sample_surface()

    x, y, z = local_orthonormal_system(light.normal)
    light_direction = random_hemisphere_uniform_weighted(x, y, z)
    ray = Ray(light_origin, light_direction)

    ray.color = light.color
    ray.local_color = light.color
    ray.normal = light.normal

    # this seems made up
    ray.p = 1 / (2 * np.pi * light.surface_area)

    return ray
