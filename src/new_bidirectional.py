from camera import Camera
from primitives import Ray, Box, unit, point
from routines import generate_light_ray, BRDF_sample, BRDF_function, BRDF_pdf, geometry_term
from collision import visibility_test, traverse_bvh
from constants import *
import numba
from utils import timed


@numba.njit
def extend_path(path, root, path_direction):
    for i in range(MAX_BOUNCES):
        ray = path[-1]
        triangle, t = traverse_bvh(root, ray)
        if triangle is not None:
            # generate new ray
            #  new vectors
            origin = ray.origin + ray.direction * t
            direction = BRDF_sample(triangle.material, -1 * ray.direction, triangle.normal, path_direction)
            new_ray = Ray(origin, direction)

            #  store info from triangle
            new_ray.normal = triangle.normal
            new_ray.material = triangle.material
            new_ray.local_color = triangle.color

            # probability, weight, and color updates
            G = geometry_term(ray, new_ray)

            if i == 0:
                # only need to multiply by G because p of this direction is already stored at creation
                new_ray.p = ray.p * G
                # same deal, brdf of source is just 1
                new_ray.color = ray.color * G
            else:
                # so the idea here is that each vertex has information about everything up to it but not including it,
                # because we can't be sure of anything about the final bounce until we know the joining vertex
                new_ray.p = ray.p * G * BRDF_pdf(ray.material, -1 * path[-2].direction, ray.normal, ray.direction, path_direction)
                new_ray.color = ray.color * ray.local_color * G * BRDF_function(ray.material, -1 * path[-2].direction,
                                                                                ray.normal, ray.direction, path_direction)
                # I will also want to cache pdf results and Gs for individual bounces but waiting to write that

            path.append(new_ray)
        else:
            break


@numba.njit
def bidirectional_pixel_sample(camera_path, light_path, root):
    extend_path(camera_path, root, Direction.FROM_CAMERA.value)
    extend_path(light_path, root, Direction.FROM_EMITTER.value)
    # nb i am leaving true s == 0 and t == 0 out for now because they're weird. just trying to get the core done
    # skipping the 1s for a moment as well to keep it simplest
    samples = 0
    total = ZEROS.copy()
    for t in range(1, len(camera_path)):
        for s in range(1, len(light_path)):
            camera_vertex = camera_path[t]
            light_vertex = light_path[t]
            dir_l_to_c = unit(camera_vertex.origin - light_vertex.origin)
            if np.dot(camera_vertex.normal, -1 * dir_l_to_c) > 0 and np.dot(light_vertex.normal, dir_l_to_c) > 0:
                if visibility_test(root, camera_vertex, light_vertex):
                    camera_brdf = BRDF_function(camera_vertex.material, -1 * camera_path[t - 1].direction,
                                                camera_vertex.normal, -1 * dir_l_to_c, Direction.FROM_CAMERA.value)
                    camera_pdf = BRDF_pdf(camera_vertex.material, -1 * camera_path[t - 1].direction,
                                                camera_vertex.normal, -1 * dir_l_to_c, Direction.FROM_CAMERA.value)
                    light_brdf = BRDF_function(light_vertex.material, -1 * light_path[t - 1].direction,
                                                light_vertex.normal, dir_l_to_c, Direction.FROM_EMITTER.value)
                    light_pdf = BRDF_pdf(light_vertex.material, -1 * light_path[t - 1].direction,
                                          light_vertex.normal, dir_l_to_c, Direction.FROM_EMITTER.value)
                    G = geometry_term(camera_vertex, light_vertex)
                    f = camera_vertex.color * camera_vertex.local_color * camera_brdf * \
                        light_vertex.color * light_vertex.local_color * light_brdf
                    p = camera_vertex.p * light_vertex.p #* camera_pdf * light_pdf
                    total += f / p
            samples += 1
    return total / samples


@timed
def bidirectional_screen_sample(camera: Camera, root: Box, samples=5):
    for _ in range(samples):
        for i in range(camera.pixel_height):
            for j in range(camera.pixel_width):
                light_path = numba.typed.List()
                light_path.append(generate_light_ray(root))
                camera_path = numba.typed.List()
                camera_path.append(camera.make_ray(i, j))
                camera.image[i][j] += bidirectional_pixel_sample(camera_path, light_path, root)
    camera.samples += 1