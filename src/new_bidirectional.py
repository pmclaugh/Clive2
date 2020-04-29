from camera import Camera
from primitives import Ray, Box, unit, point
from routines import generate_light_ray, BRDF_sample, BRDF_function, BRDF_pdf, geometry_term
from collision import visibility_test, traverse_bvh
from constants import *
import numba
from utils import timed

# G = geometry_term(path.ray, ray)
#         path.ray.direction = unit(ray.origin - path.ray.origin)
#         if path.ray.prev is None:
#             # pushing onto stack of 1, just propagate p and color (?)
#             ray.color = path.ray.color * G
#             ray.p = path.ray.p * G
#         else:
#             # pushing onto stack of 2 or more, do some work
#             brdf = BRDF_function(path.ray.material, -1 * path.ray.prev.direction, path.ray.normal, path.ray.direction,
#                                        path.direction)
#             ray.color = path.ray.local_color * path.ray.color * brdf * G
#             ray.p = path.ray.p * G * BRDF_pdf(path.ray.material, -1 * path.ray.prev.direction, path.ray.normal, path.ray.direction,
#                                           path.direction)
#         ray.bounces = path.ray.bounces + 1


#@numba.njit
def extend_path(path, root, d):
    not_d = (d + 1) % 2
    for i in range(MAX_BOUNCES):
        ray = path[-1]
        triangle, t = traverse_bvh(root, ray)
        if triangle is not None:
            # generate new ray
            #  new vectors
            origin = ray.origin + ray.direction * t
            direction = BRDF_sample(triangle.material, -1 * ray.direction, triangle.normal, d)
            new_ray = Ray(origin, direction)

            #  store info from triangle
            new_ray.normal = triangle.normal
            new_ray.material = triangle.material
            new_ray.local_color = triangle.color

            # probability, weight, and color updates
            G = geometry_term(ray, new_ray)
            # notes from reading before stopping:
            # camera rays only need to totient pc and light rays only need to totient pl
            # otherwise what I should be caching is local pdf values and geometry terms




            path.append(new_ray)
        else:
            break


#@numba.njit
def bidirectional_pixel_sample(camera_path, light_path, root):
    pass







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