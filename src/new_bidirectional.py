from camera import Camera
from primitives import Ray, Box, unit, point
from routines import generate_light_ray, BRDF_sample, BRDF_function, BRDF_pdf, geometry_term
from collision import visibility_test, traverse_bvh
from constants import *
import numba
from utils import timed


#@numba.njit
def extend_path(path, root):
    for _ in range(MAX_BOUNCES):
        ray = path[-1]
        triangle, t = traverse_bvh(root, ray)
        if triangle is not None:
            # generate new ray
            #  new vectors
            origin = ray.origin + ray.direction * t
            direction = BRDF_sample(triangle.material, -1 * ray.direction, triangle.normal, Direction.FROM_CAMERA.value)
            new_ray = Ray(origin, direction)

            # probability, weight, and color updates
            

            #  store info from triangle
            new_ray.normal = triangle.normal
            new_ray.material = triangle.material
            new_ray.local_color = triangle.color

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