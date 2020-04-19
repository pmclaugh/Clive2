from camera import Camera
from primitives import Box, Path
from routines import generate_light_ray, generate_path, visibility_test
from constants import *
import numba


@numba.jit(nogil=True)
def bidirectional_screen_sample(camera: Camera, root: Box, samples=5):
    image = camera.image * 0
    for _ in range(samples):
        for i in range(camera.pixel_height):
            for j in range(camera.pixel_width):
                light_path = generate_path(root, generate_light_ray(root))
                camera_path = generate_path(root, camera.make_ray(i, j))
                image[i][j] += bidirectional_sample(root, camera_path, light_path)
    camera.image = image / samples


@numba.jit(nogil=True)
def bidirectional_sample(root: Box, camera_path: Path, light_path: Path):
    # barebones just making sure all the collision and path gen works
    camera_ray = camera_path.ray
    accumulated = BLACK.copy()
    while camera_ray is not None:
        light_ray = light_path.ray
        while light_ray is not None:
            if visibility_test(root, light_ray, camera_ray):
                accumulated += camera_path.ray.color * light_path.ray.color
            light_ray = light_ray.prev
        camera_ray = camera_ray.prev
    return accumulated / (camera_path.ray.bounces * light_path.ray.bounces)
