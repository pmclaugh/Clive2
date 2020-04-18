from camera import Camera
from primitives import Box, Path, Ray
from routines import generate_light_ray, generate_path, visibility_test
from constants import *


def bidirectional_screen_sample(camera: Camera, root: Box, sample_number):
    image = camera.image * 0
    for i in range(camera.pixel_height):
        for j in range(camera.pixel_width):
            camera_path = generate_path(root, camera.make_ray(i, j))
            light_path = generate_path(root, generate_light_ray(root))
            image[i][j] = bidirectional_sample(root, camera_path, light_path)
    return image


def bidirectional_sample(root: Box, camera_path: Path, light_path: Path):
    # barebones just making sure all the collision and path gen works
    if visibility_test(root, camera_path.ray, light_path.ray):
        return camera_path.ray.color * light_path.ray.color
    return BLACK
