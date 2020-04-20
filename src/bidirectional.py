from camera import Camera
from primitives import Box, Path, unit, point
from routines import generate_light_ray, generate_path, visibility_test, path_pop, path_push, BRDF_function, BRDF_pdf
from constants import *
import numba
from utils import timed


@timed
# @numba.jit(nogil=True)
def bidirectional_screen_sample(camera: Camera, root: Box, samples=5):
    for _ in range(samples):
        for i in range(camera.pixel_height):
            for j in range(camera.pixel_width):
                light_path = generate_path(root, generate_light_ray(root), direction=Direction.FROM_EMITTER.value)
                camera_path = generate_path(root, camera.make_ray(i, j), direction=Direction.FROM_CAMERA.value)
                camera.image[i][j] += bidirectional_sample(root, camera_path, light_path)
    camera.samples += 1


@numba.jit(nogil=True)
def sum_probabilities(s, t):
    # given real sample X(s,t), calculate sum of p(X(s,t)) for all possible values of s, t
    sigma_p = s.ray.p * t.ray.p
    # NB these are actually len - 1, which would actually be s or t - 2 since veach counts from 1
    s_len = s.ray.bounces
    t_len = t.ray.bounces
    # push all s onto t, tally p
    for _ in range(s_len):
        path_push(t, path_pop(s))
        sigma_p += t.ray.p * s.ray.p
    # restore original state
    for _ in range(s_len):
        path_push(s, path_pop(t))

    # push all t onto s, tally p
    for _ in range(t_len):
        path_push(s, path_pop(t))
        sigma_p += t.ray.p * s.ray.p
    # restore original state
    for _ in range(t_len):
        path_push(t, path_pop(s))

    return sigma_p


@numba.jit(nogil=True)
def bidirectional_sample(root: Box, camera_path: Path, light_path: Path):
    # iterates over all possible s and t available from light_path and camera_path, respectively
    camera_stack = Path(None, Direction.FROM_CAMERA.value)
    light_stack = Path(None, Direction.FROM_EMITTER.value)
    result = point(0, 0, 0)
    while camera_path.ray.prev is not None:
        while light_path.ray.prev is not None:
            # Sample for this s, t
            # dot the normals to skip pointless visibility checks
            # NB this needs adjustment when transmissive materials are introduced
            dir_l_to_c = unit(camera_path.ray.origin - light_path.ray.origin)
            if np.dot(camera_path.ray.normal, -1 * dir_l_to_c) > 0 and np.dot(light_path.ray.normal, dir_l_to_c) > 0:
                if visibility_test(root, camera_path.ray, light_path.ray):
                    p = sum_probabilities(camera_path, light_path)
                    light_join_f = BRDF_function(light_path.ray.material, light_path.ray.prev.direction, light_path.ray.normal, dir_l_to_c, light_path.direction)
                    camera_join_f = BRDF_function(camera_path.ray.material, camera_path.ray.prev.direction, camera_path.ray.normal, -1 * dir_l_to_c, camera_path.direction)
                    f = camera_path.ray.color * light_path.ray.color * light_join_f * camera_join_f
                    result += np.maximum(0, f / p)

            # iterate s down
            z = path_pop(light_path)
            path_push(light_stack, z)

        # iterate t down
        z = path_pop(camera_path)
        path_push(camera_stack, z)

        # reset s
        while light_stack.ray is not None:
            z = path_pop(light_stack)
            path_push(light_path, z)

    return result