from camera import Camera
from primitives import Box, Path, unit, point
from routines import generate_light_ray, generate_path, visibility_test, path_pop, path_push, BRDF_function, BRDF_pdf, geometry_term
from constants import *
import numba
from utils import timed


@timed
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
    # given real sample X(s,t), calculate sum of p(X(s,t)) for all used values of s, t
    sigma_p = s.ray.p * t.ray.p * geometry_term(s.ray, t.ray)
    s_len = s.ray.bounces - 1
    t_len = t.ray.bounces - 1

    # push all s onto t, tally p
    for _ in range(s_len):
        path_push(t, path_pop(s))
        if s.ray is not None:
            sigma_p += t.ray.p * s.ray.p * geometry_term(s.ray, t.ray)
        else:
            sigma_p += t.ray.p
    # restore original state
    for _ in range(s_len):
        path_push(s, path_pop(t))

    # push all t onto s, tally p
    for _ in range(t_len):
        path_push(s, path_pop(t))
        if t.ray is not None:
            sigma_p += t.ray.p * s.ray.p * geometry_term(s.ray, t.ray)
        else:
            sigma_p += s.ray.p
    # restore original state
    for _ in range(t_len):
        path_push(t, path_pop(s))

    return sigma_p


@numba.jit(nogil=True)
def bidirectional_sample(root: Box, camera_path: Path, light_path: Path):
    # todo this is very slow, and also has some subtle issue i can't get rid of. should rewrite and simplify
    #  the slowdown is mostly in the iteration. I think paths need to be lists.
    #  probably need to go back to the pl pc plan too
    # iterates over all possible s and t available from light_path and camera_path, respectively
    camera_stack = Path(None, Direction.STORAGE.value)
    light_stack = Path(None, Direction.STORAGE.value)
    result = point(0, 0, 0)
    samples = 0
    while camera_path.ray.prev is not None:
        while light_path.ray.prev is not None:
            # Sample for this s, t
            # dot the normals to skip pointless visibility checks
            # NB this needs adjustment when transmissive materials are introduced
            dir_l_to_c = unit(camera_path.ray.origin - light_path.ray.origin)
            if np.dot(camera_path.ray.normal, -1 * dir_l_to_c) > 0 and np.dot(light_path.ray.normal, dir_l_to_c) > 0:
                if visibility_test(root, camera_path.ray, light_path.ray):
                    p = sum_probabilities(light_path, camera_path)
                    if camera_path.ray.prev is None:
                        # iirc this one's a weird one. skipping for now.
                        camera_join_f = 0
                    else:
                        camera_join_f = BRDF_function(camera_path.ray.material, -1 * camera_path.ray.prev.direction,
                                                      camera_path.ray.normal, -1 * dir_l_to_c, camera_path.direction)
                    if light_path.ray.prev is None:
                        light_join_f = np.dot(light_path.ray.normal, dir_l_to_c)
                    else:
                        light_join_f = BRDF_function(light_path.ray.material, -1 * light_path.ray.prev.direction,
                                                     light_path.ray.normal, dir_l_to_c, light_path.direction)

                    f = camera_path.ray.color * light_path.ray.color * light_join_f * camera_join_f * geometry_term(light_path.ray, camera_path.ray)
                    result += f / p
            samples += 1

            # iterate s down
            if light_path.ray is not None:
                z = path_pop(light_path)
                path_push(light_stack, z)

        # iterate t down
        if camera_path.ray is not None:
            z = path_pop(camera_path)
            path_push(camera_stack, z)

        # reset s
        while light_stack.ray is not None:
            z = path_pop(light_stack)
            path_push(light_path, z)

    return result / samples
