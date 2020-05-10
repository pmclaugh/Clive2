from camera import Camera
from primitives import Ray, FastBox, unit, point
from routines import generate_light_ray, BRDF_sample, BRDF_function, BRDF_pdf, geometry_term
from collision import visibility_test, traverse_bvh
from constants import *
import numba
from utils import timed


@numba.njit
def extend_path(path, boxes, triangles, path_direction):
    for i in range(MAX_BOUNCES):
        ray = path[-1]
        triangle, t = traverse_bvh(boxes, triangles, ray)
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
            if path_direction == Direction.FROM_CAMERA.value and triangle.emitter:
                new_ray.hit_light = True

            # probability, weight, and color updates
            G = geometry_term(ray, new_ray)

            if i == 0:
                # only need to multiply by G because p of this direction is already stored at creation
                new_ray.p = ray.p * G
                # same deal, "brdf" of source is already in ray.color
                new_ray.color = ray.color * G
                new_ray.G = G
            else:
                # so the idea here is that each vertex has information about everything up to it but not including it,
                # because we can't be sure of anything about the final bounce until we know the joining vertex
                bounce_p = BRDF_pdf(ray.material, -1 * path[-2].direction, ray.normal, ray.direction, path_direction)
                new_ray.p = ray.p * G * bounce_p
                new_ray.color = ray.color * ray.local_color * G * BRDF_function(ray.material, -1 * path[-2].direction,
                                                                                ray.normal, ray.direction, path_direction)
                new_ray.G = G
                ray.local_p = bounce_p

            path.append(new_ray)
        else:
            break


# using this might be a little inefficient but it makes the code a lot simpler
@numba.njit
def dir(a, b):
    # return direction from A to B
    return unit(b.origin - a.origin)


@numba.njit
def bidirectional_pixel_sample(camera_path, light_path, boxes, triangles):
    extend_path(camera_path, boxes, triangles, Direction.FROM_CAMERA.value)
    extend_path(light_path, boxes, triangles, Direction.FROM_EMITTER.value)
    samples = [[WHITE * -1 for t in range(len(camera_path) + 1)] for s in range(len(light_path) + 1)]
    for t in range(len(camera_path) + 1):
        for s in range(len(light_path) + 1):
            if s == 0 or t == 0:
                # no visibility test needed
                if t == 0:
                    continue
                if not camera_path[t - 1].hit_light:
                    continue
                light_p = 1.0
                light_f = WHITE
                camera_p = camera_path[t - 1].p
                camera_f = camera_path[t - 1].color
            else:
                if t == 1:
                    # project new camera_vertex for visibility test
                    continue
                # need visibility test
                light_vertex = light_path[s - 1]
                camera_vertex = camera_path[t - 1]
                dir_l_to_c = unit(camera_vertex.origin - light_vertex.origin)
                if visibility_test(boxes, triangles, camera_vertex, light_vertex):
                    if t == 1:
                        camera_brdf = 1
                    else:
                        camera_brdf = BRDF_function(camera_vertex.material, -1 * camera_path[t - 2].direction,
                                                    camera_vertex.normal, -1 * dir_l_to_c, Direction.FROM_CAMERA.value)
                    if s == 1:
                        light_brdf = np.dot(dir_l_to_c, light_vertex.normal)
                    else:
                        light_brdf = BRDF_function(light_vertex.material, -1 * light_path[s - 2].direction,
                                                    light_vertex.normal, dir_l_to_c, Direction.FROM_EMITTER.value)
                    camera_f = camera_vertex.color * camera_vertex.local_color * camera_brdf
                    camera_p = camera_vertex.p
                    light_f = light_vertex.color * light_vertex.local_color * light_brdf
                    light_p = light_vertex.p
                else:
                    continue

            f = camera_f * light_f
            p = camera_p * light_p

            path = [light_path[i] if i < s else camera_path[s - i] for i in range(s + t)]
            p_ratios = np.zeros(s + t, dtype=np.float64)
            # adapted from Veach section 10.2
            # note that this does not reach the efficiency that he describes, kind of an intermediate state
            # where it is correct in terms of calculations but not efficient yet in minimizing computation
            for i in range(s + t):
                # i is the subscript in the denominator, computing ratios of p(i + 1) / p(i)
                if i == 0:
                    num = 1
                elif i == 1:
                    num = path[0].p
                else:
                    num = BRDF_pdf(path[i - 1].material, dir(path[i - 1], path[i - 2]), path[i - 1].normal,
                               dir(path[i - 1], path[i]), Direction.FROM_CAMERA.value) * geometry_term(path[i - 1], path[i])
                if i == s + t - 1:
                    denom = 1
                elif i == s + t - 2:
                    denom = path[-1].p
                else:
                    denom = BRDF_pdf(path[i + 1].material, dir(path[i + 1], path[i + 2]), path[i + 1].normal,
                                     dir(path[i + 1], path[i]), Direction.FROM_EMITTER.value) * geometry_term(path[i + 1], path[i])
                p_ratios[i] = num / denom

            # p_ratios is like [p1/p0, p2/p1, p3/p2, ... ]

            for k in range(1, s + t):
                p_ratios[k] = p_ratios[k] * p_ratios[k - 1]

            # p_ratios is like [p1/p0, p2/p0, p3/p0 ...]

            # p_ratios[s - 1] is ps/p0
            w = np.sum(p_ratios[s - 1] / p_ratios[:-2]) + 1 / p_ratios[s - 1]
            # w is sum(ps/pi) for all pi we actually consider

            sample = np.maximum(0, f / (p * w))
            samples[s][t] = sample
    return samples

# final optimized version probably looks like:
# do all visibility tests first, save all the directions, then do brdfs and pdfs for all possible joins
# make a grid of G ratios


@timed
@numba.njit
def bidirectional_screen_sample(camera: Camera, boxes, triangles, emitters):
    for i in range(camera.pixel_height):
        for j in range(camera.pixel_width):
            light_path = numba.typed.List()
            light_path.append(generate_light_ray(emitters))
            camera_path = numba.typed.List()
            camera_path.append(camera.make_ray(i, j))

            samples = bidirectional_pixel_sample(camera_path, light_path, boxes, triangles)
            for s, row in enumerate(samples):
                for t, sample in enumerate(row):
                    if np.greater(sample, 0).all():
                        camera.images[s][t][i][j] += sample
                        camera.sample_counts[s][t] += 1
                    camera.image[i][j] += sample
    camera.samples += 1