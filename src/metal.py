import numpy as np
from camera import Camera
import objloader
from constants import INVALID, UNIT_X, UNIT_Y, UNIT_Z, RED, BLUE, GREEN, CYAN, WHITE
from bvh import construct_BVH, np_flatten_bvh
import cv2
from utils import timed
import metalcompute as mc
import time
from struct_types import Ray, Material, Path, Box
from datetime import datetime


class Triangle:
    v0 = np.zeros(3, dtype=np.float64)
    v1 = np.zeros(3, dtype=np.float64)
    v2 = np.zeros(3, dtype=np.float64)
    n0 = np.zeros(3, dtype=np.float64)
    n1 = np.zeros(3, dtype=np.float64)
    n2 = np.zeros(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    t2 = np.zeros(3, dtype=np.float64)
    material = 0
    emitter = False

    def __init__(self, v0, v1, v2, material=0, emitter=False):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material
        self.emitter = emitter


    @property
    def min(self):
        return np.minimum(self.v0, np.minimum(self.v1, self.v2))

    @property
    def max(self):
        return np.maximum(self.v0, np.maximum(self.v1, self.v2))

    @property
    def n(self):
        a = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        return a / np.linalg.norm(a)


def load_obj(obj_path):
    obj = objloader.Obj.open(obj_path)
    triangles = []
    for i, ((v0, n0, t0), (v1, n1, t1), (v2, n2, t2)) in enumerate(zip(*[iter(obj.face)] * 3)):
        triangle = Triangle(np.array(obj.vert[v0 - 1]),
                            np.array(obj.vert[v1 - 1]),
                            np.array(obj.vert[v2 - 1]))

        # normals
        triangle.n0 = np.array(obj.norm[n0 - 1]) if n0 is not None else INVALID
        triangle.n1 = np.array(obj.norm[n1 - 1]) if n1 is not None else INVALID
        triangle.n2 = np.array(obj.norm[n2 - 1]) if n2 is not None else INVALID

        # texture UVs
        triangle.t0 = np.array(obj.text[t0 - 1]) if t0 is not None else INVALID
        triangle.t1 = np.array(obj.text[t1 - 1]) if t1 is not None else INVALID
        triangle.t2 = np.array(obj.text[t2 - 1]) if t2 is not None else INVALID

        # material
        triangle.material = 0
        triangle.emitter = False

        triangles.append(triangle)
    return triangles


def get_materials():
    materials = np.zeros(7, dtype=Material)
    materials['color'] = np.zeros((7, 4), dtype=np.float32)
    materials['color'][0][:3] = RED
    materials['color'][1][:3] = BLUE
    materials['color'][2][:3] = GREEN
    materials['color'][3][:3] = CYAN
    materials['color'][4][:3] = WHITE
    materials['color'][5][:3] = WHITE
    materials['color'][6][:3] = WHITE
    materials['emission'] = np.zeros((7, 4), dtype=np.float32)
    materials['emission'][6] = np.array([1, 1, 1, 1])
    materials['type'] = 0
    materials[0]['type'] = 1
    return materials


def triangles_for_box(box_min, box_max):
    span = box_max - box_min
    left_bottom_back = box_min
    right_bottom_back = box_min + span * UNIT_X
    left_top_back = box_min + span * UNIT_Y
    left_bottom_front = box_min + span * UNIT_Z

    right_top_front = box_max
    left_top_front = box_max - span * UNIT_X
    right_bottom_front = box_max - span * UNIT_Y
    right_top_back = box_max - span * UNIT_Z

    shrink = np.array([.25, .99, .25], dtype=np.float32)
    tris = [
        # back wall
        Triangle(left_bottom_back, right_bottom_back, right_top_back, material=2),
        Triangle(left_bottom_back, right_top_back, left_top_back, material=2),
        # left wall
        Triangle(left_bottom_back, left_top_front, left_bottom_front, material=1),
        Triangle(left_bottom_back, left_top_back, left_top_front, material=1),
        # right wall
        Triangle(right_bottom_back, right_bottom_front, right_top_front, material=2),
        Triangle(right_bottom_back, right_top_front, right_top_back, material=2),
        # front wall
        Triangle(left_bottom_front, right_top_front, right_bottom_front, material=3),
        Triangle(left_bottom_front, left_top_front, right_top_front, material=3),
        # floor
        Triangle(left_bottom_back, right_bottom_front, right_bottom_back, material=4),
        Triangle(left_bottom_back, left_bottom_front, right_bottom_front, material=4),
        # ceiling
        Triangle(left_top_back, right_top_back, right_top_front, material=5),
        Triangle(left_top_back, right_top_front, left_top_front, material=5),
        # ceiling light # NB this assumes box is centered on the origin, at least wrt x and z
        Triangle(left_top_back * shrink, right_top_back * shrink, right_top_front * shrink, material=6, emitter=True),
        Triangle(left_top_back * shrink, right_top_front * shrink, left_top_front * shrink, material=6, emitter=True),
    ]
    return tris


def local_orthonormal_system(z):
    if np.abs(z[0]) > np.abs(z[1]):
        axis = UNIT_Y
    else:
        axis = UNIT_X
    x = np.cross(axis, z)
    y = np.cross(z, x)
    return x, y, z


def random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis):
    u1 = np.random.random()
    u2 = np.random.random()
    r = np.sqrt(1 - u1 * u1)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x_axis * x + y_axis * y + z_axis * u1


def fast_generate_light_rays(triangles, num_rays):
    emitters = np.array([[t['v0'], t['v1'], t['v2']] for t in triangles if t['is_light']])
    emitter_surface_area = np.sum([surface_area(t) for t in triangles if t['is_light']])
    rays = np.zeros(num_rays, dtype=Ray)
    choices = np.random.randint(0, len(emitters), num_rays)
    rand_us = np.random.rand(num_rays)
    rand_vs = np.random.rand(num_rays)
    rand_ws = 1 - rand_us - rand_vs
    rays['direction'] = unit(np.array([0, -1, 0, 0]))
    points = emitters[choices][:, 0] * rand_us[:, None] + emitters[choices][:, 1] * rand_vs[:, None] + emitters[choices][:, 2] * rand_ws[:, None]
    rays['origin'] = points + 0.0001 * rays['direction']
    rays['normal'] = rays['direction']
    rays['inv_direction'] = 1 / rays['direction']
    rays['c_importance'] = 1.0  # not accessed
    rays['l_importance'] = 1.0
    rays['tot_importance'] = 1.0
    rays['from_camera'] = 0
    rays['color'] = np.array([1, 1, 1, 1])
    rays['hit_light'] = -1
    return rays


def basic_tone_map(image):
    print(f"min {np.min(image)} max {np.max(image)}")
    if np.min(image) != np.max(image):
        image = image - np.min(image)
        image = image / np.max(image)
    return (255 * image).astype(np.uint8)


def tone_map(image):
    print(f"min {np.min(image)} max {np.max(image)}")
    tone_vector = np.array([0.0722, 0.7152, 0.2126])
    # tone_vector = ONES
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255 * result / (result + 1)).astype(np.uint8)


def unit(v):
    return v / np.linalg.norm(v)


def surface_area(t):
    e1 = (t['v1'] - t['v0'])[0:3]
    e2 = (t['v2'] - t['v0'])[0:3]
    return np.linalg.norm(np.cross(e1, e2)) / 2


if __name__ == '__main__':
    tris = load_obj('../resources/teapot.obj')

    # manually define a box around the teapot

    tris += triangles_for_box(np.array([-10, -2, -10]), np.array([10, 10, 10]))

    bvh = construct_BVH(tris)
    c = Camera(
        center=np.array([4, 2, 4]),
        direction=unit(np.array([-1, 0, -1])),
    )
    mats = get_materials()
    boxes, triangles = np_flatten_bvh(bvh)
    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()

    summed_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
    samples = 15
    to_display = np.zeros(summed_image.shape, dtype=np.uint8)

    batch_size = c.pixel_width * c.pixel_height
    out_camera_image = dev.buffer(batch_size * 16)
    out_camera_paths = dev.buffer(batch_size * Path.itemsize)
    out_camera_debug = dev.buffer(batch_size * 4)
    out_camera_debug_float = dev.buffer(batch_size * 4)

    out_light_image = dev.buffer(batch_size * 16)
    out_light_paths = dev.buffer(batch_size * Path.itemsize)
    out_light_debug = dev.buffer(batch_size * 4)
    out_light_debug_float = dev.buffer(batch_size * 4)

    final_out_samples = dev.buffer(batch_size * 16)
    final_out_debug = dev.buffer(batch_size * 4)
    final_out_float_debug = dev.buffer(batch_size * 4)

    boxes = boxes.flatten()
    triangles = triangles.flatten()

    trace_fn = dev.kernel(kernel).function("generate_paths")
    join_fn = dev.kernel(kernel).function("connect_paths")

    for i in range(samples):
        camera_rays = c.ray_batch_numpy().flatten()
        rands = np.random.rand(camera_rays.size * 32).astype(np.float32)

        start_time = time.time()
        trace_fn(batch_size, camera_rays, boxes, triangles, mats, rands, out_camera_image, out_camera_paths, out_camera_debug, out_camera_debug_float)
        print(f"Sample {i} camera trace time: {time.time() - start_time}")

        light_rays = fast_generate_light_rays(triangles, camera_rays.size)
        rands = np.random.rand(light_rays.size * 32).astype(np.float32)

        start_time = time.time()
        trace_fn(batch_size, light_rays, boxes, triangles, mats, rands, out_light_image, out_light_paths, out_light_debug, out_light_debug_float)
        print(f"Sample {i} light trace time: {time.time() - start_time}")

        start_time = time.time()
        join_fn(batch_size, out_camera_paths, out_light_paths, triangles, mats, boxes, final_out_samples, final_out_debug, final_out_float_debug)
        print(f"Sample {i} join time: {time.time() - start_time}")

        unidirectional_image = np.frombuffer(out_camera_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:,:, :3]
        retrieved_camera_debug = np.frombuffer(out_camera_debug, dtype=np.int32)
        retrieved_camera_floats = np.frombuffer(out_camera_debug_float, dtype=np.float32)
        retrieved_light_debug = np.frombuffer(out_light_debug, dtype=np.int32)
        retrieved_light_floats = np.frombuffer(out_light_debug_float, dtype=np.float32)

        bidirectional_image = np.frombuffer(final_out_samples, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]
        retrieved_values = np.frombuffer(final_out_debug, dtype=np.int32)
        retrieved_floats = np.frombuffer(final_out_float_debug, dtype=np.float32)

        print("camera", retrieved_camera_debug, np.min(retrieved_camera_debug), np.max(retrieved_camera_debug))
        print("camera floats", retrieved_camera_floats, np.min(retrieved_camera_floats), np.max(retrieved_camera_floats))
        print("light", retrieved_light_debug, np.min(retrieved_light_debug), np.max(retrieved_light_debug))
        print("light floats", retrieved_light_floats, np.min(retrieved_light_floats), np.max(retrieved_light_floats))
        print("join", retrieved_values, np.min(retrieved_values), np.max(retrieved_values))
        print("join floats", retrieved_floats, np.min(retrieved_floats), np.max(retrieved_floats))

        camera_paths = np.frombuffer(out_camera_paths, dtype=Path)
        light_paths = np.frombuffer(out_light_paths, dtype=Path)

        print("camera paths", np.max(camera_paths['length']), np.min(camera_paths['length']))
        print("light paths", np.max(light_paths['length']), np.min(light_paths['length']))

        summed_image += np.nan_to_num(bidirectional_image)
        if np.any(np.isnan(summed_image)):
            print("NaNs in summed image!!!")
            break

        to_display = tone_map(summed_image / (i + 1))

        if np.any(np.isnan(to_display)):
            print("NaNs in to_display!!!")
            break

        # open a window to display the image
        cv2.imshow('image', to_display)
        cv2.waitKey(1)
    cv2.imwrite(f'../output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', to_display)
