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

    shrink = np.array([.5, .95, .5], dtype=np.float32)
    tris = [
        # back wall
        Triangle(left_bottom_back, right_bottom_back, right_top_back, material=0),
        Triangle(left_bottom_back, right_top_back, left_top_back, material=0),
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


def tone_map(image):
    tone_vector = np.array([0.0722, 0.7152, 0.2126])
    # tone_vector = ONES
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255 * result / (result + 1)).astype(np.uint8)


if __name__ == '__main__':
    tris = load_obj('../resources/teapot.obj')

    # manually define a box around the teapot

    tris += triangles_for_box(np.array([-10, -2, -10]), np.array([10, 10, 10]))

    bvh = construct_BVH(tris)
    c = Camera(
        center=np.array([0, 5, -5]),
        direction=np.array([0, -1, 1]),
    )
    mats = get_materials()
    boxes, triangles = np_flatten_bvh(bvh)
    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()
    kernel_fn = dev.kernel(kernel).function("generate_paths")
    summed_image = np.zeros((c.pixel_height, c.pixel_width, 4), dtype=np.float32)
    samples = 25
    for i in range(samples):
        rays = c.ray_batch_numpy()
        rays = rays.flatten()
        boxes = boxes.flatten()
        triangles = triangles.flatten()
        rands = np.random.rand(np.size(rays) * 2).astype(np.float32)
        out_image = dev.buffer(np.size(rays) * 16)
        out_paths = dev.buffer(rays.size * Path.itemsize)
        out_debug = dev.buffer(16)

        start_time = time.time()
        kernel_fn(rays.size, rays, boxes, triangles, mats, rands, out_image, out_paths, out_debug)
        print(f"Sample {i} render time: {time.time() - start_time}")

        retrieved_image = np.frombuffer(out_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)
        retrieved_rays = np.frombuffer(out_paths, dtype=Path)
        retrieved_values = np.frombuffer(out_debug, dtype=np.int32)

        summed_image += retrieved_image

        # open a window to display the image
        cv2.imshow('image', tone_map(summed_image[:, :, :3] / (i + 1)))
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.imwrite(f'../output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', tone_map(summed_image[:, :, :3] / samples))
