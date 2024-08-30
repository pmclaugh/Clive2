import numpy as np
from camera import Camera
import objloader
from constants import INVALID, UNIT_X, UNIT_Y, UNIT_Z, RED, BLUE, GREEN, CYAN, WHITE
from bvh import construct_BVH, np_flatten_bvh
import cv2
import metalcompute as mc
import time
from struct_types import Ray, Material, Path
from datetime import datetime
from collections import defaultdict
import argparse
import os
from plyfile import PlyData
from functools import cached_property
from scipy.stats.qmc import Sobol


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
    camera = False

    def __init__(self, v0, v1, v2, material=0, emitter=False, camera=False):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material
        self.emitter = emitter
        self.camera = camera

    @cached_property
    def min(self):
        return np.minimum(self.v0, np.minimum(self.v1, self.v2))

    @cached_property
    def max(self):
        return np.maximum(self.v0, np.maximum(self.v1, self.v2))

    @cached_property
    def n(self):
        a = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        return a / np.linalg.norm(a)

    @cached_property
    def surface_area(self):
        e1 = (self.v1 - self.v0)[0:3]
        e2 = (self.v2 - self.v0)[0:3]
        return np.linalg.norm(np.cross(e1, e2)) / 2


def load_obj(obj_path, offset=None, material=None):
    if offset is None:
        offset = np.zeros(3)
    obj = objloader.Obj.open(obj_path)
    triangles = []
    for i, ((v0, n0, t0), (v1, n1, t1), (v2, n2, t2)) in enumerate(zip(*[iter(obj.face)] * 3)):
        triangle = Triangle(np.array(obj.vert[v0 - 1]) + offset,
                            np.array(obj.vert[v1 - 1]) + offset,
                            np.array(obj.vert[v2 - 1]) + offset)

        # normals
        triangle.n0 = np.array(obj.norm[n0 - 1]) if n0 is not None else INVALID
        triangle.n1 = np.array(obj.norm[n1 - 1]) if n1 is not None else INVALID
        triangle.n2 = np.array(obj.norm[n2 - 1]) if n2 is not None else INVALID

        # texture UVs
        triangle.t0 = np.array(obj.text[t0 - 1]) if t0 is not None else INVALID
        triangle.t1 = np.array(obj.text[t1 - 1]) if t1 is not None else INVALID
        triangle.t2 = np.array(obj.text[t2 - 1]) if t2 is not None else INVALID

        # material
        if material is None:
            triangle.material = 0
        else:
            triangle.material = material
        triangle.emitter = False

        triangles.append(triangle)
    return triangles


def load_ply(ply_path, offset=None, material=None, scale=1.0):
    if offset is None:
        offset = np.zeros(3)
    base_load_time = time.time()
    ply = PlyData.read(ply_path)
    print(f"PlyData.read in {time.time() - base_load_time}")
    triangles = []
    dropped_triangles = 0
    array_time = time.time()
    vertices = np.array(list(list(vertex) for vertex in ply['vertex'])) * scale + offset
    print(f"vertices array in {time.time() - array_time}")
    face_time = time.time()
    for face in ply['face']['vertex_indices']:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        triangle = Triangle(v0, v1, v2)

        # normals
        triangle.n0 = INVALID
        triangle.n1 = INVALID
        triangle.n2 = INVALID

        # texture UVs
        triangle.t0 = INVALID
        triangle.t1 = INVALID
        triangle.t2 = INVALID

        # material
        if material is None:
            triangle.material = 0
        else:
            triangle.material = material
        triangle.emitter = False

        if np.any(np.isnan(triangle.n)):
            dropped_triangles += 1
        else:
            triangles.append(triangle)
    print(f"faces in {time.time() - face_time}")
    print(f'done loading ply. loaded {len(triangles)} triangles, dropped {dropped_triangles}')
    return triangles


def get_materials():
    materials = np.zeros(7, dtype=Material)
    materials['color'] = np.zeros((7, 4), dtype=np.float32)
    materials['color'][0][:3] = RED
    materials['color'][1][:3] = GREEN
    materials['color'][2][:3] = WHITE
    materials['color'][3][:3] = CYAN
    materials['color'][4][:3] = WHITE
    materials['color'][5][:3] = BLUE
    materials['color'][6][:3] = WHITE
    materials['emission'] = np.zeros((7, 4), dtype=np.float32)
    materials['emission'][6] = np.array([1, 1, 1, 1])
    materials['type'] = 0

    materials['ior'] = 1.5
    materials['alpha'] = 0.01

    materials[0]['type'] = 1
    materials[5]['type'] = 1
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

    shrink = np.array([.25, .95, .25], dtype=np.float32)
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
        Triangle(left_top_back, right_top_back, right_top_front, material=4),
        Triangle(left_top_back, right_top_front, left_top_front, material=4),
        # ceiling light # NB this assumes box is centered on the origin, at least wrt x and z
        Triangle(left_top_back * shrink, right_top_back * shrink, right_top_front * shrink, material=6, emitter=True),
        Triangle(left_top_back * shrink, right_top_front * shrink, left_top_front * shrink, material=6, emitter=True),

        # wall light
        # Triangle(left_bottom_front * shrink, right_top_front * shrink, right_bottom_front * shrink, material=6, emitter=True),
        # Triangle(left_bottom_front * shrink, left_top_front * shrink, right_top_front * shrink, material=6, emitter=True),
    ]
    return tris


def camera_geometry(camera):
    bottom_corner = camera.origin + camera.dx * camera.phys_width
    top_corner = camera.origin + camera.dx * camera.phys_width + camera.dy * camera.phys_height
    other_top_corner = camera.origin + camera.dy * camera.phys_height
    tris = [
        Triangle(camera.origin, bottom_corner, top_corner, material=2, camera=True),
        Triangle(camera.origin, top_corner, other_top_corner, material=2, camera=True),
    ]
    return tris


def random_uvs(num):
    u = np.random.rand(num)
    v = np.random.rand(num)
    need_flipped = (u + v) > 1
    u[need_flipped] = 1 - u[need_flipped]
    v[need_flipped] = 1 - v[need_flipped]
    w = 1 - u - v
    return u, v, w


def fast_generate_light_rays(triangles, num_rays):
    emitter_indices = np.array([i for i, t in enumerate(triangles) if t['is_light']])
    emitters = np.array([[t['v0'], t['v1'], t['v2']] for t in triangles if t['is_light']])
    emitter_surface_area = np.sum([surface_area(t) for t in triangles if t['is_light']])
    rays = np.zeros(num_rays, dtype=Ray)
    choices = np.random.randint(0, len(emitters), num_rays)
    rand_us, rand_vs, rand_ws = random_uvs(num_rays)
    rays['direction'] = unit(np.array([0, -1, 0, 0]))
    points = emitters[choices][:, 0] * rand_us[:, None] + emitters[choices][:, 1] * rand_vs[:, None] + emitters[choices][:, 2] * rand_ws[:, None]
    rays['origin'] = points + 0.0001 * rays['direction']
    rays['normal'] = rays['direction']
    rays['inv_direction'] = 1.0 / rays['direction']
    rays['c_importance'] = 1.0  # set in kernel
    rays['l_importance'] = 1.0 / emitter_surface_area
    rays['tot_importance'] = 1.0 / emitter_surface_area
    rays['from_camera'] = 0
    rays['color'] = np.array([1.0, 1.0, 1.0, 1.0])
    rays['hit_light'] = -1
    rays['hit_camera'] = -1
    rays['triangle'] = emitter_indices[choices]
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
    tone_sums = np.sum(image * tone_vector, axis=2)
    log_tone_sums = np.log(0.1 + tone_sums)
    per_pixel_lts = np.sum(log_tone_sums) / np.prod(image.shape[:2])
    Lw = np.exp(per_pixel_lts)
    result = image * 2. / Lw
    return (255.0 * result / (result + 1)).astype(np.uint8)


def unit(v):
    return v / np.linalg.norm(v)


def surface_area(t):
    e1 = (t['v1'] - t['v0'])[0:3]
    e2 = (t['v2'] - t['v0'])[0:3]
    return np.linalg.norm(np.cross(e1, e2)) / 2


def smooth_normals(triangles):
    vertex_triangles = defaultdict(list)

    for i, t in enumerate(triangles):
        vertex_triangles[t.v0.tobytes()].append((i, 0))
        vertex_triangles[t.v1.tobytes()].append((i, 1))
        vertex_triangles[t.v2.tobytes()].append((i, 2))

    for _, l in vertex_triangles.items():
        
        avg_normal = np.zeros(3)
        for (j, v) in l:
            avg_normal += triangles[j].n
        normal_mag = np.linalg.norm(avg_normal)
        if normal_mag == 0:
            for (j, v) in l:
                if v == 0:
                    triangles[j].n0 = triangles[j].n
                elif v == 1:
                    triangles[j].n1 = triangles[j].n
                else:
                    triangles[j].n2 = triangles[j].n
        else:
            for (j, v) in l:
                if v == 0:
                    triangles[j].n0 = avg_normal / normal_mag
                elif v == 1:
                    triangles[j].n1 = avg_normal / normal_mag
                else:
                    triangles[j].n2 = avg_normal / normal_mag


def dummy_smooth_normals(triangles):
    for t in triangles:
        t.n0 = t.n
        t.n1 = t.n
        t.n2 = t.n


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--frame-number', type=int, default=0)
    parser.add_argument('--total-frames', type=int, default=1)
    parser.add_argument('--movie-name', type=str, default='default')
    args = parser.parse_args()

    os.makedirs(f'../output/{args.movie_name}', exist_ok=True)

    tris = []
    # load the teapots
    tris += load_obj('../resources/teapot.obj', offset=np.array([0, 0, 2.5]), material=0)
    tris += load_obj('../resources/teapot.obj', offset=np.array([0, 0, -2.5]), material=5)

    # load the dragon
    # load_time = time.time()
    # tris += load_ply('../resources/dragon_vrip_res2.ply', offset=np.array([0, -4, 0]), material=5, scale=50)
    # print(f"done loading dragon in {time.time() - load_time}")

    smooth_time = time.time()
    smooth_normals(tris)
    print("done smoothing normals in", time.time() - smooth_time)

    # manually define a box around the teapots, don't smooth it
    box_tris = triangles_for_box(np.array([-10, -2, -10]), np.array([10, 10, 10]))
    dummy_smooth_normals(box_tris)
    tris += box_tris

    # camera setup
    c = Camera(
        center=np.array([4, 1.5, 5]),
        direction=unit(np.array([-1, -0.2, -1])),
        pixel_width=args.width,
        pixel_height=args.height,
        phys_width=args.width / args.height,
        phys_height=1,
    )
    camera_arr = c.to_struct()
    camera_tris = camera_geometry(c)
    dummy_smooth_normals(camera_tris)
    tris += camera_tris

    # build and marshall BVH
    start_time = time.time()
    bvh = construct_BVH(tris)
    print("done building bvh", time.time() - start_time)
    boxes, triangles = np_flatten_bvh(bvh)
    print("done flattening bvh")
    boxes = boxes
    triangles = triangles

    # load materials (very basic for now)
    mats = get_materials()

    samples = args.samples

    # Metal stuff. get device, load and compile kernels
    dev = mc.Device()
    with open("trace.metal", "r") as f:
        kernel = f.read()
    trace_fn = dev.kernel(kernel).function("generate_paths")
    join_fn = dev.kernel(kernel).function("connect_paths")

    # make a bunch of buffers
    summed_image = np.zeros((c.pixel_height, c.pixel_width, 3), dtype=np.float32)
    to_display = np.zeros(summed_image.shape, dtype=np.uint8)
    batch_size = c.pixel_width * c.pixel_height

    # render loop
    for i in range(samples):
        out_camera_image = dev.buffer(batch_size * 16)
        out_camera_paths = dev.buffer(batch_size * Path.itemsize)
        out_camera_debug_image = dev.buffer(batch_size * 16)

        out_light_image = dev.buffer(batch_size * 16)
        out_light_paths = dev.buffer(batch_size * Path.itemsize)
        out_light_debug_image = dev.buffer(batch_size * 16)

        final_out_samples = dev.buffer(batch_size * 16)
        final_out_secondary_samples = dev.buffer(batch_size * 16)

        trace_fn = dev.kernel(kernel).function("generate_paths")
        join_fn = dev.kernel(kernel).function("connect_paths")

        # make camera rays and rands
        camera_rays, camera_ray_map = c.ray_batch_numpy(adaptive=i > 2)
        rands = np.random.rand(camera_rays.size * 32).astype(np.float32)

        # trace camera paths
        start_time = time.time()
        trace_fn(batch_size, camera_rays, boxes, triangles, mats, rands, out_camera_image, out_camera_paths, out_camera_debug_image)
        print(f"Sample {i} camera trace time: {time.time() - start_time}")

        # retrieve camera trace outputs
        unidirectional_image = np.frombuffer(out_camera_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]
        camera_paths = np.frombuffer(out_camera_paths, dtype=Path)
        retrieved_camera_debug_image = np.frombuffer(out_camera_debug_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

        # make light rays and rands
        light_rays = fast_generate_light_rays(triangles, camera_rays.size)
        rands = np.random.rand(light_rays.size * 32).astype(np.float32)

        # trace light paths
        start_time = time.time()
        trace_fn(batch_size, light_rays, boxes, triangles, mats, rands, out_light_image, out_light_paths, out_light_debug_image)
        print(f"Sample {i} light trace time: {time.time() - start_time}")

        # retrieve light trace outputs
        light_paths = np.frombuffer(out_light_paths, dtype=Path)
        retrieved_light_debug_image = np.frombuffer(out_light_debug_image, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

        # join paths
        start_time = time.time()
        join_fn(batch_size, out_camera_paths, out_light_paths, triangles, mats, boxes, camera_arr[0], final_out_samples, final_out_secondary_samples)
        print(f"Sample {i} join time: {time.time() - start_time}")

        # retrieve joined path outputs
        bidirectional_image = np.frombuffer(final_out_samples, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]
        bidirectional_secondary_image = np.frombuffer(final_out_secondary_samples, dtype=np.float32).reshape(c.pixel_height, c.pixel_width, 4)[:, :, :3]

        print(np.sum(np.isnan(bidirectional_image)), "nans in image")
        print(np.sum(np.any(np.isnan(bidirectional_image), axis=2)), "pixels with nans")
        print(np.sum(np.isinf(bidirectional_image)), "infs in image")
        bidirectional_image = np.nan_to_num(bidirectional_image, posinf=0.0, neginf=0.0)

        print(np.sum(np.isnan(bidirectional_secondary_image)), "nans in secondary image")
        print(np.sum(np.any(np.isnan(bidirectional_secondary_image), axis=2)), "pixels with nans in secondary image")
        print(np.sum(np.isinf(bidirectional_secondary_image)), "infs in secondary image")
        bidirectional_secondary_image = np.nan_to_num(bidirectional_secondary_image, posinf=0.0, neginf=0.0)

        single_sample_image = c.process_samples(bidirectional_image, camera_ray_map)
        # c.process_samples(bidirectional_secondary_image, camera_ray_map, increment=False)

        # post processing. tone map, sum, division
        image = c.get_image()
        # image = tone_map(bidirectional_image)

        # display the image
        cv2.imshow('image', image)
        cv2.waitKey(1)

    # save the image
    if args.total_frames == 1:
        cv2.imwrite(f'../output/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', image)
    else:
        cv2.imwrite(f'../output/{args.movie_name}/frame_{args.frame_number:04d}.png', image)
