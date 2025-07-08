import numpy as np
import objloader
from constants import (
    UNIT_X,
    UNIT_Y,
    UNIT_Z,
    RED,
    BLUE,
    GREEN,
    CYAN,
    WHITE,
    FULL_WHITE,
    INVALID,
    INF,
    NEG_INF,
    DEFAULT_BOX_MIN_CORNER,
    DEFAULT_BOX_MAX_CORNER,
    DEFAULT_LIGHT_HEIGHT,
    DEFAULT_LIGHT_SCALE,
)
from collections import defaultdict
from plyfile import PlyData
from struct_types import Ray, Material
import time
from bvh import FastTreeBox


def unit(v):
    return v / np.linalg.norm(v)


class Triangle:
    def __init__(
        self,
        v0,
        v1,
        v2,
        _min=None,
        _max=None,
        normal=None,
        surface_area=None,
        material=0,
        emitter=False,
        camera=False,
    ):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.n0 = np.zeros(3, dtype=np.float64)
        self.n1 = np.zeros(3, dtype=np.float64)
        self.n2 = np.zeros(3, dtype=np.float64)
        self.t0 = np.zeros(3, dtype=np.float64)
        self.t1 = np.zeros(3, dtype=np.float64)
        self.t2 = np.zeros(3, dtype=np.float64)
        self.material = material
        self.emitter = emitter
        self.camera = camera

        if _min is None:
            _min = np.minimum(v0, np.minimum(v1, v2))
        self.min = _min

        if _max is None:
            _max = np.maximum(v0, np.maximum(v1, v2))
        self.max = _max

        if normal is None:
            normal = unit(np.cross(v1 - v0, v2 - v0))
        self.n = normal

        if surface_area is None:
            self.surface_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
        self.surface_area = surface_area


def fast_load_obj(obj_path, offset=None, material=None, emitter=False, scale=1.0):
    if offset is None:
        offset = np.zeros(3)

    obj = objloader.Obj.open(obj_path)
    vertices = np.array(obj.vert) * scale + offset
    faces = np.array(obj.face)[:, 0].astype(np.int32).reshape(-1, 3) - 1
    return fast_load(vertices, faces, material=material, emitter=emitter)


def fast_load_ply(ply_path, offset=None, material=None, scale=1.0, emitter=False):
    if offset is None:
        offset = np.zeros(3)

    ply = PlyData.read(ply_path)
    vertices = (
        np.array(ply["vertex"].data).view(np.float32).reshape(-1, 3) * scale + offset
    )
    faces = np.stack(ply["face"]["vertex_indices"])
    return fast_load(vertices, faces, material=material, emitter=emitter)


def fast_load(vertices, faces, emitter=False, material=None):
    triangles = vertices[faces]

    mins = np.min(triangles, axis=1)
    maxes = np.max(triangles, axis=1)
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    surface_areas = np.linalg.norm(face_normals, axis=1) / 2
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]

    if material is None:
        materials = np.zeros(len(triangles), dtype=np.int32)
    else:
        materials = np.full(len(triangles), material, dtype=np.int32)

    if emitter:
        emitters = np.ones(len(triangles), dtype=np.bool_)
    else:
        emitters = np.zeros(len(triangles), dtype=np.bool_)

    return FastTreeBox(
        faces=faces,
        triangles=triangles,
        mins=mins,
        maxes=maxes,
        face_normals=face_normals,
        surface_areas=surface_areas,
        material=materials,
        emitter=emitters,
        camera=np.zeros(len(triangles), dtype=np.int32),
    )

def get_materials():
    materials = np.zeros(8, dtype=Material)
    materials["color"] = np.zeros((8, 4), dtype=np.float32)
    materials["color"][0][:3] = RED
    materials["color"][1][:3] = GREEN
    materials["color"][2][:3] = BLUE
    materials["color"][3][:3] = WHITE
    materials["color"][4][:3] = WHITE
    materials["color"][5][:3] = BLUE
    materials["color"][6][:3] = FULL_WHITE
    materials["color"][7][:3] = FULL_WHITE
    materials["emission"] = np.zeros((8, 4), dtype=np.float32)
    materials["emission"][6] = np.ones(4, dtype=np.float32)
    materials["type"] = 0

    materials["ior"] = 1.5
    materials["alpha"] = 0.0

    materials[0]["type"] = 1
    materials[5]["type"] = 1

    return materials


def triangles_for_box(
    box_min=DEFAULT_BOX_MIN_CORNER,
    box_max=DEFAULT_BOX_MAX_CORNER,
    light_height=DEFAULT_LIGHT_HEIGHT,
    light_scale=DEFAULT_LIGHT_SCALE,
):
    span = box_max - box_min
    left_bottom_back = box_min
    right_bottom_back = box_min + span * UNIT_X
    left_top_back = box_min + span * UNIT_Y
    left_bottom_front = box_min + span * UNIT_Z

    right_top_front = box_max
    left_top_front = box_max - span * UNIT_X
    right_bottom_front = box_max - span * UNIT_Y
    right_top_back = box_max - span * UNIT_Z

    # shrink = np.array([.1, .95, .1], dtype=np.float32)
    shrink = np.array([light_scale, light_height, light_scale], dtype=np.float32)
    # shrink = np.array([.95, .95, .95], dtype=np.float32)
    tris = [
        # back wall
        Triangle(left_bottom_back, right_bottom_back, right_top_back, material=4),
        Triangle(left_bottom_back, right_top_back, left_top_back, material=4),
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
        Triangle(
            left_top_back * shrink,
            right_top_back * shrink,
            right_top_front * shrink,
            material=6,
            emitter=True,
        ),
        Triangle(
            left_top_back * shrink,
            right_top_front * shrink,
            left_top_front * shrink,
            material=6,
            emitter=True,
        ),
    ]
    return tris


def camera_geometry(camera):
    bottom_corner = camera.origin + camera.dx * camera.phys_width
    top_corner = (
        camera.origin + camera.dx * camera.phys_width + camera.dy * camera.phys_height
    )
    other_top_corner = camera.origin + camera.dy * camera.phys_height
    tris = [
        Triangle(camera.origin, bottom_corner, top_corner, material=7, camera=True),
        Triangle(camera.origin, top_corner, other_top_corner, material=7, camera=True),
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
    emitter_indices = np.array([i for i, t in enumerate(triangles) if t["is_light"]])
    emitters = np.array(
        [[t["v0"], t["v1"], t["v2"]] for t in triangles if t["is_light"]]
    )
    emitter_surface_area = np.sum([surface_area(t) for t in triangles if t["is_light"]])
    rays = np.zeros(num_rays, dtype=Ray)
    choices = np.random.randint(0, len(emitters), num_rays)
    rand_us, rand_vs, rand_ws = random_uvs(num_rays)
    rays["direction"] = unit(np.array([0, -1, 0, 0]))
    points = (
        emitters[choices][:, 0] * rand_us[:, None]
        + emitters[choices][:, 1] * rand_vs[:, None]
        + emitters[choices][:, 2] * rand_ws[:, None]
    )
    rays["origin"] = points + 0.0001 * rays["direction"]
    rays["normal"] = rays["direction"]
    rays["inv_direction"] = 1.0 / rays["direction"]
    rays["c_importance"] = 1.0  # set in kernel
    rays["l_importance"] = 1.0 / emitter_surface_area
    rays["tot_importance"] = 1.0 / emitter_surface_area
    rays["from_camera"] = 0
    rays["color"] = np.array([1.0, 1.0, 1.0, 1.0])
    rays["hit_light"] = -1
    rays["hit_camera"] = -1
    rays["triangle"] = emitter_indices[choices]
    rays["material"] = 6
    return rays


def unit(v):
    return v / np.linalg.norm(v)


def surface_area(t):
    e1 = (t["v1"] - t["v0"])[0:3]
    e2 = (t["v2"] - t["v0"])[0:3]
    return np.linalg.norm(np.cross(e1, e2)) / 2


def smooth_normals(triangles):
    vertex_triangles = defaultdict(list)

    for i, t in enumerate(triangles):
        vertex_triangles[t.v0.tobytes()].append((i, 0))
        vertex_triangles[t.v1.tobytes()].append((i, 1))
        vertex_triangles[t.v2.tobytes()].append((i, 2))

    for _, l in vertex_triangles.items():

        avg_normal = np.zeros(3)
        for j, v in l:
            avg_normal += triangles[j].n
        normal_mag = np.linalg.norm(avg_normal)
        if normal_mag == 0:
            for j, v in l:
                if v == 0:
                    triangles[j].n0 = triangles[j].n
                elif v == 1:
                    triangles[j].n1 = triangles[j].n
                else:
                    triangles[j].n2 = triangles[j].n
        else:
            for j, v in l:
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
