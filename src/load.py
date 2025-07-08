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
from functools import cached_property
from struct_types import Ray, Material
import time


def unit(v):
    return v / np.linalg.norm(v)


class Triangle:
    def __init__(self, v0, v1, v2, material=0, emitter=False, camera=False):
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

        self.min = np.minimum(self.v0, np.minimum(self.v1, self.v2))
        self.max = np.maximum(self.v0, np.maximum(self.v1, self.v2))
        self.n = unit(np.cross(self.v1 - self.v0, self.v2 - self.v0))
        e1 = (self.v1 - self.v0)[0:3]
        e2 = (self.v2 - self.v0)[0:3]
        self.surface_area = np.linalg.norm(np.cross(e1, e2)) / 2



def load_obj(obj_path, offset=None, material=None, scale=1.0):
    if offset is None:
        offset = np.zeros(3)
    obj = objloader.Obj.open(obj_path)
    triangles = []
    for i, ((v0, n0, t0), (v1, n1, t1), (v2, n2, t2)) in enumerate(
        zip(*[iter(obj.face)] * 3)
    ):
        triangle = Triangle(
            np.array(obj.vert[v0 - 1]) * scale + offset,
            np.array(obj.vert[v1 - 1]) * scale + offset,
            np.array(obj.vert[v2 - 1]) * scale + offset,
        )

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


def fast_load_ply(ply_path, offset=None, material=None, scale=1.0, emitter=False):
    if offset is None:
        offset = np.zeros(3)
    base_load_time = time.time()
    ply = PlyData.read(ply_path)
    print(f"PlyData.read in {time.time() - base_load_time}")
    array_time = time.time()
    vertices = np.array(ply['vertex'].data).view(np.float32).reshape(-1, 3) * scale + offset
    print(f"vertices array in {time.time() - array_time}")

    face_time = time.time()
    faces = np.stack(ply['face']['vertex_indices'])
    triangles = vertices[faces]
    print(f"faces in {time.time() - face_time}")

    derived_time = time.time()
    mins = np.min(triangles, axis=1)
    maxs = np.max(triangles, axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    surface_areas = np.linalg.norm(normals, axis=1) / 2
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    print(f"derived values in {time.time() - derived_time}")

    return {
        "vertices": vertices,
        "triangles": triangles,
        "mins": mins,
        "maxs": maxs,
        "normals": normals,
        "surface_areas": surface_areas,
        "material": material if material is not None else 0,
        "emitter": emitter,
    }

def load_ply(ply_path, offset=None, material=None, scale=1.0, emitter=False):
    if offset is None:
        offset = np.zeros(3)
    base_load_time = time.time()
    ply = PlyData.read(ply_path)
    print(f"PlyData.read in {time.time() - base_load_time}")
    triangles = []
    dropped_triangles = 0
    array_time = time.time()
    vertices = np.array(list(list(vertex) for vertex in ply["vertex"])) * scale + offset

    # np.array(ply['vertex'].data).view(np.float32).reshape(-1, 3)
    print(f"vertices array in {time.time() - array_time}")
    face_time = time.time()
    for face in ply["face"]["vertex_indices"]:
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
        triangle.emitter = emitter

        if np.any(np.isnan(triangle.n)):
            dropped_triangles += 1
        else:
            triangles.append(triangle)
    # np.stack(ply['face']['vertex_indices'])
    print(f"faces in {time.time() - face_time}")
    print(
        f"done loading ply. loaded {len(triangles)} triangles, dropped {dropped_triangles}"
    )
    return triangles


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
