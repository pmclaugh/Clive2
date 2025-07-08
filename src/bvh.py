import numpy as np
from numba import njit
from constants import *
from struct_types import Box, Triangle


class FastTreeBox:
    def __init__(
        self, faces, triangles, mins, maxes, face_normals, smoothed_normals, surface_areas, material, emitter, camera
    ):
        self.left = None
        self.right = None
        self.parent = None

        self.faces = faces
        self.triangles = triangles
        self.mins = mins
        self.maxes = maxes
        self.min = np.min(self.mins, axis=0) if len(mins) else INF
        self.max = np.max(self.maxes, axis=0) if len(maxes) else NEG_INF
        self.face_normals = face_normals
        self.smoothed_normals = smoothed_normals
        self.surface_areas = surface_areas
        self.material = material
        self.emitter = emitter
        self.camera = camera

    @classmethod
    def empty_box(cls):
        return cls(
            faces=np.empty((0, 3), dtype=np.uint32),
            triangles=np.empty((0, 3, 3), dtype=np.float32),
            mins=np.empty((0, 3), dtype=np.float32),
            maxes=np.empty((0, 3), dtype=np.float32),
            face_normals=np.empty((0, 3), dtype=np.float32),
            smoothed_normals=np.empty((0, 3, 3), dtype=np.float32),
            surface_areas=np.empty((0,), dtype=np.float32),
            material=np.empty((0,), dtype=np.int32),
            emitter=np.empty((0,), dtype=np.bool_),
            camera=np.empty((0,), dtype=np.bool_),
        )

    @classmethod
    def from_triangle_objects(cls, triangle_objects):
        triangles = np.array(
            [[t.v0, t.v1, t.v2] for t in triangle_objects], dtype=np.float32
        )
        faces = np.zeros(
            (len(triangle_objects), 3), dtype=np.uint32
        )
        mins = np.min(triangles, axis=1)
        maxes = np.max(triangles, axis=1)
        normals = np.array([t.n for t in triangle_objects], dtype=np.float32)
        smoothed_normals = np.array([[t.n] * 3 for t in triangle_objects], dtype=np.float32)
        surface_areas = np.array(
            [t.surface_area for t in triangle_objects], dtype=np.float32
        )
        material = np.array([t.material for t in triangle_objects], dtype=np.int32)
        emitter = np.array([t.emitter for t in triangle_objects], dtype=np.bool_)
        camera = np.array([t.camera for t in triangle_objects], dtype=np.bool_)

        return cls(
            faces, triangles, mins, maxes, normals, smoothed_normals, surface_areas, material, emitter, camera
        )

    def __add__(self, other):
        if not isinstance(other, FastTreeBox):
            raise TypeError("Can only add another FastTreeBox")

        new_faces = np.concatenate((self.faces, other.faces), axis=0)
        new_triangles = np.concatenate((self.triangles, other.triangles), axis=0)
        new_mins = np.concatenate((self.mins, other.mins))
        new_maxes = np.concatenate((self.maxes, other.maxes))
        new_normals = np.concatenate((self.face_normals, other.face_normals), axis=0)
        new_smoothed_normals = np.concatenate((self.smoothed_normals, other.smoothed_normals), axis=0)
        new_surface_areas = np.concatenate((self.surface_areas, other.surface_areas))
        new_material = np.concatenate((self.material, other.material))
        new_emitter = np.concatenate((self.emitter, other.emitter))
        new_camera = np.concatenate((self.camera, other.camera))

        return FastTreeBox(
            new_faces,
            new_triangles,
            new_mins,
            new_maxes,
            new_normals,
            new_smoothed_normals,
            new_surface_areas,
            new_material,
            new_emitter,
            new_camera,
        )


@njit
def surface_area(mins, maxes):
    span = maxes - mins
    return 2 * (span[0] * span[1] + span[1] * span[2] + span[2] * span[0])


@njit
def surface_areas(mins, maxes):
    spans = maxes - mins
    return 2 * (
        spans[:, 0] * spans[:, 1]
        + spans[:, 1] * spans[:, 2]
        + spans[:, 2] * spans[:, 0]
    )


def object_split(box: FastTreeBox):
    best_sah = np.inf
    best_split = 0
    best_sort = None

    triangle_centers = (box.mins + box.maxes) / 2

    for axis in [0, 1, 2]:
        axis_sorted = np.argsort(triangle_centers[:, axis])

        ltr_maxes = np.maximum.accumulate(box.maxes[axis_sorted])
        ltr_mins = np.minimum.accumulate(box.mins[axis_sorted])
        rtl_maxes = np.maximum.accumulate(box.maxes[axis_sorted[::-1]])[::-1]
        rtl_mins = np.minimum.accumulate(box.mins[axis_sorted[::-1]])[::-1]

        left_surface_areas = surface_areas(ltr_mins, ltr_maxes)[:-1]
        right_surface_areas = surface_areas(rtl_mins, rtl_maxes)[1:]

        sah = left_surface_areas * np.arange(
            len(left_surface_areas)
        ) + right_surface_areas * (
            len(right_surface_areas) - np.arange(len(right_surface_areas))
        )
        min_sah_index = np.argmin(sah)
        if sah[min_sah_index] < best_sah:
            best_sah = sah[min_sah_index]
            best_split = min_sah_index
            best_sort = axis_sorted

    i = best_split + 1

    left_box = FastTreeBox(
        faces=box.faces[best_sort, :],
        triangles=box.triangles[best_sort[:i]],
        mins=box.mins[best_sort[:i]],
        maxes=box.maxes[best_sort[:i]],
        face_normals=box.face_normals[best_sort[:i]],
        smoothed_normals=box.smoothed_normals[best_sort[:i]],
        surface_areas=box.surface_areas[best_sort[:i]],
        material=box.material[best_sort[:i]],
        emitter=box.emitter[best_sort[:i]],
        camera=box.camera[best_sort[:i]],
    )

    right_box = FastTreeBox(
        faces=box.faces[best_sort, :],
        triangles=box.triangles[best_sort[i:]],
        mins=box.mins[best_sort[i:]],
        maxes=box.maxes[best_sort[i:]],
        face_normals=box.face_normals[best_sort[i:]],
        smoothed_normals=box.smoothed_normals[best_sort[i:]],
        surface_areas=box.surface_areas[best_sort[i:]],
        material=box.material[best_sort[i:]],
        emitter=box.emitter[best_sort[i:]],
        camera=box.camera[best_sort[i:]],
    )

    assert len(left_box.triangles) + len(right_box.triangles) == len(box.triangles)

    return best_sah, left_box, right_box


def construct_BVH(root_box):
    max_depth = 0
    stack = [root_box]
    while stack:
        box = stack.pop()
        max_depth = max(max_depth, len(stack))
        if (len(box.triangles) <= MAX_MEMBERS) or len(stack) > MAX_DEPTH:
            continue

        sah, l, r = object_split(box)

        if r is not None:
            box.right = r
            r.parent = box
            stack.append(r)
        if l is not None:
            box.left = l
            l.parent = box
            stack.append(l)

    print(f"max BVH depth: {max_depth}")
    return root_box


def count_boxes(root: FastTreeBox):
    box_stack = [root]
    count = 0
    while box_stack:
        box = box_stack.pop()
        count += 1
        if box.right is not None:
            box_stack.append(box.right)
        if box.left is not None:
            box_stack.append(box.left)
    return count

def np_flatten_bvh(root: FastTreeBox):
    box_count = count_boxes(root)
    box_arr = np.zeros(box_count, dtype=Box)

    triangle_count = len(root.triangles)
    triangle_arr = np.zeros(triangle_count, dtype=Triangle)

    box_index = 0
    triangle_index = 0
    box_queue = [root]
    while box_queue:
        box = box_queue.pop(0)

        box_arr[box_index]["min"][:3] = box.min
        box_arr[box_index]["max"][:3] = box.max

        if box.right is not None and box.left is not None:
            # if inner node (non-leaf), left is an index in flat_boxes
            box_arr[box_index]["left"] = box_index + len(box_queue) + 1
            # right will always be at left + 1, so use right as inner-vs-leaf flag
            box_arr[box_index]["right"] = 0
            # push children to queue
            box_queue.append(box.left)
            box_queue.append(box.right)
        elif box.right is not None or box.left is not None:
            raise ValueError("Box has only one child")
        else:
            l = triangle_index
            r = triangle_index + len(box.triangles)

            box_arr[box_index]["left"] = l
            box_arr[box_index]["right"] = r

            triangle_arr[l:r]["v0"][:, :3] = box.triangles[:, 0, :]
            triangle_arr[l:r]["v1"][:, :3] = box.triangles[:, 1, :]
            triangle_arr[l:r]["v2"][:, :3] = box.triangles[:, 2, :]
            triangle_arr[l:r]["n0"][:, :3] = box.smoothed_normals[:, 0]
            triangle_arr[l:r]["n1"][:, :3] = box.smoothed_normals[:, 1]
            triangle_arr[l:r]["n2"][:, :3] = box.smoothed_normals[:, 2]
            triangle_arr[l:r]["normal"][:, :3] = box.face_normals
            triangle_arr[l:r]["material"] = box.material
            triangle_arr[l:r]["is_light"] = box.emitter
            triangle_arr[l:r]["is_camera"] = box.camera

            triangle_index = r

        box_index += 1

    print(
        "flattened",
        box_count,
        "boxes and",
        triangle_count,
        "triangles. should match",
        triangle_index,
    )

    assert box_index == box_count
    assert triangle_index == triangle_count

    return box_arr, triangle_arr
