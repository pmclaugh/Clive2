import numpy as np
from constants import *
from struct_types import Box, Triangle


class TreeBox:
    def __init__(self, triangles):
        self.triangles = triangles
        self.left = None
        self.right = None
        self.parent = None

        self.min = INF
        self.max = NEG_INF
        for triangle in triangles:
            self.min = np.minimum(self.min, triangle.min)
            self.max = np.maximum(self.max, triangle.max)


def surface_area(mins, maxes):
    span = maxes - mins
    return 2 * (span[0] * span[1] + span[1] * span[2] + span[2] * span[0])


def volume(b: TreeBox):
    dims = b.max - b.min
    return dims[0] * dims[1] * dims[2]


def object_split(box: TreeBox):
    best_sah = np.inf
    best_split = None
    best_axis = None

    triangle_mins = np.array([t.min for t in box.triangles])
    triangle_maxes = np.array([t.max for t in box.triangles])
    triangle_centers = (triangle_mins + triangle_maxes) / 2

    for axis in [UNIT_X, UNIT_Y, UNIT_Z]:
        axis_only = np.dot(triangle_centers, axis)
        axis_sorted = np.argsort(axis_only)

        ltr_maxes = np.maximum.accumulate(triangle_maxes[axis_sorted])
        ltr_mins = np.minimum.accumulate(triangle_mins[axis_sorted])
        rtl_maxes = np.maximum.accumulate(triangle_maxes[axis_sorted[::-1]])[::-1]
        rtl_mins = np.minimum.accumulate(triangle_mins[axis_sorted[::-1]])[::-1]

        for i, left_max in enumerate(ltr_maxes[:-1]):
            left_min = ltr_mins[i]
            right_max = rtl_maxes[i + 1]
            right_min = rtl_mins[i + 1]
            sah = surface_area(left_min, left_max) * i + surface_area(right_min, right_max) * (len(ltr_maxes) - i)
            if sah < best_sah:
                best_sah = sah
                best_split = (i, left_min, left_max, right_min, right_max)
                best_axis = axis

    i, left_min, left_max, right_min, right_max = best_split
    left_triangles = []
    right_triangles = []
    sort_keys = np.argsort(np.dot(triangle_centers, best_axis))
    for j, key in enumerate(sort_keys):
        if j <= i:
            left_triangles.append(box.triangles[sort_keys[j]])
        else:
            right_triangles.append(box.triangles[sort_keys[j]])

    left_box = TreeBox(left_triangles)
    right_box = TreeBox(right_triangles)

    # print(best_axis, 'produced best split:', best_split)
    # print('sah', best_sah)
    # print('left vol:', volume(left_box) / volume(box), 'right vol:', volume(right_box) / volume(box))
    # print('left count:', len(left_box.triangles), 'right count:', len(right_box.triangles))

    return best_sah, left_box, right_box



def construct_BVH(triangles):
    start_box = TreeBox(triangles)

    stack = [start_box]
    while stack:
        box = stack.pop()
        if (len(box.triangles) <= MAX_MEMBERS) or len(stack) > MAX_DEPTH:
            continue

        sah_object, l_object, r_object = object_split(box)
        # todo - alpha test. compare (surface area of intersection of l_object + r_object) / surface area of root
        #  to some small alpha. if under, don't bother doing (costly) spatial split
        # sah_spatial, l_spatial, r_spatial = spatial_split(box)

        # if sah_spatial < sah_object:
        #     l, r = l_spatial, r_spatial
        # else:
        #     l, r = l_object, r_object

        l, r = l_object, r_object

        if r is not None:
            box.right = r
            r.parent = box
            stack.append(r)
        if l is not None:
            box.left = l
            l.parent = box
            stack.append(l)

    return start_box


def count_boxes(root: TreeBox):
    box_queue = [root]
    count = 0
    while box_queue:
        box = box_queue[0]
        count += 1
        if box.right is not None:
            box_queue.append(box.right)
        if box.left is not None:
            box_queue.append(box.left)
        box_queue = box_queue[1:]
    return count


def np_flatten_bvh(root: TreeBox):
    box_count = count_boxes(root)
    box_arr = np.zeros(box_count, dtype=Box)

    triangle_count = len(root.triangles)
    triangle_arr = np.zeros(triangle_count, dtype=Triangle)

    box_index = 0
    triangle_index = 0
    box_queue = [root]
    while box_queue:
        box = box_queue[0]

        box_arr[box_index]['min'][:3] = box.min
        box_arr[box_index]['max'][:3] = box.max

        if box.right is not None and box.left is not None:
            # if inner node (non-leaf), left is an index in flat_boxes
            box_arr[box_index]['left'] = box_index + len(box_queue)
            # right will always be at left + 1, so use right as inner-vs-leaf flag
            box_arr[box_index]['right'] = 0
            # push children to queue
            box_queue.append(box.left)
            box_queue.append(box.right)
        else:
            # if leaf, left is index into flat_triangles
            box_arr[box_index]['left'] = triangle_index

            for triangle in box.triangles:
                triangle_arr[triangle_index] = 0
                triangle_arr[triangle_index]['v0'][:3] = triangle.v0
                triangle_arr[triangle_index]['v1'][:3] = triangle.v1
                triangle_arr[triangle_index]['v2'][:3] = triangle.v2
                triangle_arr[triangle_index]['n0'][:3] = triangle.n0
                triangle_arr[triangle_index]['n1'][:3] = triangle.n1
                triangle_arr[triangle_index]['n2'][:3] = triangle.n2
                triangle_arr[triangle_index]['normal'][:3] = triangle.n
                triangle_arr[triangle_index]['material'] = triangle.material
                triangle_arr[triangle_index]['is_light'] = triangle.emitter
                triangle_index += 1

            # so now flat_triangles[left:right] is the triangles in this box. nonzero right signals leaf in traverse.
            box_arr[box_index]['right'] = triangle_index

        box_index += 1
        box_queue = box_queue[1:]

    return box_arr, triangle_arr
