import numba
from utils import timed
from constants import *
from primitives import TreeBox, FastBox
from load import load_obj


@timed
# @numba.njit
def fastBVH(triangles):
    root = construct_fastBVH(triangles)
    return flatten_fastBVH(root)


@numba.njit
def flatten_fastBVH(root: TreeBox):
    flat_boxes = numba.typed.List()
    flat_triangles = numba.typed.List()
    emitters = numba.typed.List()
    box_queue = [root]
    while box_queue:
        box = box_queue[0]
        fast_box = FastBox(box.min, box.max)
        if box.right is not None and box.left is not None:
            fast_box.left = len(flat_boxes) + len(box_queue)
            box_queue.append(box.left)
            box_queue.append(box.right)
            # right will always be left + 1, this is how we signal inner node in fast traverse
            fast_box.right = 0
        else:
            fast_box.left = len(flat_triangles)
            for triangle in box.triangles:
                if triangle.emitter:
                    emitters.append(triangle)
                flat_triangles.append(triangle)
            fast_box.right = len(flat_triangles)
            # so now triangles[left:right] is the triangles in this box
        flat_boxes.append(fast_box)
        box_queue = box_queue[1:]
    return flat_boxes, flat_triangles, emitters


# @numba.njit
@timed
def construct_fastBVH(triangles):
    start_box = bound_triangles(triangles)

    stack = [start_box]
    while len(stack) > 0:
        box = stack.pop()
        if (len(box.triangles) <= MAX_MEMBERS) or len(stack) > MAX_DEPTH:
            continue
        sah_object, l_object, r_object = object_split(box)
        # sah_spatial, l_spatial, r_spatial = spatial_split(box)
        # l, r = (l_spatial, r_spatial) if sah_spatial < sah_object else (l_object, r_object)
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


@numba.njit
def divide_box(box: TreeBox, axis, fraction: float):
    delta = (box.max - box.min) * axis
    return TreeBox(box.min, box.max - delta * fraction), TreeBox(box.min + delta * (1 - fraction), box.max)


@numba.njit
def surface_area_heuristic(l: TreeBox, r: TreeBox):
    return TRAVERSAL_COST + sah_component(l) + sah_component(r)


@numba.njit
def sah_component(box: TreeBox):
    return len(box.triangles) * INTERSECT_COST * box.surface_area()


@numba.njit
def surface_area(mins, maxes):
    span = maxes - mins
    return 2 * (span[0] * span[1] + span[1] * span[2] + span[2] * span[0])


@numba.njit
def volume(b: TreeBox):
    dims = b.max - b.min
    return dims[0] * dims[1] * dims[2]


@numba.njit
def overlap(l: TreeBox, r: TreeBox):
    # NYI
    return 0


@timed
def object_split(box: TreeBox):
    best_sah = np.inf
    best_split = None
    best_axis = None
    triangle_mins = np.array([t.mins for t in box.triangles])
    triangle_maxes = np.array([t.maxes for t in box.triangles])
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
    left_triangles = numba.typed.List()
    right_triangles = numba.typed.List()
    sort_keys = np.argsort(np.dot(triangle_centers, best_axis))
    for j, key in enumerate(sort_keys):
        if j <= i:
            left_triangles.append(box.triangles[sort_keys[j]])
        else:
            right_triangles.append(box.triangles[sort_keys[j]])
    left_box = TreeBox(left_min, left_max)
    left_box.triangles = left_triangles
    right_box = TreeBox(right_min, right_max)
    right_box.triangles = right_triangles

    print(best_axis, 'produced best split:', best_split)
    print('left vol:', volume(left_box) / volume(box), 'right vol:', volume(right_box) / volume(box))

    # todo make this a separate test
    test_left = bound_triangles(left_box.triangles)
    test_right = bound_triangles(right_box.triangles)
    assert np.equal(test_left.min, left_box.min).all()
    assert np.equal(test_left.max, left_box.max).all()
    assert np.equal(test_right.min, right_box.min).all()
    assert np.equal(test_right.max, right_box.max).all()

    return best_sah, left_box, right_box


@numba.njit
def bound_triangles(triangles):
    box = TreeBox(INF, NEG_INF)
    for triangle in triangles:
        box.extend(triangle)
    box.triangles = triangles
    return box


# @numba.njit
def partition_box(box: TreeBox, axis, num_partitions):
    delta = (box.max - box.min) * axis
    partitions = []
    delta_step = delta / num_partitions
    part_min = box.min
    part_max = box.max * (1 - axis) + box.min * axis + delta_step
    for _ in range(num_partitions - 1):
        partitions.append(TreeBox(part_min, part_max))
        part_min = part_min + delta_step
        part_max = part_max + delta_step
    return partitions


# @numba.njit
@timed
def spatial_split(box: TreeBox):
    best_sah = np.inf
    best_split = None
    for axis in [UNIT_X, UNIT_Y, UNIT_Z]:
        boxes = partition_box(box, axis, SPATIAL_SPLITS)


if __name__ == '__main__':
    # a = np.array([3, 0, 8, 2])
    # print(np.maximum.accumulate(a))
    # print(np.minimum.accumulate(a[::-1])[::-1])

    # teapot = load_obj('../resources/teapot.obj')

    from primitives import Triangle
    simple_triangles = numba.typed.List()
    for shift in [0, 2, 5, 12, 32]:
        delta = UNIT_X * shift
        t = Triangle(ZEROS + delta, UNIT_X + delta, UNIT_Y + UNIT_Z + delta)
        print(t.v0, t.v1, t.v2)
        print(t.maxes, t.mins)
        simple_triangles.append(t)

    test_box = bound_triangles(simple_triangles)
    print('test_box bounds:', test_box.min, test_box.max)
    spatial_split(test_box)