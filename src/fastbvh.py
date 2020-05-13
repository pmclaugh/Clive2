import numba
from utils import timed
from constants import *
from primitives import TreeBox, FastBox
from load import load_obj


@timed
@numba.njit
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


@numba.njit
def construct_fastBVH(triangles):
    start_box = bound_triangles(triangles)

    stack = [start_box]
    while len(stack) > 0:
        box = stack.pop()
        if (len(box.triangles) <= MAX_MEMBERS) or len(stack) > MAX_DEPTH:
            continue
        l, r = object_split(box)
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
def object_split(box: TreeBox):
    boxes = [divide_box(box, axis, f) for axis in [UNIT_X, UNIT_Y, UNIT_Z] for f in np.arange(1 / OBJ_SPLITS, 1, 1 / OBJ_SPLITS)]
    bags = []
    # the bag/box metaphor: boxes are rigid and represent the division of space; bags are flexible & grow to fit members
    for left_box, right_box in boxes:
        left_bag = TreeBox(INF, NEG_INF)
        right_bag = TreeBox(INF, NEG_INF)
        left_tris = numba.typed.List()
        right_tris = numba.typed.List()
        for triangle in box.triangles:
            if left_box.contains_triangle(triangle):
                left_tris.append(triangle)
                left_bag.extend(triangle)
            elif right_box.contains_triangle(triangle):
                right_tris.append(triangle)
                right_bag.extend(triangle)
        if left_tris and right_tris:
            left_bag.triangles = left_tris
            right_bag.triangles = right_tris
            bags.append((left_bag, right_bag))
    # pick best one
    best_val = np.inf
    best_ind = 0
    for i, (lbag, rbag) in enumerate(bags):
        sah = surface_area_heuristic(lbag, rbag)
        if sah < best_val:
            best_val = sah
            best_ind = i
    return bags[best_ind]


@numba.njit
def surface_area(mins, maxes):
    span = maxes - mins
    return 2 * (span[0] * span[1] + span[1] * span[2] + span[2] * span[0])


@numba.njit
def overlap(l: TreeBox, r: TreeBox):
    # NYI
    return 0


@timed
def real_object_split(box: TreeBox):
    best_sah = np.inf
    best_split = None
    triangle_mins = np.array([t.mins for t in box.triangles])
    triangle_maxes = np.array([t.maxes for t in box.triangles])
    triangle_centers = (triangle_mins + triangle_maxes) / 2
    for axis in [UNIT_X, UNIT_Y, UNIT_Z]:
        axis_only = np.dot(triangle_centers, axis)
        axis_sorted = np.argsort(axis_only)
        ltr_maxes = np.maximum.accumulate(triangle_maxes[axis_sorted])
        ltr_mins = np.minimum.accumulate(triangle_mins[axis_sorted])
        rtl_maxes = np.maximum.accumulate(triangle_mins[axis_sorted[::-1]])[::-1]
        rtl_mins = np.minimum.accumulate(triangle_mins[axis_sorted[::-1]])[::-1]
        for i, left_max in enumerate(ltr_maxes[:-1]):
            left_min = ltr_mins[i]
            right_max = rtl_maxes[i + 1]
            right_min = rtl_mins[i + 1]
            sah = surface_area(left_min, left_max) * i + surface_area(right_min, right_max) * (len(ltr_maxes) - i)
            if sah < best_sah:
                best_sah = sah
                best_split = (i, left_min, left_max, right_min, right_max)

    i, left_min, left_max, right_min, right_max = best_split
    left_box = TreeBox(left_min, left_max)
    left_box.triangles = box.triangles[:i]
    right_box = TreeBox(right_min, right_max)
    right_box.triangles = box.triangles[i:]
    print('best split:', best_split, 'overlap:', overlap(left_box, right_box))

    return best_sah, left_box, right_box




@numba.njit
def bound_triangles(triangles):
    box = TreeBox(INF, NEG_INF)
    for triangle in triangles:
        box.extend(triangle)
    box.triangles = triangles
    return box


if __name__ == '__main__':
    # a = np.array([3, 0, 8, 2])
    # print(np.maximum.accumulate(a))
    # print(np.minimum.accumulate(a[::-1])[::-1])
    teapot = load_obj('../resources/teapot.obj')
    test_box = bound_triangles(teapot)
    print('test_box bounds:', test_box.min, test_box.max)
    real_object_split(test_box)
