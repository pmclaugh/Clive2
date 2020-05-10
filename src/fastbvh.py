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
    start_box = TreeBox(INF, NEG_INF)
    for triangle in triangles:
        start_box.extend(triangle)
    start_box.triangles = triangles

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
    l_sa = l.surface_area()
    r_sa = r.surface_area()
    return TRAVERSAL_COST + l_sa * len(l.triangles) * INTERSECT_COST + r_sa * len(r.triangles) * INTERSECT_COST


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


if __name__ == '__main__':
    teapot = load_obj('../resources/teapot.obj')
    flat_boxes, flat_triangles = fastBVH(teapot)