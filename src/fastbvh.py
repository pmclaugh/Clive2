import numba
import numpy as np
from utils import timed
from constants import *
from primitives import TreeBox
from load import load_obj


@timed
@numba.njit
def fastBVH(triangles):
    start_box = TreeBox(INF, NEG_INF)
    for triangle in triangles:
        start_box.extend(triangle)
    start_box.triangles = triangles
    for triangle in triangles:
        assert start_box.contains_triangle(triangle)

    stack = [start_box]
    while len(stack) > 0:
        box = stack.pop()
        l, r = object_split(box)
    return None, None


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
def object_split(box: TreeBox, n=8):
    boxes = [divide_box(box, axis, fraction)
              for axis in [UNIT_X, UNIT_Y, UNIT_Z]
              for fraction in np.arange(1/n, 1, 1/n)]
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
            else:
                print('triangle not in either box!')
        if left_tris and right_tris:
            left_bag.triangles = left_tris
            right_bag.triangles = right_tris
            bags.append((left_bag, right_bag))
        else:
            print('one-sided split!')
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
    fastBVH(teapot)
    print('done')