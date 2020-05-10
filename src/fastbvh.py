import numba
from utils import timed
from constants import INF, NEG_INF
from primitives import TreeBox, FastBox, Triangle
from load import load_obj


@timed
@numba.njit
def fastBVH(triangles):
    start_box = TreeBox(INF, NEG_INF)
    for triangle in triangles:
        start_box.extend(triangle)

    stack = numba.typed.List()
    stack.append(start_box)
    while len(stack) > 0:
        box = stack.pop()


if __name__ == '__main__':
    teapot = load_obj('../resources/teapot.obj')
    fastBVH(teapot)