from primitives import Ray, Box, Triangle, ray_box_intersect, ray_triangle_intersect, BoxStack
import numpy as np
from typing import List
from load import load_obj
import logging
from constants import *
import numba
from utils import timed

logger = logging.getLogger('rtv3-BVH')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class TreeBox:
    def __init__(self, low_corner, high_corner, parent=None, children=None, members=None):
        self.parent = parent
        self.children = children if children is not None else []
        self.members = members if members is not None else []
        self.box = Box(low_corner, high_corner)

    def extend(self, triangle: Triangle):
        self.box.extend(triangle)
        self.members.append(triangle)

    def finalize(self):
        members = numba.typed.List()
        if not self.children:
            [members.append(member) for member in self.members]
            self.box.triangles = members
        else:
            self.box.left = self.children[0].box
            self.box.right = self.children[1].box

    def add(self, triangle: Triangle):
        # useful if we know the bounds won't change, or don't want them to
        self.members.append(triangle)

    def collide(self, ray: Ray):
        return ray_box_intersect(ray, self.box)

    def __contains__(self, triangle: Triangle):
        return self.box.contains(triangle.v0) or self.box.contains(triangle.v1) or self.box.contains(triangle.v2)

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return "{members} members, area {area}".format(members=len(self.members), area=np.product(self.box.span))


def AABB(triangles: List[Triangle]):
    # < 0.001 seconds
    box = TreeBox(MAX3, MIN3)
    for triangle in triangles:
        box.extend(triangle)
    return box


class BoundingVolumeHierarchy:
    def __init__(self, triangles):
        self.root = None
        self.triangles = triangles
        self.build()

    def build(self, max_members=32, max_depth=15):
        self.root = AABB(self.triangles)
        self.root.members = list(self.triangles)
        logger.info('root bounding box is from %s to %s', self.root.box.min, self.root.box.max)
        stack: List[TreeBox] = [self.root]
        while stack:
            box = stack.pop()
            if (len(box.members) <= max_members) or len(stack) > max_depth:
                if len(stack) > max_depth and len(box.members) > max_members:
                    logger.info('too deep! making a leaf with %d members', len(box.members))
                box.finalize()
                continue
            l, r = self.split(box)
            if r is not None:
                stack.append(r)
            if l is not None:
                stack.append(l)
            box.finalize()

    @staticmethod
    def divide_box(box: Box, axis: int, fraction: float):
        # ~0.0001 seconds
        # return two boxes resulting from splitting input box along axis at fraction
        left_max = box.max.copy()
        left_max[axis] -= box.span[axis] * fraction
        right_min = box.min.copy()
        right_min[axis] += box.span[axis] * (1 - fraction)
        return TreeBox(box.min, left_max), TreeBox(right_min, box.max)

    @staticmethod
    def surface_area_heuristic(split):
        l, r = split
        l_sa = l.box.surface_area()
        r_sa = r.box.surface_area()
        return TRAVERSAL_COST + l_sa * len(l.members) * INTERSECT_COST + r_sa * len(r.members) * INTERSECT_COST

    @classmethod
    @timed
    def split(cls, box_to_split: TreeBox, n=4):
        # simple rigid splitting method for now
        box = box_to_split.box
        triangles = box_to_split.members
        splits = [cls.divide_box(box, axis, fraction)
                  for axis in [0, 1, 2]
                  for fraction in np.arange(1/n, 1, 1/n)]
        for triangle in triangles:
            for lbox, rbox in splits:
                # arbitrary decision to tiebreak toward left
                if triangle in lbox:
                    lbox.add(triangle)
                elif triangle in rbox:
                    rbox.add(triangle)
                else:
                    logger.error('Triangle %s in neither box')
        splits = [(lbox, rbox) for lbox, rbox in splits if lbox.members and rbox.members]
        splits = sorted(splits, key=lambda s: cls.surface_area_heuristic(s))

        if splits:
            split = splits[0]
            left, right = AABB(split[0].members), AABB(split[1].members)
            box_to_split.children = [left, right]
            left.parent = box_to_split
            right.parent = box_to_split
            return left, right
        else:
            logger.info('splitting failed on box: %s', box_to_split)
            return None, None


@numba.jit(nogil=True)
def traverse_bvh(root: Box, ray: Ray):
    least_t = np.inf
    least_hit = None
    stack = BoxStack()
    stack.push(root)
    while stack.size:
        box = stack.pop()
        if box.left is not None or box.right is not None:
            if bvh_hit_inner(ray, box, least_t):
                stack.push(box.left)
                stack.push(box.right)
        else:
            hit, t = bvh_hit_leaf(ray, box, least_t)
            if hit is not None and t < least_t:
                least_hit = hit
                least_t = t

    return least_hit, least_t


@numba.jit(nogil=True, fastmath=True)
def bvh_hit_inner(ray: Ray, box: Box, least_t: float):
    hit, t_low, t_high = ray_box_intersect(ray, box)
    return hit and t_low <= least_t


@numba.jit(nogil=True, fastmath=True)
def bvh_hit_leaf(ray: Ray, box: Box, least_t):
    hit, t_low, t_high = ray_box_intersect(ray, box)
    if not hit:
        return None, least_t
    least_hit = None
    for triangle in box.triangles:
        t = ray_triangle_intersect(ray, triangle)
        if t is not None and 0 < t < least_t:
            least_t = t
            least_hit = triangle
    return least_hit, least_t


def triangles_for_box(box: Box):
    left_bottom_back = box.min
    right_bottom_back = box.min + box.span * UNIT_X
    left_top_back = box.min + box.span * UNIT_Y
    left_bottom_front = box.min + box.span * UNIT_Z

    right_top_front = box.max
    left_top_front = box.max - box.span * UNIT_X
    right_bottom_front = box.max - box.span * UNIT_Y
    right_top_back = box.max - box.span * UNIT_Z

    shrink = np.array([.5, .95, .5], dtype=np.float32)

    tris = [
        # back wall
        Triangle(left_bottom_back, right_bottom_back, right_top_back, color=RED),
        Triangle(left_bottom_back, right_top_back, left_top_back, color=RED),
        # left wall
        Triangle(left_bottom_back, left_top_front, left_bottom_front, color=BLUE),
        Triangle(left_bottom_back, left_top_back, left_top_front, color=BLUE),
        # right wall
        Triangle(right_bottom_back, right_bottom_front, right_top_front, color=GREEN),
        Triangle(right_bottom_back, right_top_front, right_top_back, color=GREEN),
        # front wall
        Triangle(left_bottom_front, right_top_front, right_bottom_front, color=GRAY),
        Triangle(left_bottom_front, left_top_front, right_top_front, color=GRAY),
        # floor
        Triangle(left_bottom_back, right_bottom_front, right_bottom_back, color=WHITE),
        Triangle(left_bottom_back, left_bottom_front, right_bottom_front, color=WHITE),
        # ceiling
        Triangle(left_top_back, right_top_back, right_top_front, color=CYAN),
        Triangle(left_top_back, right_top_front, left_top_front, color=CYAN),
        # ceiling light # NB this assumes box is centered on the origin, at least wrt x and z
        Triangle(left_top_back * shrink, right_top_back * shrink, right_top_front * shrink, color=WHITE, emitter=True),
        Triangle(left_top_back * shrink, right_top_front * shrink, left_top_front * shrink, color=WHITE, emitter=True),
    ]
    return tris


if __name__ == '__main__':
    teapot = load_obj('../resources/teapot.obj')
    BoundingVolumeHierarchy(teapot)