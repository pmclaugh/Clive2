from primitives import Ray, Box, Triangle, ray_box_intersect, ray_triangle_intersect
import numpy as np
from typing import List
from load import load_obj
import logging
from constants import MIN3, MAX3, ZEROS, ONES
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
    minimum = MAX3
    maximum = MIN3
    for triangle in triangles:
        minimum = np.minimum(triangle.mins, minimum)
        maximum = np.maximum(triangle.maxes, maximum)
    return TreeBox(minimum, maximum, members=list(triangles))


class BoundingVolumeHierarchy:
    def __init__(self, triangles):
        self.root = None
        self.triangles = triangles
        self.build()

    def build(self, max_members=16, max_depth=10):
        self.root = AABB(self.triangles)
        self.root.members = list(self.triangles)
        logger.info('root bounding box is from %s to %s', self.root.box.min, self.root.box.max)
        stack: List[TreeBox] = [self.root]
        while stack:
            box = stack.pop()
            if (box.members is not None and len(box.members) <= max_members) or len(stack) > max_depth:
                # leaf node
                logger.info('leaf node: %s', box)
                continue
            l, r = self.split(box)
            stack.append(l)
            stack.append(r)

    @staticmethod
    def divide_box(box: Box, axis: int, fraction: float):
        # return two boxes resulting from splitting input box along axis at fraction
        left_max = box.max.copy()
        left_max[axis] -= box.span[axis] * fraction
        right_min = box.min.copy()
        right_min[axis] += box.span[axis] * (1 - fraction)
        return TreeBox(box.min, left_max), TreeBox(right_min, box.max)

    @classmethod
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

        # allow boxes to shrink
        splits = [(AABB(lbox.members), AABB(rbox.members)) for lbox, rbox in splits if lbox.members and rbox.members]

        left, right = sorted(splits, key=lambda s: np.product(s[0].box.span) + np.product(s[1].box.span))[0]
        box_to_split.children = [left, right]
        left.parent = box_to_split
        right.parent = box_to_split
        return left, right

    @timed
    def hit(self, ray: Ray):
        least_t = np.inf
        least_hit = None
        stack: List[TreeBox] = [self.root]
        while stack:
            box = stack.pop()
            hit, t_low, t_high = box.collide(ray)
            if hit and t_low <= least_t:
                if box.children:
                    stack += box.children
                else:
                    for triangle in box.members:
                        t = ray_triangle_intersect(ray, triangle)
                        if t is not None and t < least_t:
                            least_t = t
                            least_hit = triangle
        return least_hit, least_t


if __name__ == '__main__':
    teapot = load_obj('resources/teapot.obj')
    BoundingVolumeHierarchy(teapot)