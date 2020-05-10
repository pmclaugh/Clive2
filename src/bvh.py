from primitives import FastBox, Triangle, BoxStack
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
        self.box = FastBox(low_corner, high_corner)

    def extend(self, triangle: Triangle):
        self.box.extend(triangle)
        self.members.append(triangle)

    def finalize(self):
        if not self.children:
            members = numba.typed.List()
            [members.append(member) for member in self.members]
            self.box.triangles = members
        else:
            self.box.left = self.children[0].box
            self.box.right = self.children[1].box

        # root box holds list of emissive surfaces
        if self.parent is None:
            lights = numba.typed.List()
            [lights.append(member) for member in self.members if member.emitter]
            self.box.lights = lights
            self.box.light_SA = sum(light.surface_area for light in lights)

    def add(self, triangle: Triangle):
        # useful if we know the bounds won't change, or don't want them to
        self.members.append(triangle)

    def __contains__(self, triangle: Triangle):
        return self.box.contains(triangle.v0) or self.box.contains(triangle.v1) or self.box.contains(triangle.v2)

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return "{members} members, area {area}".format(members=len(self.members), area=np.product(self.box.span))


def AABB(triangles: List[Triangle]):
    box = TreeBox(MAX3, MIN3)
    for triangle in triangles:
        box.extend(triangle)
    return box


class BoundingVolumeHierarchy:
    def __init__(self, triangles):
        self.root = None
        self.triangles = triangles
        self.build()

    def __getstate__(self):
        flat_boxes, flat_triangles = self.flatten()
        return {'boxes': [self.jsonize_box(box) for box in flat_boxes],
                'triangles': [self.jsonize_triangle(triangle) for triangle in flat_triangles]}

    def __setstate__(self, state):
        pass

    @staticmethod
    def jsonize_triangle(triangle: Triangle):
        return {'v0': triangle.v0,
                'v1': triangle.v1,
                'v2': triangle.v2,
                'color': triangle.color,
                'emitter': triangle.emitter,
                'material': triangle.material,
                }

    @staticmethod
    def jsonize_box(box: FastBox):
        return {'min': box.min, 'max': box.max, 'count': 0 if box.triangles is None else len(box.triangles)}

    def flatten(self):
        traversal_queue = [self.root.box]
        flattened_boxes = []
        flattened_triangles = []
        while traversal_queue:
            b = traversal_queue.pop(0)
            if b.left is not None:
                traversal_queue.append(b.left)
            if b.right is not None:
                traversal_queue.append(b.right)
            if b.triangles is not None:
                for triangle in b.triangles:
                    flattened_triangles.append(triangle)
            flattened_boxes.append(b)
        return flattened_boxes, flattened_triangles

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
                box.children.append(r)
                r.parent = box
                stack.append(r)
            if l is not None:
                box.children.append(l)
                l.parent = box
                stack.append(l)
            box.finalize()

    @staticmethod
    def divide_box(box: FastBox, axis: int, fraction: float):
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
            return left, right
        else:
            logger.info('splitting failed on box: %s', box_to_split)
            return None, None


@timed
@numba.njit
def fastBVH(triangles):
    start_box = FastBox(INF, NEG_INF)
    for triangle in triangles:
        start_box.extend(triangle)

    stack = BoxStack()
    stack.push(start_box)
    while stack.size > 0:
        box = stack.pop()



def triangles_for_box(box: FastBox, material=Material.DIFFUSE.value):
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
        Triangle(left_bottom_back, right_bottom_back, right_top_back, color=RED, material=material),
        Triangle(left_bottom_back, right_top_back, left_top_back, color=RED, material=material),
        # left wall
        Triangle(left_bottom_back, left_top_front, left_bottom_front, color=BLUE, material=material),
        Triangle(left_bottom_back, left_top_back, left_top_front, color=BLUE, material=material),
        # right wall
        Triangle(right_bottom_back, right_bottom_front, right_top_front, color=GREEN, material=material),
        Triangle(right_bottom_back, right_top_front, right_top_back, color=GREEN, material=material),
        # front wall
        Triangle(left_bottom_front, right_top_front, right_bottom_front, color=CYAN, material=material),
        Triangle(left_bottom_front, left_top_front, right_top_front, color=CYAN, material=material),
        # floor
        Triangle(left_bottom_back, right_bottom_front, right_bottom_back, color=WHITE, material=material),
        Triangle(left_bottom_back, left_bottom_front, right_bottom_front, color=WHITE, material=material),
        # ceiling
        Triangle(left_top_back, right_top_back, right_top_front, color=WHITE, material=material),
        Triangle(left_top_back, right_top_front, left_top_front, color=WHITE, material=material),
        # ceiling light # NB this assumes box is centered on the origin, at least wrt x and z
        Triangle(left_top_back * shrink, right_top_back * shrink, right_top_front * shrink, color=WHITE, emitter=True),
        Triangle(left_top_back * shrink, right_top_front * shrink, left_top_front * shrink, color=WHITE, emitter=True),
    ]
    return tris


if __name__ == '__main__':
    teapot = load_obj('../resources/teapot.obj')
    fastBVH(teapot)