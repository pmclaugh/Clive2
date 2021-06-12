import numba
import numpy as np
from collision import simple_collision_screen_sample
from bidirectional import bidirectional_screen_sample
from unidirectional import unidirectional_screen_sample
from primitives import *
import pickle
from fastbvh import fastBVH
from camera import composite_image, tone_map
from load import load_obj


def triangles_for_box(box: FastBox, material=Material.DIFFUSE.value):
    span = box.max - box.min
    left_bottom_back = box.min
    right_bottom_back = box.min + span * UNIT_X
    left_top_back = box.min + span * UNIT_Y
    left_bottom_front = box.min + span * UNIT_Z

    right_top_front = box.max
    left_top_front = box.max - span * UNIT_X
    right_bottom_front = box.max - span * UNIT_Y
    right_top_back = box.max - span * UNIT_Z

    shrink = np.array([.5, .95, .5], dtype=np.float32)

    ret = numba.typed.List()
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
    for tri in tris:
        ret.append(tri)
    return ret


teapot_scene = load_obj('../resources/teapot.obj')
for triangle in triangles_for_box(FastBox(point(-10, -3, -10), point(10, 17, 10))):
    teapot_scene.append(triangle)


default_config = {
    'cam_center': point(0, 2, 6),
    'cam_direction': point(0, 0, -1),
    'window_width': 160,
    'window_height': 90,
    'sample_count': 20,
    'primitives': triangles_for_box(FastBox(point(-10, -3, -10), point(10, 17, 10))),
    'bvh_constructor': fastBVH,
    'sample_function': unidirectional_screen_sample,
    'postprocess_function': lambda x: tone_map(x.image),
}

deluxe_config = {
    'window_width': 640,
    'window_height': 480,
    'sample_count': 100,
}

bidirectional_config = {
    'sample_function': bidirectional_screen_sample,
    'postprocess_function': composite_image,
}

collision_test_config = {
    'sample_count': 10,
    'sample_function': simple_collision_screen_sample,
    'postprocess_function': lambda x: tone_map(x.image),
}

teapot_config = {
    'primitives': teapot_scene,
}