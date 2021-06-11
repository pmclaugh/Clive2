from bvh import triangles_for_box
from collision import simple_collision_screen_sample
from bidirectional import bidirectional_screen_sample
from unidirectional import unidirectional_screen_sample
from primitives import point, FastBox
import pickle
from fastbvh import fastBVH
from camera import composite_image, tone_map
from load import load_obj

teapot_scene = load_obj('../resources/teapot.obj')
# build simple room
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