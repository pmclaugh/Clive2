import cv2
from camera import Camera, composite_image, tone_map
from primitives import point, FastBox
from datetime import datetime
from bvh import BoundingVolumeHierarchy, triangles_for_box
from load import load_obj
from constants import Material
from bidirectional import bidirectional_screen_sample
from unidirectional import unidirectional_screen_sample
from collections import ChainMap
import pickle
from fastbvh import fastBVH

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 90
SAMPLE_COUNT = 40


default_config = {
    'cam_center': point(0, 2, 6),
    'cam_direction': point(0, 0, -1),
    'window_width': 160,
    'window_height': 90,
    'sample_count': 10,
    'primitives': triangles_for_box(FastBox(point(-10, -3, -10), point(10, 17, 10))),
    'bvh_constructor': BoundingVolumeHierarchy,
    'sample_function': unidirectional_screen_sample,
    'postprocess_function': lambda x: tone_map(x.image),
}

bidirectional_config = {
    'sample_function': bidirectional_screen_sample,
    'postprocess_function': composite_image,
}

teapot_config = {
    'bvh_constructor': fastBVH,
    'primitives': load_obj('../resources/teapot.obj'),
}

# todo: branch 'bvh_improvements':
#  - bvh caching
#     - make caching an explicit action and use configs to use them (skips the whole "did the config change??" question)
#  - bvh performance test
#  - unit tests for bvh methods
#  - spatial splitting


if __name__ == '__main__':
    cfg = ChainMap(teapot_config, default_config)
    camera = Camera(cfg['cam_center'], cfg['cam_direction'], pixel_height=cfg['window_height'],
                    pixel_width=cfg['window_width'], phys_width=cfg['window_width'] / cfg['window_height'], phys_height=1.)
    bvh = cfg['bvh_constructor'](cfg['primitives'])
    try:
        for n in range(cfg['sample_count']):
            cfg['sample_function'](camera, bvh)
            print('sample', n, 'done')
            cv2.imshow('render', cfg['postprocess_function'](camera))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print('stopped early')
    else:
        print('done')
    cv2.imwrite('../renders/%s.jpg' % datetime.now(), cfg['postprocess_function'](camera))


# todo: Feature Schedule
#  - BVH improvements
#  - t == 0, 1
#  - glossy brdf
#  - glossy in bidirectional
#  - specular in bidirectional
#  - normal smoothing
#  - textures

# todo: Tech Debt
#  - do a deep dive on where i do and don't need to use .copy() in the jit classes
#  - Automated tests
#       - BVH unit tests
#       - integration tests around paths in a simple scene
#  - BVH caching
#  - fix having to use this .value thing on all the enums. numba is supposed to support them

# todo: Known Bugs
#  - sample 0 does not display properly

