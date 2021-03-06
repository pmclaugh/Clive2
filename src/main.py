import cv2
from camera import Camera, composite_image, tone_map
from primitives import point, Box
from utils import timed
from datetime import datetime
from bvh import BoundingVolumeHierarchy, triangles_for_box
from load import load_obj
from bidirectional import bidirectional_screen_sample
from unidirectional import unidirectional_screen_sample
from constants import Material
from collections import ChainMap

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 90
SAMPLE_COUNT = 40


default_config = {
    'cam_center': point(0, 2, 6),
    'cam_direction': point(0, 0, -1),
    'window_width': 160,
    'window_height': 90,
    'sample_count': 10,
    'primitives': triangles_for_box(Box(point(-10, -3, -10), point(10, 17, 10))),
    'bvh_constructor': BoundingVolumeHierarchy,
    'sample_function': unidirectional_screen_sample,
    'postprocess_function': lambda x: tone_map(x.image),
}

bidirectional_config = ChainMap({
    'sample_function': bidirectional_screen_sample,
    'postprocess_function': composite_image,
}, default_config)


if __name__ == '__main__':
    cfg = bidirectional_config
    camera = Camera(cfg['cam_center'], cfg['cam_direction'], pixel_height=cfg['window_height'],
                    pixel_width=cfg['window_width'], phys_width=cfg['window_width'] / cfg['window_height'], phys_height=1.)
    bvh = cfg['bvh_constructor'](cfg['primitives'])

    try:
        for n in range(cfg['sample_count']):
            cfg['sample_function'](camera, bvh.root.box)
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
#  - Automated tests
#       - BVH unit tests
#       - integration tests around paths in a simple scene
#  - jit OBJ loading and bvh construction, eliminate TreeBox class
#  - BVH caching
#  - fix having to use this .value thing on all the enums. numba is supposed to support them
#  - requirements.txt

# todo: Known Bugs
#  - sample 0 does not display properly

