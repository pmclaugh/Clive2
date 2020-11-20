import cv2
from camera import Camera
from primitives import point, FastBox
from datetime import datetime
from bvh import triangles_for_box
from load import load_obj
from collections import ChainMap
from config import default_config, bidirectional_config, teapot_config

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 90
SAMPLE_COUNT = 40

teapot_scene = load_obj('../resources/teapot.obj')

# build simple room
for triangle in triangles_for_box(FastBox(point(-10, -3, -10), point(10, 17, 10))):
    teapot_scene.append(triangle)

if __name__ == '__main__':
    cfg = ChainMap(bidirectional_config, teapot_config, default_config)
    camera = Camera(cfg['cam_center'], cfg['cam_direction'], pixel_height=cfg['window_height'],
                    pixel_width=cfg['window_width'], phys_width=cfg['window_width'] / cfg['window_height'], phys_height=1.)
    boxes, triangles, emitters = cfg['bvh_constructor'](cfg['primitives'])
    try:
        for n in range(cfg['sample_count']):
            cfg['sample_function'](camera, boxes, triangles, emitters)
            print('sample', n, 'done')
            cv2.imshow('render', cfg['postprocess_function'](camera))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print('stopped early')
    else:
        print('done')
    cv2.imwrite('../renders/%s.jpg' % datetime.now(), cfg['postprocess_function'](camera))


# todo: branch 'bvh_improvements':
#  - full implementation of nvidia sbvh - done except for some details
#    - alpha test for spatial
#    - reference unsplitting
#  - bvh caching
#  - bvh performance test
#  - unit tests for bvh methods
#  - jit object splitting
#  - jit spatial splitting


# todo: Feature Schedule
#  - BVH improvements
#  - optimization pass - FastTriangle, parallelization
#  - t == 0, 1
#  - glossy brdf
#  - glossy in bidirectional
#  - specular in bidirectional
#  - normal smoothing
#  - textures

# todo: Tech Debt
#  - do a deep dive on where i do and don't need to use .copy() in the jit classes
#  - new config pattern has some problems (my triangle-generating functions now return numba lists)
#  - Automated tests
#       - BVH unit tests
#       - integration tests around paths in a simple scene
#  - BVH caching
#  - fix having to use this .value thing on all the enums. numba is supposed to support them

# todo: Known Bugs
#  - sample 0 does not display properly

