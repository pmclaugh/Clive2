import cv2
from datetime import datetime
from collections import ChainMap

from camera import Camera
from config import default_config, bidirectional_config, teapot_config, deluxe_config

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 90
SAMPLE_COUNT = 40

if __name__ == '__main__':
    cfg = ChainMap(teapot_config, default_config)
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

# todo cleanup:
#  review code and identify cleanup areas
#  examine uncommitted changes
#  reassess performance and multithreading in light of newer numba version
#  fix sample 0 problem
#  bvh tests (performance and unit)

# todo features:
#  bvh caching
#  bvh details: reference unsplitting, alpha test in spatial
#  normal smoothing
#  simple textures
#  GGX BRDF
#  s == 0, 1
#  t == 0, 1
