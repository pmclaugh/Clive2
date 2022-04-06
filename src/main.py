import cv2
from datetime import datetime
from collections import ChainMap
from primitives import FastBox, Triangle
from camera import Camera
from config import default_config, bidirectional_config, teapot_config, deluxe_config

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 90
SAMPLE_COUNT = 40

if __name__ == '__main__':
    cfg = ChainMap(bidirectional_config, teapot_config, default_config)
    camera = Camera(cfg['cam_center'], cfg['cam_direction'], pixel_height=cfg['window_height'],
                    pixel_width=cfg['window_width'], phys_width=cfg['window_width'] / cfg['window_height'], phys_height=1.)
    boxes = cfg['bvh_constructor'](cfg['primitives'])
    print("OK")
    # try:
    #     for n in range(cfg['sample_count']):
    #         cfg['sample_function'](camera, boxes, triangles, emitters)
    #         print('sample', n, 'done')
    #         cv2.imshow('render', cfg['postprocess_function'](camera))
    #         cv2.waitKey(1)
    # except KeyboardInterrupt:
    #     print('stopped early')
    # else:
    #     print('done')
    # cv2.imwrite('../renders/%s.jpg' % datetime.now(), cfg['postprocess_function'](camera))

# todo, returning to this after a year:
#  eliminate jitclass entirely, use numpy structured arrays
#  revisit multithreading
#  I hate this chainmap-config pattern, main should be clearer, maybe bring it back much later on
#  finish bvh implementation
#  integration tests