import numpy as np
from struct_primitives import load_obj


if __name__ == '__main__':
    camera = np.zeros
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