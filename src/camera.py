import numpy as np
from constants import FLOAT_TOLERANCE, UNIT_Y, UNIT_Z, UNIT_X, H_FOV

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 90
SAMPLE_COUNT = 40
CAM_CENTER = np.array([0, 2, 6], dtype=float)
CAM_DIRECTION = np.array([0, 0, -1], dtype=float)
CAM_PHYS_WIDTH = 1.
CAM_PHYS_HEIGHT = 1.
CAM_PIX_WIDTH = 160
CAM_PIX_HEIGHT = 90

Camera = np.dtype([
    ('origin', (float, 3)),
    ('dx_dp', (float, 3)),
    ('dy_dp', (float, 3)),
    ('focal_point', (float, 3)),
])

def unit(v):
    return v / np.linalg.norm(v)

def setup_camera():
    if abs(CAM_DIRECTION[0]) < FLOAT_TOLERANCE:
        dx = UNIT_X if CAM_DIRECTION[2] > 0 else UNIT_X * -1
    else:
        dx = unit(np.cross(CAM_DIRECTION * (UNIT_X + UNIT_Z), UNIT_Y * -1))

    if abs(CAM_DIRECTION[1]) < FLOAT_TOLERANCE:
        dy = UNIT_Y
    else:
        dy = unit(np.cross(CAM_DIRECTION, dx))

    camera = np.zeros(1, dtype=Camera)[0]
    focal_dist = (CAM_PHYS_WIDTH / 2.) / np.tan(H_FOV / 2.0)
    camera['focal_point']  = CAM_CENTER + focal_dist * CAM_DIRECTION
    camera['dx_dp'] = dx * CAM_PHYS_WIDTH / CAM_PIX_WIDTH
    camera['dy_dp'] = dy * CAM_PHYS_HEIGHT / CAM_PIX_HEIGHT
    camera['origin'] = CAM_CENTER - dx * CAM_PHYS_WIDTH / 2 - dy * CAM_PHYS_HEIGHT / 2

    return camera


