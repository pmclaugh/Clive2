import numpy as np
import enum

H_FOV = 90

FLOAT_TOLERANCE = 0.00001

UNIT_X = np.array([1, 0, 0], dtype=np.float32)
UNIT_Y = np.array([0, 1, 0], dtype=np.float32)
UNIT_Z = np.array([0, 0, 1], dtype=np.float32)

MAX3 = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
MIN3 = -1 * MAX3
ZEROS = np.array([0, 0, 0], dtype=np.float32)
ONES = np.array([1, 1, 1], dtype=np.float32)

# cv2 color order, colors defined [0, 1]
BLACK = np.array([0., 0., 0.], dtype=np.float32)
WHITE = np.array([1., 1., 1.], dtype=np.float32)
GRAY = np.array([.5, .5, .5], dtype=np.float32)
RED = np.array([0.4, 0.4, 1.], dtype=np.float32)
GREEN = np.array([0.4, 1., 0.4], dtype=np.float32)
BLUE = np.array([1., 0.4, 0.4], dtype=np.float32)
CYAN = np.array([1., 1., 0.4], dtype=np.float32)


# BVH constants
TRAVERSAL_COST = 1
INTERSECT_COST = 2

# Tracing constants
MAX_BOUNCES = 3
COLLISION_OFFSET = 0.001


# Bidirectional constants
class Direction(enum.Enum):
    FROM_CAMERA = 1
    FROM_EMITTER = 2