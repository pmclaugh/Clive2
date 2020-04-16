import numpy as np

H_FOV = 90

FLOAT_TOLERANCE = 0.00001

UNIT_X = np.array([1, 0, 0], dtype=np.float32)
UNIT_Y = np.array([0, 1, 0], dtype=np.float32)
UNIT_Z = np.array([0, 0, 1], dtype=np.float32)

MAX3 = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
MIN3 = -1 * MAX3
ZEROS = np.array([0, 0, 0], dtype=np.float32)
ONES = np.array([1, 1, 1], dtype=np.float32)

# cv2 color order!
BLACK = np.array([0, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
RED = np.array([0, 0, 255], dtype=np.uint8)
GREEN = np.array([0, 255, 0], dtype=np.uint8)
BLUE = np.array([255, 0, 0], dtype=np.uint8)