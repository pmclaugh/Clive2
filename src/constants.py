import numpy as np

# camera constants
H_FOV = 110.0 * np.pi / 180.0

# directions
UNIT_X = np.array([1, 0, 0], dtype=np.float64)
UNIT_Y = np.array([0, 1, 0], dtype=np.float64)
UNIT_Z = np.array([0, 0, 1], dtype=np.float64)
INVALID = np.array([np.nan, np.nan, np.nan])
INF = np.array([np.inf, np.inf, np.inf])
NEG_INF = np.array([-np.inf, -np.inf, -np.inf])

# cv2 color order, colors defined [0, 1]
BLACK = np.array([0., 0., 0.], dtype=np.float64)
WHITE = np.array([.7, .7, .7], dtype=np.float64)
FULL_WHITE = np.array([1., 1., 1.], dtype=np.float64)
GRAY = np.array([.5, .5, .5], dtype=np.float64)
RED = np.array([0.3, 0.3, .8], dtype=np.float64)
GREEN = np.array([0.541, 0.807, 0.0], dtype=np.float64)
BLUE = np.array([.8, 0.3, 0.3], dtype=np.float64)
CYAN = np.array([.8, .8, 0.3], dtype=np.float64)


# BVH constants
MAX_MEMBERS = 16
MAX_DEPTH = 32
SPATIAL_SPLITS = 8
