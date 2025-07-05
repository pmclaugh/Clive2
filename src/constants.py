import numpy as np

# camera constants
H_FOV = 110.0 * np.pi / 180.0

# directions
UNIT_X = np.array([1, 0, 0], dtype=np.float64)
UNIT_Y = np.array([0, 1, 0], dtype=np.float64)
UNIT_Z = np.array([0, 0, 1], dtype=np.float64)
ZERO_VECTOR = np.array([0, 0, 0], dtype=np.float64)
INVALID = np.array([np.nan, np.nan, np.nan])
INF = np.array([np.inf, np.inf, np.inf])
NEG_INF = np.array([-np.inf, -np.inf, -np.inf])

# cv2 color order, colors defined [0, 1]
BLACK = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WHITE = np.array([0.7, 0.7, 0.7], dtype=np.float64)
FULL_WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float64)
GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float64)
RED = np.array([0.3, 0.3, 0.8], dtype=np.float64)
GREEN = np.array([0.541, 0.807, 0.0], dtype=np.float64)
BLUE = np.array([0.8, 0.3, 0.3], dtype=np.float64)
CYAN = np.array([0.8, 0.8, 0.3], dtype=np.float64)


# BVH constants
MAX_MEMBERS = 16
MAX_DEPTH = 32
SPATIAL_SPLITS = 8

# Scene constants
DEFAULT_BOX_MIN_CORNER = np.array([-10, -2, -10])
DEFAULT_BOX_MAX_CORNER = np.array([10, 10, 10])
DEFAULT_LIGHT_HEIGHT = 0.95
DEFAULT_LIGHT_SCALE = 0.25
