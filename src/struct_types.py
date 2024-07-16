import numpy as np


Ray = np.dtype([
    ('origin', np.float32, 4),
    ('direction', np.float32, 4),
    ('inv_direction', np.float32, 4),
    ('color', np.float32, 4),
    ('normal', np.float32, 4),
    ('material', np.int32),
    ('triangle', np.int32),
    ('c_importance', np.float32),
    ('l_importance', np.float32),
    ('tot_importance', np.float32),
    ('hit_light', np.int32),
    ('from_camera', np.int32),
    ('pad', np.int32, 1),
])

Path = np.dtype([
    ('rays', Ray, 8),
    ('length', np.int32),
    ('from_camera', np.int32),
    ('pad', np.int32, 2),
])

Box = np.dtype([
    ('min', np.float32, 4),
    ('max', np.float32, 4),
    ('left', np.int32),
    ('right', np.int32),
    ('pad', np.int32, 2),
])

Triangle = np.dtype([
    ('v0', np.float32, 4),
    ('v1', np.float32, 4),
    ('v2', np.float32, 4),
    ('n0', np.float32, 4),
    ('n1', np.float32, 4),
    ('n2', np.float32, 4),
    ('normal', np.float32, 4),
    ('material', np.int32),
    ('is_light', np.int32),
    ('pad', np.int32, 2),
])

Material = np.dtype([
    ('color', np.float32, 4),
    ('emission', np.float32, 4),
    ('type', np.int32),
    ('alpha', np.float32),
    ('ior', np.float32),
    ('pad', np.int32),
])
