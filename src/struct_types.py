import numpy as np


Ray = np.dtype([
    ('origin', np.float32, 4),
    ('direction', np.float32, 4),
    ('inv_direction', np.float32, 4),
    ('color', np.float32, 4),
    ('importance', np.float32),
    ('hit_light', np.int32),
    ('i', np.int32),
    ('j', np.int32),
])

Path = np.dtype([
    ('rays', Ray, 16),
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
    ('normal', np.float32, 4),
    ('material', np.int32),
    ('is_light', np.int32),
    ('pad', np.int32, 2),
])

Material = np.dtype([
    ('color', np.float32, 4),
    ('emission', np.float32, 4),
    ('type', np.int32),
    ('pad', np.int32, 3),
])
