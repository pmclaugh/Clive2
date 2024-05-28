import numpy as np

# define some numpy structs
Ray = np.dtype([
    ('origin', np.float32, 4),
    ('direction', np.float32, 4),
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
])
