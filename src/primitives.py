import numpy as np


def point(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def vec(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def unit(v):
    return v / np.linalg.norm(v)
