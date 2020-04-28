import pytest
from routines import random_hemisphere_uniform_weighted, random_hemisphere_cosine_weighted, local_orthonormal_system
import numpy as np
from constants import FLOAT_TOLERANCE

NUM_ITERS = 10


@pytest.mark.unittest
def test_nonnegative_dot_product_uniform(x_axis, y_axis, z_axis):
    for _ in range(NUM_ITERS):
        d = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        assert np.dot(d, z_axis) >= 0


@pytest.mark.unittest
def test_nonnegative_dot_product_cosine(x_axis, y_axis, z_axis):
    for _ in range(NUM_ITERS):
        d = random_hemisphere_cosine_weighted(x_axis, y_axis, z_axis)
        assert np.dot(d, z_axis) >= 0


@pytest.mark.unittest
def test_basic_orthornormal(z_axis):
    x, y, z = local_orthonormal_system(z_axis)
    assert np.dot(x, y) == 0
    assert np.dot(y, z) == 0
    assert np.dot(x, z) == 0
    assert (z == z_axis).all()


@pytest.mark.unittest
def test_random_orthonormal(z_axis):
    for _ in range(NUM_ITERS):
        random_z = random_hemisphere_uniform_weighted(*local_orthonormal_system(z_axis))
        x, y, z = local_orthonormal_system(random_z)
        assert np.isclose(np.dot(x, y), 0, atol=FLOAT_TOLERANCE)
        assert np.isclose(np.dot(y, z), 0, atol=FLOAT_TOLERANCE)
        assert np.isclose(np.dot(x, z), 0, atol=FLOAT_TOLERANCE)
        assert (z == random_z).all()
