import pytest
from routines import BRDF_function, BRDF_pdf, BRDF_sample, specular_reflection, random_hemisphere_uniform_weighted
import numpy as np

NUM_ITERS = 10


def test_specular_reflection(x_axis, y_axis, z_axis):
    for _ in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        exitant = specular_reflection(incident, z_axis)
        assert np.isclose(np.dot(incident, z_axis), np.dot(exitant, z_axis))
