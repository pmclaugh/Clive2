from routines import unit
from collision import ray_triangle_intersect
import pytest
import numpy as np


@pytest.mark.unittest
def test_basic_triangle(basic_triangle, z_axis):
    assert (basic_triangle.normal == z_axis).all()


@pytest.mark.unittest
def test_wrong_handed_triangle(wrong_handed_triangle, z_axis):
    assert (wrong_handed_triangle.normal == z_axis * -1).all()


@pytest.mark.unittest
def test_should_hit(basic_triangle, ray_that_hits):
    t = ray_triangle_intersect(ray_that_hits, basic_triangle)

    assert t is not None
    assert t == 5.0


@pytest.mark.unittest
def test_should_miss(basic_triangle, ray_that_misses):
    t = ray_triangle_intersect(ray_that_misses, basic_triangle)

    assert t is None


@pytest.mark.unittest
def test_backface_culling(wrong_handed_triangle, ray_that_hits):
    t = ray_triangle_intersect(ray_that_hits, wrong_handed_triangle)

    assert t is None


@pytest.mark.unittest
def test_edge_hit(basic_triangle, ray_that_barely_hits):
    t = ray_triangle_intersect(ray_that_barely_hits, basic_triangle)

    assert t is not None
    assert t == 5.0


@pytest.mark.unittest
def test_sample_surface(basic_triangle):
    p = basic_triangle.sample_surface()

    assert 0 <= p[0] <= 1
    assert 0 <= p[1] <= 1
    assert p[2] == 0

    # unit vectors from chosen point to vertices
    pv0 = unit(basic_triangle.v0 - p)
    pv1 = unit(basic_triangle.v1 - p)
    pv2 = unit(basic_triangle.v2 - p)

    # if the point is in the triangle, the sum of the angles between those vectors is 2pi
    angle_sum = 0
    angle_sum += np.arccos(np.dot(pv0, pv1))
    angle_sum += np.arccos(np.dot(pv1, pv2))
    angle_sum += np.arccos(np.dot(pv2, pv0))
    
    assert np.isclose(angle_sum, 2 * np.pi)
