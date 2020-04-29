import pytest
from collision import ray_box_intersect
from primitives import point
from constants import ONES, ZEROS


@pytest.mark.unittest
def test_should_hit(ray_that_hits, unit_box):
    hit, tmin, tmax = ray_box_intersect(ray_that_hits, unit_box)
    assert hit
    assert tmin > 0
    assert tmax > tmin


@pytest.mark.unittest
def test_should_miss(ray_that_misses, unit_box):
    hit, tmin, tmax = ray_box_intersect(ray_that_misses, unit_box)
    assert not hit
    assert tmin == 0
    assert tmax == 0


@pytest.mark.unittest
def test_inside_box(ray_inside_box, unit_box):
    hit, tmin, tmax = ray_box_intersect(ray_inside_box, unit_box)
    assert hit
    assert tmin < 0
    assert tmax > 0


def test_surface_area(unit_box):
    assert unit_box.surface_area() == 6


def test_contains_point(unit_box, ray_inside_box, x_axis, y_axis, z_axis):
    assert unit_box.contains(ray_inside_box.origin)
    assert unit_box.contains(x_axis)
    assert unit_box.contains(y_axis)
    assert unit_box.contains(z_axis)
    assert not unit_box.contains(-1 * x_axis)
    assert not unit_box.contains(-1 * y_axis)
    assert not unit_box.contains(-1 * z_axis)
    assert not unit_box.contains(point(10, 2, 4))


def test_extend(unit_box, big_triangle):
    assert (unit_box.span == ONES).all()
    assert (unit_box.min == ZEROS).all()
    assert (unit_box.max == ONES).all()
    unit_box.extend(big_triangle)
    assert (unit_box.span == point(5, 5, 1)).all()
    assert (unit_box.min == ZEROS).all()
    assert (unit_box.max == point(5, 5, 1)).all()