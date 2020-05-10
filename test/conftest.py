import pytest
from primitives import Ray, Triangle, FastBox, point
from constants import UNIT_X, UNIT_Y, UNIT_Z, ZEROS, ONES


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "unittest: mark test as an Unit Test"
    )


@pytest.fixture
def x_axis():
    return UNIT_X


@pytest.fixture
def y_axis():
    return UNIT_Y


@pytest.fixture
def z_axis():
    return UNIT_Z


@pytest.fixture
def unit_box():
    return FastBox(ZEROS, ONES)


@pytest.fixture
def basic_triangle():
    return Triangle(ZEROS, UNIT_X, UNIT_Y)


@pytest.fixture
def wrong_handed_triangle():
    return Triangle(ZEROS, UNIT_Y, UNIT_X)


@pytest.fixture
def big_triangle():
    return Triangle(ZEROS, UNIT_X * 5, UNIT_Y * 5)


@pytest.fixture
def ray_that_barely_hits():
    # hits object at origin
    return Ray(UNIT_Z * 5, -1 * UNIT_Z)


@pytest.fixture
def ray_that_hits():
    # hits object in center
    return Ray(point(0.2, 0.2, 5), -1 * UNIT_Z)


@pytest.fixture
def ray_inside_box():
    return Ray(point(0.5, 0.5, 0.5), UNIT_Z)


@pytest.fixture
def ray_that_misses():
    return Ray(ONES * 5, UNIT_Y)


