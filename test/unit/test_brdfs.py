import pytest
from routines import brdf_function, brdf_pdf, brdf_sample, specular_reflection, random_hemisphere_uniform_weighted, \
    random_hemisphere_cosine_weighted, local_orthonormal_system
import numpy as np
from constants import Material, Direction

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

        assert np.isclose(np.dot(x, y), 0)
        assert np.isclose(np.dot(y, z), 0)
        assert np.isclose(np.dot(x, z), 0)
        assert (z == random_z).all()


def test_specular_reflection(x_axis, y_axis, z_axis):
    for _ in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        exitant = specular_reflection(incident, z_axis)
        in_angle = np.dot(incident, z_axis)
        out_angle = np.dot(exitant, z_axis)
        assert np.isclose(in_angle, out_angle)
        # all 3 should be coplanar
        assert np.isclose(np.linalg.det(np.dstack([incident, z_axis, exitant])), 0)

    assert (specular_reflection(x_axis, z_axis) == -1 * x_axis).all()
    assert (specular_reflection(y_axis, z_axis) == -1 * y_axis).all()


def test_specular_reflection_wrong_side(x_axis, y_axis, z_axis):
    # the specular reflection function doesn't care which side of the normal you're on.
    for _ in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, -1 * z_axis)
        exitant = specular_reflection(incident, z_axis)

        assert np.isclose(np.dot(incident, -1 * z_axis), np.dot(exitant, -1 * z_axis))


def test_diffuse_BRDF_sample(x_axis, y_axis, z_axis):
    for _ in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        exitant = brdf_sample(Material.DIFFUSE.value, incident, z_axis, Direction.FROM_CAMERA.value)

        assert np.dot(exitant, z_axis) >= 0


# for now these tests are basically just documenting behavior. desired behavior might change
def test_diffuse_BRDF_sample_bad_incident(x_axis, y_axis, z_axis):
    incident = random_hemisphere_uniform_weighted(x_axis, y_axis, -1 * z_axis)
    exitant = brdf_sample(Material.DIFFUSE.value, incident, z_axis, Direction.FROM_CAMERA.value)

    # wrong-sided incident direction will NOT affect right-sidedness of output
    assert np.dot(exitant, z_axis) >= 0


def test_diffuse_BRDF_sample_bad_normal(x_axis, y_axis, z_axis):
    incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
    exitant = brdf_sample(Material.DIFFUSE.value, incident, -1 * z_axis, Direction.FROM_CAMERA.value)

    # wrong-sided normal direction WILL (obviously) affect right-sidedness of output
    assert np.dot(exitant, z_axis) <= 0


def test_diffuse_BRDF_function(x_axis, y_axis, z_axis):
    for i in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        exitant = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)

        assert brdf_function(Material.DIFFUSE.value, incident, z_axis, exitant, Direction.FROM_CAMERA.value) > 0


def test_diffuse_BRDF_function_sign_issues(x_axis, y_axis, z_axis):
    for i in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        exitant = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)

        # negative BRDF_function values come from directions pointing into the point rather than away
        assert brdf_function(Material.DIFFUSE.value, incident, z_axis, -1 * exitant, Direction.FROM_CAMERA.value) < 0
        assert brdf_function(Material.DIFFUSE.value, -1 * incident, z_axis, exitant, Direction.FROM_EMITTER.value) < 0


def test_diffuse_BRDF_pdf(x_axis, y_axis, z_axis):
    for i in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)
        exitant = random_hemisphere_uniform_weighted(x_axis, y_axis, z_axis)

        assert brdf_pdf(Material.DIFFUSE.value, incident, z_axis, exitant, Direction.FROM_CAMERA.value) > 0
        assert brdf_pdf(Material.DIFFUSE.value, incident, z_axis, exitant, Direction.FROM_EMITTER.value) > 0


def test_diffuse_BRDF_pdf_sign_issues(x_axis, y_axis, z_axis):
    for i in range(NUM_ITERS):
        incident = random_hemisphere_uniform_weighted(x_axis, y_axis, -1 * z_axis)
        exitant = random_hemisphere_uniform_weighted(x_axis, y_axis, -1 * z_axis)

        assert brdf_pdf(Material.DIFFUSE.value, incident, z_axis, exitant, Direction.FROM_CAMERA.value) < 0
        # brdf from emitter is constant and is never negative
        assert brdf_pdf(Material.DIFFUSE.value, incident, z_axis, exitant, Direction.FROM_EMITTER.value) > 0