import numpy as np
from opentir import snell_refract, fresnel_reflectance, critical_angle, PMMA, AIR


def test_snell_normal_incidence_no_bending():
    d = np.array([1.0, 0.0])
    n = np.array([-1.0, 0.0])  # points against the incident ray
    result = snell_refract(d, n, 1.0, 1.5)
    assert result is not None
    d_t, cos_i, cos_t = result
    assert np.isclose(cos_i, 1.0)
    assert np.isclose(cos_t, 1.0)
    assert np.allclose(d_t, [1.0, 0.0], atol=1e-8)


def test_fresnel_normal_incidence_matches_simple_formula():
    # At normal incidence, R = ((n1-n2)/(n1+n2))^2 regardless of polarization
    n1, n2 = 1.0, 1.5
    R_expected = ((n1 - n2) / (n1 + n2)) ** 2
    R = fresnel_reflectance(1.0, 1.0, n1, n2)
    assert np.isclose(R, R_expected, atol=1e-9)


def test_critical_angle_pmma_air():
    theta_c = critical_angle(PMMA.n, AIR.n)
    assert theta_c is not None
    assert np.isclose(np.degrees(theta_c), 42.16, atol=0.1)


def test_critical_angle_none_when_entering_denser_medium():
    assert critical_angle(AIR.n, PMMA.n) is None


def test_tir_beyond_critical_angle():
    theta_c = critical_angle(PMMA.n, AIR.n)
    # ray at 60 deg (> critical angle) inside PMMA hitting a flat interface
    theta = np.radians(60)
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-1.0, 0.0])
    result = snell_refract(d, n, PMMA.n, AIR.n)
    assert result is None  # TIR


def test_refraction_below_critical_angle_transmits():
    theta = np.radians(20)  # well below ~42 deg critical angle
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-1.0, 0.0])
    result = snell_refract(d, n, PMMA.n, AIR.n)
    assert result is not None
    d_t, cos_i, cos_t = result
    # going from dense to less dense medium, the ray bends away from the normal
    assert cos_t < cos_i
