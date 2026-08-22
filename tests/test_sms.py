import numpy as np
from opentir import (
    Ray, OpticalSystem, Surface, Segment, AIR, PMMA,
    design_cartesian_oval_collimator, build_cartesian_oval_surface,
)


def _trace_collimation_error(theta_deg, n_points=400):
    points = design_cartesian_oval_collimator(
        AIR.n, PMMA.n, vertex_distance=3.0, theta_max_deg=45.0, n_points=n_points)
    surfaces = build_cartesian_oval_surface(points, AIR, PMMA)
    system = OpticalSystem()
    for s in surfaces:
        system.add(s)
    theta = np.radians(theta_deg)
    ray = Ray([0.0, 0.0], [np.cos(theta), np.sin(theta)], power=1.0, medium=AIR)
    branches = system.trace_ray(ray, max_bounces=4, min_power=1e-4)
    best = max(branches, key=lambda b: b["power"])
    path = best["path"]
    out_dir = np.array(path[-1]) - np.array(path[-2])
    out_dir /= np.linalg.norm(out_dir)
    return abs(np.degrees(np.arctan2(out_dir[1], out_dir[0])))


def test_collimation_on_axis():
    assert _trace_collimation_error(0.0) < 0.05


def test_collimation_various_angles():
    for theta_deg in [10, 20, 30, 40]:
        err = _trace_collimation_error(theta_deg)
        assert err < 0.05, f"theta={theta_deg}: errore {err} deg troppo grande"


def test_theta_max_limit_raises():
    import pytest
    with pytest.raises(ValueError):
        design_cartesian_oval_collimator(AIR.n, PMMA.n, vertex_distance=3.0,
                                          theta_max_deg=60.0, n_points=10)


def test_critical_angle_matches_theta_limit():
    # the max collectible half-angle should be arccos(n_source/n_lens)
    theta_limit = np.degrees(np.arccos(AIR.n / PMMA.n))
    # should NOT raise just below the limit
    design_cartesian_oval_collimator(AIR.n, PMMA.n, vertex_distance=3.0,
                                      theta_max_deg=theta_limit - 1.0, n_points=10)
