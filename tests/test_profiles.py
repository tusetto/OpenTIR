import numpy as np
from opentir import (
    Ray, OpticalSystem, Surface, Segment, AIR, PMMA,
    build_conic_profile, profile_to_surfaces, conic_sag,
)


def test_conic_sphere_matches_circle_equation():
    R = 10.0
    r = np.linspace(0, 8, 20)
    z = conic_sag(r, R, k=0.0)
    assert np.allclose((R - z) ** 2 + r ** 2, R ** 2, atol=1e-10)


def test_conic_parabola_matches_formula():
    R = 10.0
    r = np.linspace(0, 8, 20)
    z = conic_sag(r, R, k=-1.0)
    assert np.allclose(z, r ** 2 / (2 * R), atol=1e-12)


def test_conic_general_equation_hyperbola_and_ellipse():
    R = 10.0
    r = np.linspace(0, 5, 10)
    for k in [-3.0, -0.5, 1.0]:
        z = conic_sag(r, R, k)
        resid = (1 + k) * z ** 2 - 2 * R * z + r ** 2
        assert np.max(np.abs(resid)) < 1e-9


def test_parabolic_mirror_collimates_from_focus():
    # focal length f: vertex at origin, focus at (f, 0); for k=-1, R=2f
    f = 15.0
    R = 2 * f
    points = build_conic_profile(vertex=[0.0, 0.0], R=R, k=-1.0, r_max=20.0, n_points=200)
    surfaces = profile_to_surfaces(points, kind="mirror", name="parabola")

    system = OpticalSystem()
    for s in surfaces:
        system.add(s)
    target = Segment([500.0, 0], [500.0, 60], name="target")
    system.add(Surface(target, kind="target", name="target"))

    for theta_deg in [5, 15, 30, 45]:
        theta = np.radians(theta_deg)
        origin = [f, 0.0]
        direction = [-np.cos(theta), np.sin(theta)]  # aim back toward the mirror (z < f)
        ray = Ray(origin, direction, power=1.0, medium=AIR)
        branches = system.trace_ray(ray, max_bounces=3, min_power=1e-4)
        best = max(branches, key=lambda b: b["power"])
        path = best["path"]
        assert len(path) >= 3, "il raggio dovrebbe riflettersi sullo specchio e arrivare al target"
        out_dir = np.array(path[-1]) - np.array(path[-2])
        out_dir /= np.linalg.norm(out_dir)
        angle_err = abs(np.degrees(np.arctan2(out_dir[1], out_dir[0])))
        assert angle_err < 0.5, f"theta={theta_deg}: errore collimazione {angle_err} deg"


def test_freeform_profile_roundtrip():
    from opentir import build_freeform_profile
    pts = [[0, 0], [1, 1], [2, 3]]
    profile = build_freeform_profile(pts)
    assert profile.shape == (3, 2)
    surfaces = profile_to_surfaces(profile, kind="mirror", name="free")
    assert len(surfaces) == 2
