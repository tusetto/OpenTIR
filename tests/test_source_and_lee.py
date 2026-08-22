"""Tests for opentir.source (LES) and opentir.lee (LEE breakdown)."""
import numpy as np
import pytest

from opentir import AIR, PMMA, LEDSource
from opentir.lee import compute_lee, LEEResult
from opentir import OpticalSystem, Surface, Segment
from opentir.profiles import build_conic_profile, profile_to_surfaces


# ═══════════════════════════════════════════════════════════════════════════════
# LEDSource / LES tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLEDSourcePoint:
    def test_point_ray_count(self):
        src = LEDSource([0, 0], n_rays=21, les_shape="point")
        assert len(src.generate_rays()) == 21

    def test_point_power_normalised(self):
        src = LEDSource([0, 0], n_rays=41, distribution="lambertian")
        rays = src.generate_rays()
        assert abs(sum(r.power for r in rays) - 1.0) < 1e-9

    def test_point_uniform_power(self):
        src = LEDSource([0, 0], n_rays=11, distribution="uniform")
        rays = src.generate_rays()
        assert abs(sum(r.power for r in rays) - 1.0) < 1e-9

    def test_point_medium_set(self):
        src = LEDSource([0, 0], n_rays=5, medium=PMMA)
        for r in src.generate_rays():
            assert r.medium is PMMA

    def test_total_rays_property(self):
        src = LEDSource([0, 0], n_rays=21, les_shape="point")
        assert src.total_rays == 21


class TestLEDSourceSquare:
    def test_square_ray_count(self):
        src = LEDSource([0, 0], n_rays=11, les_shape="square",
                        les_size=1.0, n_les=5)
        rays = src.generate_rays()
        assert len(rays) == 11 * 5

    def test_square_power_normalised(self):
        src = LEDSource([0, 0], n_rays=21, les_shape="square",
                        les_size=2.0, n_les=7, distribution="lambertian")
        rays = src.generate_rays()
        assert abs(sum(r.power for r in rays) - 1.0) < 1e-9

    def test_square_origin_spread(self):
        src = LEDSource([0, 0], n_rays=1, les_shape="square",
                        les_size=2.0, n_les=5)
        rays = src.generate_rays()
        r_vals = [r.origin[1] for r in rays]
        # sub-sources span from -1.0 to +1.0 mm (half of 2.0)
        assert min(r_vals) < -0.9
        assert max(r_vals) >  0.9

    def test_square_n1_gives_single_center(self):
        src = LEDSource([0, 0], n_rays=5, les_shape="square",
                        les_size=1.0, n_les=1)
        rays = src.generate_rays()
        assert len(rays) == 5
        for r in rays:
            assert abs(r.origin[1]) < 1e-9  # centre

    def test_total_rays_property(self):
        src = LEDSource([0, 0], n_rays=11, les_shape="square",
                        les_size=1.0, n_les=5)
        assert src.total_rays == 11 * 5


class TestLEDSourceCircle:
    def test_circle_ray_count(self):
        src = LEDSource([0, 0], n_rays=11, les_shape="circle",
                        les_size=1.0, n_les=9)
        assert len(src.generate_rays()) == 11 * 9

    def test_circle_power_normalised(self):
        src = LEDSource([0, 0], n_rays=21, les_shape="circle",
                        les_size=2.0, n_les=12, distribution="lambertian")
        rays = src.generate_rays()
        assert abs(sum(r.power for r in rays) - 1.0) < 1e-9

    def test_circle_origins_within_radius(self):
        diameter = 2.0
        src = LEDSource([0, 0], n_rays=1, les_shape="circle",
                        les_size=diameter, n_les=20)
        rays = src.generate_rays()
        for r in rays:
            assert abs(r.origin[1]) <= diameter / 2 + 1e-9

    def test_les_description(self):
        src = LEDSource([0, 0], n_rays=1, les_shape="circle", les_size=1.5)
        desc = src.les_description()
        assert "circolare" in desc
        assert "1.50" in desc


class TestLEDSourceInvalid:
    def test_invalid_les_shape_raises(self):
        src = LEDSource([0, 0], n_rays=5, les_shape="hexagon", les_size=1.0)
        with pytest.raises(ValueError):
            src.generate_rays()

    def test_invalid_distribution_raises(self):
        src = LEDSource([0, 0], n_rays=5, distribution="cosine2")
        with pytest.raises(ValueError):
            src.generate_rays()


# ═══════════════════════════════════════════════════════════════════════════════
# LEE tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mirror_system_with_target():
    """Parabolic mirror — most rays reach target."""
    f = 20.0
    pts = build_conic_profile([0, 0], R=2*f, k=-1.0, r_max=15, n_points=30)
    pts_sym = np.vstack([pts[::-1] * [1, -1], pts[1:]])
    system = OpticalSystem()
    for s in profile_to_surfaces(pts_sym, "mirror", name="mirror"):
        system.add(s)
    system.add(Surface(Segment([80, -40], [80, 40], name="target"),
                       "target", "target"))
    return system


class TestComputeLEE:
    def _run(self, n_rays=21):
        system = _make_mirror_system_with_target()
        src = LEDSource([20, 0], axis_deg=180, half_angle_deg=35,
                        n_rays=n_rays, medium=AIR)
        base_rays = src.generate_rays()
        total_in = sum(r.power for r in base_rays)
        traces = system.trace_many(base_rays, max_bounces=4, min_power=0.001)
        return compute_lee(traces, total_in), total_in

    def test_returns_lee_result(self):
        result, _ = self._run()
        assert isinstance(result, LEEResult)

    def test_power_conservation(self):
        result, total_in = self._run()
        accounted = (result.target_power + result.fresnel_power +
                     result.tir_power + result.blocked_power +
                     result.escaped_power)
        assert abs(accounted - total_in) < 1e-9

    def test_eta_sum_to_one(self):
        result, _ = self._run()
        total_eta = (result.eta_target + result.eta_fresnel +
                     result.eta_tir + result.eta_blocked +
                     result.eta_escaped)
        assert abs(total_eta - 1.0) < 1e-9

    def test_mirror_system_has_high_target_efficiency(self):
        result, _ = self._run(n_rays=41)
        # mirror system: >90% should reach target
        assert result.eta_target > 0.80

    def test_eta_values_in_range(self):
        result, _ = self._run()
        for eta in (result.eta_target, result.eta_fresnel,
                    result.eta_tir, result.eta_blocked, result.eta_escaped):
            assert 0.0 <= eta <= 1.0

    def test_summary_contains_percentages(self):
        result, _ = self._run()
        summary = result.summary()
        assert "%" in summary
        assert "LEE" in summary

    def test_target_name_filter(self):
        system = _make_mirror_system_with_target()
        # add a second target that no rays hit
        system.add(Surface(Segment([200, -5], [200, 5], name="target2"),
                           "target", "target2"))
        src = LEDSource([20, 0], axis_deg=180, half_angle_deg=35,
                        n_rays=21, medium=AIR)
        base = src.generate_rays()
        traces = system.trace_many(base, max_bounces=4, min_power=0.001)
        total = sum(r.power for r in base)
        r_all   = compute_lee(traces, total)
        r_main  = compute_lee(traces, total, target_name="target")
        r_dummy = compute_lee(traces, total, target_name="target2")
        assert r_main.target_power >= r_dummy.target_power
        assert r_dummy.target_power == pytest.approx(0.0, abs=1e-6)
