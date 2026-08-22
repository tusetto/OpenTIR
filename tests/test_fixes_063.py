"""
Tests for the three fixes in 0.6.3:
  1. No inner-offset curves in plot_system / _plot_chromatic
  2. Auto-zoom on optical elements (not ray endpoints)
  3. Physical symmetry: r>0 and r<0 rays refract identically
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opentir import (
    OpticalSystem, Surface, Segment, Ray, LEDSource,
    AIR, PMMA, build_conic_profile,
)
from opentir.profiles import profile_to_surfaces
from opentir.visualize import plot_system, _offset_points
from opentir.gui import _build_symmetric_system, _mirror_surface_object


# ── helpers ──────────────────────────────────────────────────────────────────

def _lens_system_full_r(r_max=12, n_pts=30):
    """Lens with surfaces defined on full r range [-r_max, +r_max]."""
    system = OpticalSystem()
    system.add(Surface(
        Segment([5, -r_max], [5, r_max], name="fronte"),
        "refract", "fronte", AIR, PMMA))
    pts = build_conic_profile([8, 0], R=20, k=0, r_max=r_max, n_points=n_pts)
    pts_sym = np.vstack([pts[::-1] * [1, -1], pts[1:]])
    for s in profile_to_surfaces(pts_sym, "refract",
                                  material_in=PMMA, material_out=AIR,
                                  name="retro"):
        system.add(s)
    system.add(Surface(
        Segment([80, -20], [80, 20], name="target"), "target", "target"))
    return system


# ── Fix 1: no inner-offset curves ────────────────────────────────────────────

class TestNoOffsetCurves:
    def test_plot_system_line_count_no_offset(self):
        """With offset removed, each non-target surface produces max 2 lines
        (upper solid + lower dashed mirror). Before the fix it was 4 lines."""
        system = OpticalSystem()
        system.add(Surface(
            Segment([5, 0], [5, 10], name="lens"), "refract", "lens", AIR, PMMA))

        fig, ax = plt.subplots()
        plot_system(system, traces=[], ax=ax, symmetric=True)
        lines = ax.get_lines()
        # axis line + 1 surface×2 (upper+lower) = 3 lines max
        # with offset it would have been axis + 4 = 5+ lines
        assert len(lines) <= 3, (
            f"Troppe linee ({len(lines)}): le curve di offset non devono essere presenti")
        plt.close(fig)

    def test_no_linewidth_4_lines(self):
        """Offset curves had linewidth=4; verify none remain."""
        system = OpticalSystem()
        system.add(Surface(
            Segment([5, 0], [5, 10], name="lens"), "refract", "lens", AIR, PMMA))
        fig, ax = plt.subplots()
        plot_system(system, traces=[], ax=ax, symmetric=True)
        lw4_lines = [l for l in ax.get_lines() if l.get_linewidth() >= 3.5]
        assert len(lw4_lines) == 0, "Nessuna curva di offset (lw=4) attesa"
        plt.close(fig)


# ── Fix 2: auto-zoom on optical elements ─────────────────────────────────────

class TestAutoZoom:
    def test_xlim_within_surface_bbox(self):
        """After plot_system, xlim must be within surface bounding box + padding,
        NOT extended to ray endpoints (which can be 1000 mm away)."""
        system = OpticalSystem()
        system.add(Surface(
            Segment([5, -10], [5, 10], name="lens"), "refract", "lens", AIR, PMMA))
        system.add(Surface(
            Segment([20, -10], [20, 10], name="target"), "target", "target"))

        # traces with very long endpoint
        long_trace = {
            "path": [np.array([0, 0]), np.array([5, 5]), np.array([1000, 500])],
            "hits": [], "power": 0.9, "reflected": False, "wavelength_nm": 589.3
        }

        fig, ax = plt.subplots()
        plot_system(system, traces=[long_trace], ax=ax, symmetric=True)
        xmin, xmax = ax.get_xlim()
        # surface z range is [5, 20]; with padding should be well below 200
        assert xmax < 200, f"xlim max={xmax} troppo grande (raggi fino a z=1000)"
        assert xmin > -30, f"xlim min={xmin} troppo piccolo"
        plt.close(fig)

    def test_ylim_covers_surfaces(self):
        """ylim must cover the surface radial extent."""
        system = OpticalSystem()
        system.add(Surface(
            Segment([5, -12], [5, 12], name="lens"), "refract", "lens", AIR, PMMA))
        fig, ax = plt.subplots()
        plot_system(system, traces=[], ax=ax, symmetric=True)
        ymin, ymax = ax.get_ylim()
        assert ymin <= -12, f"ylim min={ymin} non copre r=-12"
        assert ymax >= 12,  f"ylim max={ymax} non copre r=+12"
        plt.close(fig)


# ── Fix 3: physical symmetry ─────────────────────────────────────────────────

class TestPhysicalSymmetry:
    def _r_final(self, system, r_orig):
        ray = Ray([0, r_orig], [1, 0], medium=AIR)
        branches = system.trace_ray(ray, max_bounces=4, min_power=0.001)
        best = max(branches, key=lambda b: b["power"])
        return best["path"][-1][1]

    def test_symmetric_refraction_single_surface(self):
        """A flat refractive surface spanning [-r,+r] must refract
        symmetric rays to symmetric exit positions."""
        system = OpticalSystem()
        system.add(Surface(
            Segment([5, -15], [5, 15], name="lens"), "refract", "lens", AIR, PMMA))
        system.add(Surface(
            Segment([100, -30], [100, 30], name="target"), "target", "target"))

        for r in [2, 4, 6, 8]:
            rp = self._r_final(system, +r)
            rn = self._r_final(system, -r)
            assert abs(rp + rn) < 0.05, (
                f"r={r}: r_final(+)={rp:.4f} r_final(-)={rn:.4f} "
                f"non simmetrici (diff={abs(rp+rn):.4f})")

    def test_symmetric_full_lens(self):
        """Full lens with bilateral surfaces must give symmetric results."""
        system = _lens_system_full_r()

        for r in [2, 5, 9]:
            rp = self._r_final(system, +r)
            rn = self._r_final(system, -r)
            assert abs(rp + rn) < 0.1, (
                f"Lente completa r={r}: (+)={rp:.3f} (-)={rn:.3f} "
                f"non simmetrici")

    def test_mirror_surface_object_only_for_target_block(self):
        """_mirror_surface_object must return None for refract/mirror kinds."""
        surf_refract = Surface(
            Segment([5, 0], [5, 10], name="x"), "refract", "x", AIR, PMMA)
        surf_mirror = Surface(
            Segment([5, 0], [5, 10], name="y"), "mirror", "y")
        surf_target = Surface(
            Segment([80, 0], [80, 10], name="z"), "target", "z")

        assert _mirror_surface_object(surf_refract) is None
        assert _mirror_surface_object(surf_mirror)  is None
        assert _mirror_surface_object(surf_target)  is not None

    def test_build_symmetric_system_count(self):
        """_build_symmetric_system must add mirror only for target/block."""
        raw = [
            Surface(Segment([5, 0], [5, 10], name="fronte"),
                    "refract", "fronte", AIR, PMMA),
            Surface(Segment([30, 0], [30, 10], name="specchio"),
                    "mirror", "specchio"),
            Surface(Segment([80, 0], [80, 10], name="target"),
                    "target", "target"),
        ]
        system = _build_symmetric_system(raw)
        # only target should be mirrored → 4 surfaces total
        assert len(system.surfaces) == 4, (
            f"Attese 4 superfici (3 originali + 1 mirror del target), "
            f"trovate {len(system.surfaces)}")

    def test_already_symmetric_target_not_duplicated(self):
        """A target already spanning r<0 must NOT be duplicated."""
        raw = [
            Surface(Segment([80, -10], [80, 10], name="target"),
                    "target", "target"),
        ]
        system = _build_symmetric_system(raw)
        assert len(system.surfaces) == 1
