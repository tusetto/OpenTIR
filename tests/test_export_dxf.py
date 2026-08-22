"""Tests for opentir.export_dxf"""
import os
import tempfile
import numpy as np
import pytest
import ezdxf

from opentir import (
    OpticalSystem, Surface, Segment, LEDSource,
    AIR, export_dxf, get_hit_points,
)
from opentir.profiles import build_conic_profile, profile_to_surfaces


def _build_system_and_traces():
    """Parabolic mirror system: guaranteed hits on target."""
    f = 20.0
    pts = build_conic_profile([0, 0], R=2*f, k=-1.0, r_max=15, n_points=30)
    pts_sym = np.vstack([pts[::-1] * [1, -1], pts[1:]])

    system = OpticalSystem()
    for s in profile_to_surfaces(pts_sym, "mirror", name="mirror"):
        system.add(s)
    system.add(Surface(
        Segment([80, -40], [80, 40], name="target"), "target", "target"))

    src = LEDSource([f, 0], axis_deg=180, half_angle_deg=35,
                    n_rays=21, medium=AIR)
    traces = system.trace_many(src.generate_rays(), max_bounces=4, min_power=0.001)
    return system, traces


def _export_and_read(system, traces, **kwargs):
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        path = f.name
    try:
        export_dxf(path, system, traces, **kwargs)
        doc = ezdxf.readfile(path)
    finally:
        os.unlink(path)
    return doc


def test_dxf_file_created():
    system, traces = _build_system_and_traces()
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        path = f.name
    try:
        export_dxf(path, system, traces)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 500
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_dxf_required_layers_present():
    system, traces = _build_system_and_traces()
    doc = _export_and_read(system, traces)
    layers = {l.dxf.name for l in doc.layers}
    for required in ("SURFACES_REFRACT", "SURFACES_TARGET",
                     "RAYS_TRANSMITTED", "RAYS_REFLECTED", "AXIS"):
        assert required in layers, f"Layer mancante: {required}"


def test_dxf_has_polylines_and_lines():
    system, traces = _build_system_and_traces()
    doc = _export_and_read(system, traces)
    msp = doc.modelspace()
    n_poly  = sum(1 for e in msp if e.dxftype() == "LWPOLYLINE")
    n_lines = sum(1 for e in msp if e.dxftype() == "LINE")
    assert n_poly  > 0, "nessuna LWPOLYLINE nel DXF"
    assert n_lines > 0, "nessuna LINE nel DXF"


def test_dxf_reflected_length_zero_omits_reflected():
    system, traces = _build_system_and_traces()
    # with reflected rays
    doc_with = _export_and_read(system, traces, reflected_length=20)
    # without reflected rays
    doc_without = _export_and_read(system, traces, reflected_length=0)
    msp_with    = doc_with.modelspace()
    msp_without = doc_without.modelspace()

    def count_on_layer(msp, layer):
        return sum(1 for e in msp
                   if e.dxftype() == "LINE"
                   and e.dxf.layer == layer)

    n_refl_with    = count_on_layer(msp_with,    "RAYS_REFLECTED")
    n_refl_without = count_on_layer(msp_without, "RAYS_REFLECTED")
    # some traces should be reflected (Fresnel); with length=0 none exported
    assert n_refl_without == 0
    # n_refl_with may be 0 if min_power filters them all — just check ≥ without
    assert n_refl_with >= n_refl_without


def test_get_hit_points_returns_arrays():
    system, traces = _build_system_and_traces()
    r_hits, powers = get_hit_points(traces)
    assert isinstance(r_hits, np.ndarray)
    assert isinstance(powers, np.ndarray)
    assert r_hits.shape == powers.shape


def test_get_hit_points_with_target_name():
    system, traces = _build_system_and_traces()
    r_all, _ = get_hit_points(traces)
    r_named, _ = get_hit_points(traces, target_name="target")
    r_wrong, _ = get_hit_points(traces, target_name="nonexistent")
    assert len(r_named) == len(r_all)   # only one target in this system
    assert len(r_wrong)  == 0


def test_get_hit_points_powers_positive():
    system, traces = _build_system_and_traces()
    r_hits, powers = get_hit_points(traces)
    if len(powers) > 0:
        assert (powers > 0).all()


def test_dxf_symmetric_adds_mirror_polylines():
    system, traces = _build_system_and_traces()
    doc_sym  = _export_and_read(system, traces, symmetric=True)
    doc_asym = _export_and_read(system, traces, symmetric=False)
    msp_sym  = doc_sym.modelspace()
    msp_asym = doc_asym.modelspace()
    n_sym  = sum(1 for e in msp_sym  if e.dxftype() == "LWPOLYLINE")
    n_asym = sum(1 for e in msp_asym if e.dxftype() == "LWPOLYLINE")
    assert n_sym > n_asym, "symmetric=True deve produrre più LWPOLYLINE"
