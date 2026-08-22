"""
opentir.export_dxf
~~~~~~~~~~~~~~~~~~~
Export the optical system geometry and ray paths to a DXF file.

Layer structure
---------------
SURFACES_REFRACT   – refractive surfaces (cyan)
SURFACES_MIRROR    – mirror surfaces (blue)
SURFACES_TARGET    – target planes (green)
SURFACES_BLOCK     – opaque baffles (red)
LENS_HATCH         – lens interior fill hatching (magenta, HATCH entities)
RAYS_TRANSMITTED   – transmitted / TIR / mirror rays (yellow)
RAYS_REFLECTED     – Fresnel partial-reflection rays (gray)
AXIS               – optical axis dashed line (white)

All coordinates are in mm, in the (z, r) plane, mapped to (X, Y) in
the DXF file (X = optical axis, Y = radial coordinate).
"""

import numpy as np
import ezdxf
from ezdxf import colors as dxf_colors
from ezdxf.enums import TextEntityAlignment

# DXF ACI colour indices
_COL = {
    "cyan":    4,
    "blue":    5,
    "green":   3,
    "red":     1,
    "magenta": 6,
    "yellow":  2,
    "gray":    8,
    "white":   7,
    "orange":  30,
}

_LAYER_DEFS = [
    ("SURFACES_REFRACT",  _COL["cyan"]),
    ("SURFACES_MIRROR",   _COL["blue"]),
    ("SURFACES_TARGET",   _COL["green"]),
    ("SURFACES_BLOCK",    _COL["red"]),
    ("LENS_HATCH",        _COL["magenta"]),
    ("RAYS_TRANSMITTED",  _COL["yellow"]),
    ("RAYS_REFLECTED",    _COL["gray"]),
    ("AXIS",              _COL["white"]),
]

_KIND_TO_LAYER = {
    "refract": "SURFACES_REFRACT",
    "mirror":  "SURFACES_MIRROR",
    "target":  "SURFACES_TARGET",
    "block":   "SURFACES_BLOCK",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _pts2d(pts_zr):
    """Convert (z, r) array to list of (x, y, 0) tuples for DXF."""
    return [(float(z), float(r), 0.0) for z, r in pts_zr]


def _add_lwpolyline(msp, pts_zr, layer, closed=False, ltype=None):
    pts = [(float(z), float(r)) for z, r in pts_zr]
    pl = msp.add_lwpolyline(pts, close=closed, dxfattribs={"layer": layer})
    if ltype:
        pl.dxf.linetype = ltype
    return pl


def _mirror_pts(pts_zr):
    """Mirror a (z, r) profile to (z, -r) for the lower half."""
    arr = np.array(pts_zr)
    return np.column_stack([arr[:, 0], -arr[:, 1]])


# ── main export function ──────────────────────────────────────────────────────

def export_dxf(path, system, traces,
                lens_fill_data=None,
                reflected_length=20.0,
                symmetric=True,
                axis_z_range=None,
                title="OpenTIR optical system"):
    """
    Export the optical system and ray traces to a DXF file.

    Parameters
    ----------
    path            : output file path (.dxf)
    system          : OpticalSystem instance
    traces          : list of trace dicts from system.trace_many()
    lens_fill_data  : list of (f_pts, r_pts, color, name) from GUI, or None
    reflected_length: max length of Fresnel-reflected ray segments (mm);
                      0 = omit reflected rays from export
    symmetric       : if True, mirror surface profiles below r=0
    axis_z_range    : (z_min, z_max) for the optical axis line, or None
                      (auto-computed from surface bounding box)
    title           : drawing title inserted as a text entity
    """
    doc = _build_dxf_doc(system, traces, lens_fill_data,
                          reflected_length, symmetric, axis_z_range, title)
    with open(path, "w", encoding="utf-8") as f:
        doc.write(f)


def _write_dxf_to_bytes(path, system, traces,
                         lens_fill_data=None,
                         reflected_length=20.0,
                         symmetric=True,
                         axis_z_range=None,
                         title="OpenTIR optical system"):
    """Internal: same as export_dxf but writes via open() for compatibility."""
    import io
    # build in-memory first
    buf = io.StringIO()
    # re-use the builder by calling export_dxf with a temp ezdxf document
    # We just write with open() directly instead
    doc = _build_dxf_doc(system, traces, lens_fill_data,
                          reflected_length, symmetric, axis_z_range, title)
    with open(path, "w", encoding="utf-8") as f:
        doc.write(f)


def _build_dxf_doc(system, traces, lens_fill_data=None,
                    reflected_length=20.0, symmetric=True,
                    axis_z_range=None,
                    title="OpenTIR optical system"):
    """Build and return the ezdxf document (without saving)."""
    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LTSCALE"] = 5.0
    doc.linetypes.add("DASHED",  pattern=[5.0, -2.0])
    doc.linetypes.add("DASHDOT", pattern=[5.0, -2.0, 0.5, -2.0])
    for name, colour in _LAYER_DEFS:
        doc.layers.add(name, color=colour)
    msp = doc.modelspace()
    msp.add_text(title, dxfattribs={"layer":"AXIS","height":3.0,"insert":(0,-20,0)})

    all_z = []
    for surf in system.surfaces:
        all_z.extend(surf.geometry.sample_points()[:,0].tolist())
    if traces:
        for t in traces:
            for pt in t["path"]:
                all_z.append(pt[0])

    if axis_z_range:
        z0, z1 = axis_z_range
    elif all_z:
        margin = max((max(all_z)-min(all_z))*0.05, 5.0)
        z0, z1 = min(all_z)-margin, max(all_z)+margin
    else:
        z0, z1 = -10.0, 100.0

    msp.add_line((z0,0,0),(z1,0,0),dxfattribs={"layer":"AXIS","linetype":"DASHED"})

    for surf in system.surfaces:
        pts   = surf.geometry.sample_points()
        layer = _KIND_TO_LAYER.get(surf.kind,"SURFACES_REFRACT")
        _add_lwpolyline(msp, pts, layer)
        if symmetric and surf.kind != "target":
            _add_lwpolyline(msp, _mirror_pts(pts), layer, ltype="DASHED")

    if lens_fill_data:
        for entry in lens_fill_data:
            f_pts, r_pts, color, name = entry
            upper_z = np.concatenate([f_pts[:,0], r_pts[::-1,0]])
            upper_r = np.concatenate([f_pts[:,1], r_pts[::-1,1]])
            for sign in (1, -1):
                boundary = list(zip(upper_z.tolist(),(upper_r*sign).tolist()))
                boundary.append(boundary[0])
                hatch = msp.add_hatch(color=_COL["magenta"],
                                      dxfattribs={"layer":"LENS_HATCH"})
                hatch.set_pattern_fill("ANSI31", scale=0.5, angle=45)
                hatch.paths.add_polyline_path(
                    [(z,r) for z,r in boundary], is_closed=True)
            msp.add_line((float(f_pts[0,0]),0,0),(float(r_pts[0,0]),0,0),
                         dxfattribs={"layer":"LENS_HATCH"})

    for trace in traces:
        is_refl = trace.get("reflected", False)
        path    = np.array(trace["path"])
        if is_refl:
            if reflected_length <= 0:
                continue
            clipped, total = [path[0]], 0.0
            for i in range(1,len(path)):
                sv  = path[i]-path[i-1]; sl = np.linalg.norm(sv)
                if total+sl >= reflected_length:
                    frac = (reflected_length-total)/sl
                    clipped.append(path[i-1]+frac*sv); break
                clipped.append(path[i]); total += sl
            path  = np.array(clipped)
            layer = "RAYS_REFLECTED"
        else:
            layer = "RAYS_TRANSMITTED"
        for i in range(len(path)-1):
            msp.add_line((float(path[i,0]),float(path[i,1]),0.0),
                         (float(path[i+1,0]),float(path[i+1,1]),0.0),
                         dxfattribs={"layer":layer})
    return doc


def get_hit_points(traces, target_name=None):
    """
    Return (r_values, powers) arrays for all ray hits on target surfaces.
    Used by the isophote plot.
    """
    r_hits, powers = [], []
    for trace in traces:
        for surface, point in trace["hits"]:
            if target_name and surface.name != target_name:
                continue
            r_hits.append(float(point[1]))
            powers.append(float(trace["power"]))
    return np.array(r_hits), np.array(powers)
