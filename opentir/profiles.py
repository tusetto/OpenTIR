"""
opentir.profiles
~~~~~~~~~~~~~~~~~
General surface profile generators.

Covers the full range of classic optical surface shapes via the
standard conic + aspheric sag equation, plus arbitrary free-form
profiles from a user-supplied point list:

    z(r) = c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2)) + A4*r^4 + A6*r^6 + ...

    c = 1/R   (R = vertex radius of curvature)
    k = conic constant:
        k = 0            -> sphere / circle (equivalent to the Arc primitive)
        k = -1           -> parabola
        k < -1           -> hyperbola
        -1 < k < 0       -> ellipse (prolate)
        k > 0            -> ellipse (oblate)
    A4, A6, ... = even-order aspheric correction coefficients (optional)

Surfaces are represented, as elsewhere in OpenTIR, as a chain of many
short Segment elements (a polyline approximation) rather than an
analytic primitive - consistent with how the SMS module builds its
profiles, and simplest to combine with the existing exact-segment ray
tracer. Increase n_points for a more accurate approximation of the
true curve (see release 0.3 notes on discretization error).
"""

import numpy as np

from .geometry import Segment
from .optics import Surface


def conic_sag(r, R, k, coeffs=()):
    """Sag z(r) of a conic + aspheric surface with vertex at the origin."""
    r = np.asarray(r, dtype=float)
    if R == 0:
        z = np.zeros_like(r)
    else:
        c = 1.0 / R
        under_sqrt = 1.0 - (1.0 + k) * c ** 2 * r ** 2
        if np.any(under_sqrt < 0):
            raise ValueError(
                "Superficie non definita per questo r_max (radicando negativo): "
                "riduci r_max o aumenta il raggio di curvatura R"
            )
        z = c * r ** 2 / (1.0 + np.sqrt(under_sqrt))
    for i, A in enumerate(coeffs):
        order = 4 + 2 * i
        z = z + A * r ** order
    return z


def build_conic_profile(vertex, R, k, r_max, coeffs=(), n_points=80, flip_z=False):
    """
    Build a [z, r] point array for a conic/aspheric surface.

    vertex  : [z, r] position of the surface's vertex (apex, r=0 point)
    R       : vertex radius of curvature (mm)
    k       : conic constant (see module docstring for the classic cases)
    r_max   : maximum radial extent of the profile
    coeffs  : even-order aspheric coefficients (A4, A6, A8, ...)
    n_points: number of points used to approximate the curve
    flip_z  : mirror the sag about the vertex (z -> -z), for surfaces
              curving toward -z instead of +z
    """
    vertex = np.array(vertex, dtype=float)
    r = np.linspace(0.0, r_max, n_points)
    z = conic_sag(r, R, k, coeffs)
    if flip_z:
        z = -z
    points = np.column_stack([vertex[0] + z, vertex[1] + r])
    return points


def build_freeform_profile(points):
    """
    Wrap an arbitrary user-supplied list of [z, r] points as a profile
    array, for fully free-form surfaces not described by any conic
    formula (e.g. digitized from a scan, or hand-designed point by
    point). Points are used exactly as given, in order - no smoothing.
    """
    return np.array(points, dtype=float)


def profile_to_surfaces(points, kind="refract", material_in=None, material_out=None,
                         outward_direction=(1.0, 0.0), name="surface"):
    """
    Convert a [z, r] point array (from build_conic_profile,
    build_freeform_profile, or any other source) into a list of
    opentir Surface objects (a chain of Segments), ready to add to an
    OpticalSystem.

    outward_direction: reference unit vector used to decide which side
        is `material_out` - whichever side each segment's own geometric
        normal is closer to. For a typical exit surface facing away
        from the source in +z, the default (1,0) is usually right;
        use (-1,0) for a surface whose 'outer' (material_out) side
        faces -z instead.
    """
    points = np.array(points, dtype=float)
    outward_direction = np.array(outward_direction, dtype=float)
    outward_direction = outward_direction / np.linalg.norm(outward_direction)

    surfaces = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        d = p2 - p1
        if np.linalg.norm(d) < 1e-12:
            continue
        seg = Segment(p1, p2, name=name)
        geom_normal = np.array([-d[1], d[0]])
        geom_normal = geom_normal / np.linalg.norm(geom_normal)
        same_side = np.dot(geom_normal, outward_direction) > 0
        m_out = material_out if same_side else material_in
        m_in = material_in if same_side else material_out
        surfaces.append(Surface(seg, kind=kind, name=name,
                                 material_in=m_in, material_out=m_out))
    return surfaces
