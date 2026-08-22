"""
opentir.sms
~~~~~~~~~~~~
Simultaneous Multiple Surface (SMS) synthesis - release 0.3.

Implements the classic 2D "SMS chain" marching algorithm for TWO
refractive surfaces (S1, S2) that simultaneously couple TWO point
sources P1, P2 to two prescribed output directions d1, d2. This is
the standard formulation used for LED collimator design, where P1/P2
represent the two edges of an extended emitter and the goal is to
collimate both edges into (typically) the same output direction -
achieving much better control over an extended source's etendue than
a single-surface (Cartesian oval) design built for a single point.

Simplifications in this first implementation
----------------------------------------------
- Meridian (2D) construction only: no full 3D free-form / skew rays.
- The seed (starting point of the marching chain) uses a flat initial
  surface normal at a user-chosen axial position. This is a common
  simplification; more rigorous seed constructions exist in the SMS
  literature (e.g. seed ribs derived from a local spherical or
  Cartesian-oval approximation) and may be added later.
- As in the wider SMS literature, numerical robustness is not
  guaranteed for arbitrarily large apertures or angular ranges: the
  marching can break down (TIR, negative discriminant, non-physical
  step) before reaching the requested number of points. When this
  happens the chain is truncated and a warning message is returned
  rather than raising an exception.

Reference: J.C. Minano, P. Benitez et al., "SMS design method", as
presented in Winston / Minano / Benitez, "Nonimaging Optics" (2005).
"""

import numpy as np

from .optics import snell_refract


def _opl_plane_phase(point, direction, n_medium, s0=0.0):
    """Optical path length from `point` to the reference plane
    {x : dot(x, direction) = s0}, travelling along `direction`."""
    return n_medium * (s0 - np.dot(point, direction))


class SMSChainResult:
    def __init__(self, s1_points, s2_points, s1_normals, s2_normals, warning=None):
        self.s1_points = np.array(s1_points)
        self.s2_points = np.array(s2_points)
        self.s1_normals = np.array(s1_normals)
        self.s2_normals = np.array(s2_normals)
        self.warning = warning


def design_sms_collimator(p1, p2, n_source_medium, n_lens, n_out=1.0,
                           d1=(1.0, 0.0), d2=(1.0, 0.0),
                           seed_z=5.0, seed_thickness=3.0,
                           n_steps=25, s0=0.0):
    """
    Run the SMS chain marching algorithm.

    p1, p2          : [z, r] positions of the two point sources (e.g.
                       the two edges of an extended LED emitter)
    n_source_medium : refractive index of the medium the sources emit into
    n_lens          : refractive index of the lens material (between S1, S2)
    n_out           : refractive index of the output medium (air = 1.0)
    d1, d2          : desired OUTPUT directions for bundle 1 / bundle 2
                       (unit vectors; default: both collimated along +z)
    seed_z          : axial position of the first S1 seed point
    seed_thickness  : assumed internal path length, used only to fix the
                       OPL constants L1, L2 at the seed (a free design
                       choice, roughly sets the lens thickness at the axis)
    n_steps         : number of alternating marching steps to attempt
    s0              : reference-plane offset (does not affect the
                       resulting shape, only an internal constant)

    Returns an SMSChainResult with s1_points, s2_points, their SMS-
    derived normals, and an optional warning string if marching was
    truncated before n_steps was reached.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d1 = np.array(d1, dtype=float) / np.linalg.norm(d1)
    d2 = np.array(d2, dtype=float) / np.linalg.norm(d2)
    n1_med = float(n_source_medium)
    n2_med = float(n_lens)
    n_out = float(n_out)

    # ---------------------------------------------------------------
    # Seed: flat initial normal at an axial point on S1
    # ---------------------------------------------------------------
    A0 = np.array([seed_z, 0.0])
    N_A0 = np.array([1.0, 0.0])

    d_in2 = (A0 - p2) / np.linalg.norm(A0 - p2)
    res2 = snell_refract(d_in2, N_A0, n1_med, n2_med)
    if res2 is None:
        return SMSChainResult([], [], [], [], warning="TIR al seed (bundle 2): scegli un seed_z diverso")
    r_after2, _, _ = res2
    B0 = A0 + seed_thickness * r_after2
    L2 = (n1_med * np.linalg.norm(A0 - p2) + n2_med * seed_thickness
          + _opl_plane_phase(B0, d2, n_out, s0))

    d_in1 = (A0 - p1) / np.linalg.norm(A0 - p1)
    res1 = snell_refract(d_in1, N_A0, n1_med, n2_med)
    if res1 is None:
        return SMSChainResult([], [], [], [], warning="TIR al seed (bundle 1): scegli un seed_z diverso")
    r_after1, _, _ = res1
    B0_virtual = A0 + seed_thickness * r_after1
    L1 = (n1_med * np.linalg.norm(A0 - p1) + n2_med * seed_thickness
          + _opl_plane_phase(B0_virtual, d1, n_out, s0))

    N_B0 = n_out * d2 - n2_med * r_after2
    N_B0 = N_B0 / np.linalg.norm(N_B0)

    s1_points, s1_normals = [A0], [N_A0]
    s2_points, s2_normals = [B0], [N_B0]
    A_k, N_Ak = A0, N_A0
    B_k, N_Bk = B0, N_B0
    warning = None

    for step in range(n_steps):
        # ---- next point on S1, from B_k using bundle 1 (quadratic) ----
        res = snell_refract(-d1, N_Bk, n_out, n2_med)
        if res is None:
            warning = f"Marcia interrotta al passo {step}: TIR su S1"
            break
        r_before = -res[0]  # direction from A_{k+1} towards B_k

        V = B_k - p1
        K = L1 - _opl_plane_phase(B_k, d1, n_out, s0)
        a = n1_med ** 2 - n2_med ** 2
        b = 2 * (K * n2_med - n1_med ** 2 * np.dot(V, r_before))
        c = n1_med ** 2 * np.dot(V, V) - K ** 2

        s_val = None
        if abs(a) < 1e-9:
            if abs(b) > 1e-12:
                candidate = -c / b
                if candidate > 1e-6 and (K - n2_med * candidate) >= -1e-6:
                    s_val = candidate
        else:
            disc = b ** 2 - 4 * a * c
            if disc >= 0:
                sq = np.sqrt(disc)
                candidates = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
                # Squaring the original (unsquared) OPL equation can
                # introduce an extraneous root; keep only roots that
                # satisfy the ORIGINAL sign condition n1*|..| = K - n2*s
                # (i.e. K - n2*s >= 0), and prefer the smallest valid one.
                valid = sorted(s for s in candidates
                                if s > 1e-6 and (K - n2_med * s) >= -1e-6)
                if valid:
                    s_val = valid[0]
        if s_val is None or s_val <= 0:
            warning = f"Marcia interrotta al passo {step}: nessuna soluzione fisica per S1"
            break

        A_next = B_k - s_val * r_before
        d_in1_next = (A_next - p1) / np.linalg.norm(A_next - p1)
        N_next = n2_med * r_before - n1_med * d_in1_next
        if np.linalg.norm(N_next) < 1e-9:
            warning = f"Marcia interrotta al passo {step}: normale S1 degenere"
            break
        N_next = N_next / np.linalg.norm(N_next)
        s1_points.append(A_next)
        s1_normals.append(N_next)
        A_k, N_Ak = A_next, N_next

        # ---- next point on S2, from A_{k+1} using bundle 2 (linear) ----
        d_in2_next = (A_k - p2) / np.linalg.norm(A_k - p2)
        res2n = snell_refract(d_in2_next, N_Ak, n1_med, n2_med)
        if res2n is None:
            warning = f"Marcia interrotta al passo {step}: TIR su S2"
            break
        r_after2n, _, _ = res2n

        denom = n2_med - n_out * np.dot(r_after2n, d2)
        if abs(denom) < 1e-9:
            warning = f"Marcia interrotta al passo {step}: passo S2 non definito"
            break
        numer = (L2 - n1_med * np.linalg.norm(A_k - p2)
                 - n_out * s0 + n_out * np.dot(A_k, d2))
        t_next = numer / denom
        if t_next <= 1e-6:
            warning = f"Marcia interrotta al passo {step}: passo S2 non fisico"
            break

        B_next = A_k + t_next * r_after2n
        N_Bnext = n_out * d2 - n2_med * r_after2n
        if np.linalg.norm(N_Bnext) < 1e-9:
            warning = f"Marcia interrotta al passo {step}: normale S2 degenere"
            break
        N_Bnext = N_Bnext / np.linalg.norm(N_Bnext)
        s2_points.append(B_next)
        s2_normals.append(N_Bnext)
        B_k, N_Bk = B_next, N_Bnext

        if A_k[1] < 0 or B_k[1] < 0:
            warning = f"Marcia interrotta al passo {step}: punto uscito dal semipiano r>=0"
            break

    return SMSChainResult(s1_points, s2_points, s1_normals, s2_normals, warning=warning)


def build_sms_surfaces(result, n_source, n_lens, n_out=None, name_s1="SMS_S1", name_s2="SMS_S2"):
    """
    Convert an SMSChainResult into a pair of lists of opentir Surface
    objects (kind='refract'), ready to be added to an OpticalSystem
    and traced with the normal ray-tracing engine.

    Material sidedness (material_in / material_out) is assigned per
    segment by comparing the segment's own geometric normal against
    the SMS-derived normal at that point, so the result is correct
    regardless of the point ordering produced by the marching chain.
    """
    from .geometry import Segment
    from .optics import Surface

    if n_out is None:
        n_out = n_source

    def _build(points, normals, mat_before, mat_after, name):
        surfaces = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            seg = Segment(p1, p2, name=name)
            d = np.array(p2) - np.array(p1)
            if np.linalg.norm(d) < 1e-12:
                continue
            geom_normal = np.array([-d[1], d[0]])
            geom_normal = geom_normal / np.linalg.norm(geom_normal)
            ref_normal = np.array(normals[i]) + np.array(normals[i + 1])
            if np.linalg.norm(ref_normal) < 1e-12:
                ref_normal = np.array(normals[i])
            ref_normal = ref_normal / np.linalg.norm(ref_normal)
            # ref_normal points toward the "after" medium (see module docstring)
            same_side = np.dot(geom_normal, ref_normal) > 0
            material_out = mat_after if same_side else mat_before
            material_in = mat_before if same_side else mat_after
            surfaces.append(Surface(seg, kind="refract", name=name,
                                     material_in=material_in, material_out=material_out))
        return surfaces

    s1_surfaces = _build(result.s1_points, result.s1_normals, n_source, n_lens, name_s1)
    s2_surfaces = _build(result.s2_points, result.s2_normals, n_lens, n_out, name_s2)
    return s1_surfaces, s2_surfaces


# =======================================================================
# Single-surface Cartesian oval (analytic, exact) - the well-established
# building block behind SMS. Given ONE point source and a desired
# collimated (plane-wave) output through ONE refracting surface, the
# exact solution is a conic (hyperbola for n_lens > n_source) with the
# source at its real focus. Closed-form, no marching, no seed problem -
# provided here as a reliable, independently verified release-0.3
# deliverable while the two-surface chain above is refined further.
# =======================================================================


def design_cartesian_oval_collimator(n_source, n_lens, vertex_distance,
                                      theta_max_deg, n_points=60):
    """
    Exact single-surface refractive profile that collimates a point
    source at the origin (emitting into `n_source`) into a beam
    parallel to +z after refracting into `n_lens`.

    vertex_distance : distance from the source to the surface apex
                       (on-axis), i.e. the lens' on-axis thickness
    theta_max_deg   : maximum emission half-angle (from the source)
                       to cover; must stay below the angle at which
                       the conic's denominator vanishes (physically,
                       the point past which no single convex surface
                       can collimate further)

    Returns an array of [z, r] points along the profile, ordered by
    increasing theta (i.e. increasing r).
    """
    n1, n2 = float(n_source), float(n_lens)
    theta_max = np.radians(theta_max_deg)

    theta_limit = np.arccos(min(n1 / n2, 0.999999)) if n2 > n1 else np.pi
    if theta_max >= theta_limit:
        raise ValueError(
            f"theta_max_deg troppo grande: un'unica superficie convessa non puo' "
            f"collimare oltre {np.degrees(theta_limit):.1f} deg per n_source={n1}, n_lens={n2}"
        )

    const = vertex_distance * (n1 - n2)  # fixes rho(0) = vertex_distance
    thetas = np.linspace(0.0, theta_max, n_points)
    rho = const / (n1 - n2 * np.cos(thetas))
    z = rho * np.cos(thetas)
    r = rho * np.sin(thetas)
    return np.column_stack([z, r])


def build_cartesian_oval_surface(points, material_source, material_lens, name="cartesian_oval"):
    """Build a list of opentir refract Surfaces from the profile points
    returned by design_cartesian_oval_collimator.

    material_source, material_lens: opentir Material instances (not
    raw floats) for the source-side and lens-side media.
    """
    from .geometry import Segment
    from .optics import Surface

    n_source, n_lens = material_source.n, material_lens.n
    surfaces = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        seg = Segment(p1, p2, name=name)
        d = np.array(p2) - np.array(p1)
        geom_normal = np.array([-d[1], d[0]])
        geom_normal = geom_normal / np.linalg.norm(geom_normal)
        midpoint = (np.array(p1) + np.array(p2)) / 2
        # the true surface normal at this conic point, from the OPL
        # gradient, points along n_lens*d_out - n_source*d_in; since
        # d_out=(1,0) and d_in = unit(midpoint) (source at origin):
        d_in = midpoint / np.linalg.norm(midpoint)
        ref_normal = n_lens * np.array([1.0, 0.0]) - n_source * d_in
        ref_normal = ref_normal / np.linalg.norm(ref_normal)
        same_side = np.dot(geom_normal, ref_normal) > 0
        material_out = material_lens if same_side else material_source
        material_in = material_source if same_side else material_lens
        surfaces.append(Surface(seg, kind="refract", name=name,
                                 material_in=material_in, material_out=material_out))
    return surfaces
