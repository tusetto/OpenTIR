"""
opentir.optics
~~~~~~~~~~~~~~
Ray definition and geometric/physical ray tracing engine.

Release 0.1: pure geometry - mirrors (specular reflection) and
             targets/blocks (absorption).
Release 0.2: adds refractive materials, Snell's law, Fresnel
             reflectance and total internal reflection (TIR). A ray
             hitting a 'refract' surface splits into a reflected and
             a transmitted branch, weighted by the Fresnel
             reflectance, unless TIR occurs (100% reflected).
Chromatic aberration: each Ray carries a `wavelength_nm` (defaulting
             to the 589.3 nm reference "d-line", at which every
             Material's `.n` is calibrated exactly). Refraction always
             looks up each material's index at the ray's own
             wavelength via `Material.n_at()`, so a dispersive
             material (one with an Abbe number set) naturally bends
             different wavelengths differently - see opentir.chromatic
             for splitting a beam into multiple wavelengths and for
             wavelength -> RGB color mapping.
"""

import numpy as np

from .materials import AIR

REFERENCE_WAVELENGTH_NM = 589.3  # helium d-line; matches Material.n exactly

# ---------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------


class Ray:
    def __init__(self, origin, direction, power=1.0, medium=None,
                 wavelength_nm=REFERENCE_WAVELENGTH_NM):
        self.origin = np.array(origin, dtype=float)
        d = np.array(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        self.power = power
        self.medium = medium if medium is not None else AIR
        self.wavelength_nm = wavelength_nm


# ---------------------------------------------------------------------
# Reflection / refraction / Fresnel
# ---------------------------------------------------------------------


def reflect(direction, normal):
    """Law of reflection: d' = d - 2(d.n)n"""
    n = normal / np.linalg.norm(normal)
    return direction - 2 * np.dot(direction, n) * n


def snell_refract(direction, normal, n1, n2):
    """
    Vector form of Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t).

    `normal` is automatically flipped so that it points against the
    incident ray (into medium 1, the medium the ray is coming from).

    Returns (refracted_direction, cos_theta_i, cos_theta_t), or None
    if the incidence angle exceeds the critical angle (total internal
    reflection).
    """
    d = direction / np.linalg.norm(direction)
    n = normal / np.linalg.norm(normal)
    cos_i = -np.dot(d, n)
    if cos_i < 0:
        n = -n
        cos_i = -cos_i
    eta = n1 / n2
    sin2_t = eta ** 2 * max(0.0, 1.0 - cos_i ** 2)
    if sin2_t > 1.0:
        return None  # total internal reflection
    cos_t = np.sqrt(1.0 - sin2_t)
    d_t = eta * d + (eta * cos_i - cos_t) * n
    return d_t, cos_i, cos_t


def fresnel_reflectance(cos_i, cos_t, n1, n2):
    """Unpolarized Fresnel reflectance R = 0.5*(Rs + Rp)."""
    Rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    Rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    return 0.5 * (Rs + Rp)


def critical_angle(n1, n2):
    """Critical angle (rad) for a ray going from n1 to n2, or None if n1 <= n2."""
    if n1 <= n2:
        return None
    return np.arcsin(n2 / n1)


# ---------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------


class Surface:
    """
    A physical surface in the optical system.

    geometry : Segment or Arc
    kind     : 'mirror'  -> specular reflection
               'target'  -> ray absorbed here, hit point recorded
               'block'   -> ray absorbed, no data recorded (opaque baffle)
               'refract' -> Snell/Fresnel refraction + TIR (release 0.2),
                            with per-wavelength dispersion if the
                            material has an Abbe number set

    For kind='refract', the surface geometry's normal is used as the
    reference orientation:
      - material_out : medium on the side the normal points TOWARD
      - material_in  : medium on the side the normal points AWAY from
    """

    def __init__(self, geometry, kind="mirror", name=None,
                 material_in=None, material_out=None):
        self.geometry = geometry
        self.kind = kind
        self.name = name or geometry.name
        self.material_in = material_in
        self.material_out = material_out

    def intersect(self, ray):
        return self.geometry.intersect(ray)


# ---------------------------------------------------------------------
# Optical system / tracer
# ---------------------------------------------------------------------


class OpticalSystem:
    def __init__(self):
        self.surfaces = []

    def add(self, surface):
        self.surfaces.append(surface)
        return self

    def _closest_hit(self, ray):
        """Find the closest surface intersection ahead of the ray.
        No axis fold: rays cross r=0 freely (2D plane geometry)."""
        best = None
        best_surface = None
        for surf in self.surfaces:
            hit = surf.intersect(ray)
            if hit is None:
                continue
            t, point, normal = hit
            if best is None or t < best[0]:
                best = (t, point, normal)
                best_surface = surf
        return best, best_surface

    def trace_ray(self, ray, max_bounces=20, min_power=1e-3):
        """
        Trace a single ray through the system. Returns a LIST of trace
        dicts, one per terminated branch, each with:
          'path'          : list of [z, r] points along the branch
          'hits'          : list of (surface, point) on 'target' surfaces
          'power'         : final power of this branch
          'wavelength_nm' : wavelength carried by this branch
          'reflected'     : True if this branch is a Fresnel partial
                            reflection; False for transmitted / mirror /
                            TIR branches.

        Geometry note: the system uses a 2D plane model. Surfaces are
        defined for r >= 0 (upper half), but rays are free to cross
        r = 0 and continue into r < 0 without any artificial fold.
        For a rotationally-symmetric optic, define surfaces on both
        sides of the axis (r > 0 and r < 0 segments) or use the
        source positioned at r != 0 for off-axis simulation.
        """
        return self._trace(ray, [ray.origin.copy()], 0, max_bounces,
                           min_power, reflected=False)

    def _trace(self, ray, path, depth, max_bounces, min_power, reflected=False):
        if depth >= max_bounces or ray.power < min_power:
            endpoint = ray.origin + ray.direction * 1e2
            return [{"path": path + [endpoint], "hits": [], "power": ray.power,
                     "wavelength_nm": ray.wavelength_nm, "reflected": reflected,
                     "_color": getattr(ray, "_color", None)}]

        best, surface = self._closest_hit(ray)
        if best is None:
            endpoint = ray.origin + ray.direction * 1e3
            return [{"path": path + [endpoint], "hits": [], "power": ray.power,
                     "wavelength_nm": ray.wavelength_nm, "reflected": reflected,
                     "_color": getattr(ray, "_color", None)}]

        t, point, normal = best
        new_path = path + [point]

        if surface.kind == "mirror":
            new_dir = reflect(ray.direction, normal)
            new_ray = Ray(point, new_dir, ray.power, medium=ray.medium,
                           wavelength_nm=ray.wavelength_nm)
            if hasattr(ray, "_color"):
                new_ray._color = ray._color
            return self._trace(new_ray, new_path, depth + 1, max_bounces,
                                min_power, reflected=reflected)

        elif surface.kind == "target":
            return [{"path": new_path, "hits": [(surface, point)],
                     "power": ray.power, "wavelength_nm": ray.wavelength_nm,
                     "reflected": reflected, "_color": getattr(ray, "_color", None)}]

        elif surface.kind == "block":
            return [{"path": new_path, "hits": [], "power": ray.power,
                     "wavelength_nm": ray.wavelength_nm, "reflected": reflected,
                     "_color": getattr(ray, "_color", None)}]

        elif surface.kind == "refract":
            return self._trace_refract(ray, surface, point, normal, new_path,
                                        depth, max_bounces, min_power, reflected)

        else:
            return [{"path": new_path, "hits": [], "power": ray.power,
                     "wavelength_nm": ray.wavelength_nm, "reflected": reflected,
                     "_color": getattr(ray, "_color", None)}]

    def _trace_refract(self, ray, surface, point, normal, new_path,
                        depth, max_bounces, min_power, reflected=False):
        d = ray.direction
        cos_i_raw = -np.dot(d, normal)
        if cos_i_raw >= 0:
            mat_from, mat_to = surface.material_out, surface.material_in
            n_use = normal
        else:
            mat_from, mat_to = surface.material_in, surface.material_out
            n_use = -normal

        n_from = mat_from.n_at(ray.wavelength_nm)
        n_to   = mat_to.n_at(ray.wavelength_nm)

        result = snell_refract(d, n_use, n_from, n_to)

        branches = []
        if result is None:
            # TIR: 100% reflected — keep 'reflected=False' (it IS the main beam)
            refl_dir = reflect(d, n_use)
            refl_ray = Ray(point, refl_dir, ray.power, medium=mat_from,
                            wavelength_nm=ray.wavelength_nm)
            if hasattr(ray, "_color"):
                refl_ray._color = ray._color
            branches = self._trace(refl_ray, new_path, depth + 1, max_bounces,
                                    min_power, reflected=False)
        else:
            d_t, cos_i, cos_t = result
            R = fresnel_reflectance(cos_i, cos_t, n_from, n_to)
            refl_power  = ray.power * R
            trans_power = ray.power * (1.0 - R)

            if refl_power >= min_power and depth + 1 < max_bounces:
                refl_dir = reflect(d, n_use)
                refl_ray = Ray(point, refl_dir, refl_power, medium=mat_from,
                                wavelength_nm=ray.wavelength_nm)
                if hasattr(ray, "_color"):
                    refl_ray._color = ray._color
                # Fresnel partial reflection → mark as reflected=True
                branches += self._trace(refl_ray, new_path, depth + 1,
                                         max_bounces, min_power, reflected=True)

            if trans_power >= min_power and depth + 1 < max_bounces:
                trans_ray = Ray(point, d_t, trans_power, medium=mat_to,
                                 wavelength_nm=ray.wavelength_nm)
                if hasattr(ray, "_color"):
                    trans_ray._color = ray._color
                # transmitted branch inherits the parent's reflected status
                branches += self._trace(trans_ray, new_path, depth + 1,
                                         max_bounces, min_power, reflected=reflected)

            if not branches:
                branches = [{"path": new_path, "hits": [], "power": ray.power,
                             "wavelength_nm": ray.wavelength_nm, "reflected": reflected,
                             "_color": getattr(ray, "_color", None)}]

        return branches

    def trace_many(self, rays, max_bounces=20, min_power=1e-3):
        all_traces = []
        for r in rays:
            all_traces.extend(self.trace_ray(r, max_bounces=max_bounces, min_power=min_power))
        return all_traces
