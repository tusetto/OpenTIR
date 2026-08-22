"""
opentir.materials
~~~~~~~~~~~~~~~~~~
Optical materials, with an optional dispersion model for chromatic
aberration simulation.

`n` is the refractive index at the reference "d-line" (589.3 nm,
helium), the standard reference wavelength in optics. If an Abbe
number (`abbe`, Vd) is given, wavelength-dependent index n(lambda) is
computed via a 2-term Cauchy model:

    n(lambda) = A + B / lambda^2

calibrated so that n(589.3 nm) = n exactly, and the dispersion
(n_F - n_C) between the hydrogen F (486.1 nm) and C (656.3 nm) lines
matches the given Abbe number: Vd = (n - 1) / (n_F - n_C). This is a
standard, widely used approximation good enough for visualizing
chromatic aberration; it is NOT a substitute for manufacturer-supplied
Sellmeier coefficients in precision lens design.

If no Abbe number is given (custom materials, or media like air where
dispersion is negligible for this purpose), n(lambda) is just the
constant `n` - no chromatic splitting effect from that material.
"""

_LAMBDA_D_UM = 0.5893  # helium d-line (reference), micrometers
_LAMBDA_F_UM = 0.4861  # hydrogen F-line (blue)
_LAMBDA_C_UM = 0.6563  # hydrogen C-line (red)


def _cauchy_from_nd_vd(nd, vd):
    B = (nd - 1.0) / (vd * (1.0 / _LAMBDA_F_UM ** 2 - 1.0 / _LAMBDA_C_UM ** 2))
    A = nd - B / _LAMBDA_D_UM ** 2
    return A, B


class Material:
    def __init__(self, n=1.0, name="material", abbe=None):
        self.n = float(n)  # index at the d-line (589.3 nm)
        self.name = name
        self.abbe = abbe  # Abbe number Vd; None = no dispersion modeled
        if abbe is not None:
            self._cauchy_A, self._cauchy_B = _cauchy_from_nd_vd(self.n, abbe)
        else:
            self._cauchy_A, self._cauchy_B = self.n, 0.0

    def n_at(self, wavelength_nm):
        """Refractive index at the given wavelength (nm)."""
        if self.abbe is None:
            return self.n
        lam_um = wavelength_nm / 1000.0
        return self._cauchy_A + self._cauchy_B / lam_um ** 2

    def __repr__(self):
        if self.abbe is not None:
            return f"Material({self.name}, n={self.n}, Vd={self.abbe})"
        return f"Material({self.name}, n={self.n})"


# Common materials used in LED illumination optics.
# Abbe numbers are typical published approximations (manufacturer
# datasheets vary somewhat by grade/supplier) - good for a realistic
# qualitative/comparative chromatic aberration simulation, not for
# certifying a real optical design.
AIR = Material(1.0, "air")  # dispersion negligible for this purpose
PMMA = Material(1.49, "PMMA", abbe=57.4)
POLYCARBONATE = Material(1.585, "polycarbonate", abbe=30.0)
BK7_GLASS = Material(1.517, "BK7 glass", abbe=64.17)
SODA_LIME_GLASS = Material(1.52, "soda-lime glass", abbe=58.5)
