"""
opentir.materials
~~~~~~~~~~~~~~~~~~
Optical materials with a constant (non-dispersive) refractive index.
Dispersion (n as a function of wavelength) is out of scope for release
0.2 and may be added later.
"""


class Material:
    def __init__(self, n=1.0, name="material"):
        self.n = float(n)
        self.name = name

    def __repr__(self):
        return f"Material({self.name}, n={self.n})"


# Common materials used in LED illumination optics
AIR = Material(1.0, "air")
PMMA = Material(1.49, "PMMA")
POLYCARBONATE = Material(1.585, "polycarbonate")
BK7_GLASS = Material(1.517, "BK7 glass")
SODA_LIME_GLASS = Material(1.52, "soda-lime glass")
