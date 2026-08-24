"""
opentir.materials
~~~~~~~~~~~~~~~~~~
Optical materials with refractive index that can be either constant
(non-dispersive) or wavelength-dependent (dispersive).

For dispersive materials, the Sellmeier equation is used to compute
the refractive index as a function of wavelength.
"""


class Material:
    def __init__(self, n=1.0, name="material", sellmeier=None):
        self.n = float(n)
        self.name = name
        # sellmeier: list of (B, C) tuples for Sellmeier equation
        # n^2(lambda) = 1 + sum(B_i * lambda^2 / (lambda^2 - C_i))
        # lambda in micrometers
        self.sellmeier = sellmeier

    def n_at(self, wavelength_nm):
        """Return refractive index at given wavelength (nm).
        
        If sellmeier coefficients are defined, uses Sellmeier equation.
        Otherwise returns constant n.
        """
        if self.sellmeier is not None:
            lambda_um = wavelength_nm / 1000.0  # convert nm to micrometers
            n_sq = 1.0
            for B, C in self.sellmeier:
                n_sq += B * lambda_um**2 / (lambda_um**2 - C)
            return n_sq ** 0.5
        return self.n

    def __repr__(self):
        return f"Material({self.name}, n={self.n})"


# Common materials used in LED illumination optics
# Sellmeier coefficients from various sources (B, C) tuples
# For PMMA: https://refractiveindex.info/?shelf=organic&book=polymethyl_methacrylate&page=Sultanova
PMMA = Material(1.49, "PMMA", sellmeier=[
    (0.59382, 0.00796),
    (0.28872, 0.05076),
])

# For Polycarbonate: https://refractiveindex.info/?shelf=organic&book=polycarbonate&page=Sultanova
POLYCARBONATE = Material(1.585, "polycarbonate", sellmeier=[
    (1.43107, 0.01043),
    (0.04697, 0.06873),
])

# For BK7 glass: standard Sellmeier coefficients
BK7_GLASS = Material(1.517, "BK7 glass", sellmeier=[
    (1.03961212, 0.00600069867),
    (0.231792344, 0.0200179144),
    (1.01046945, 103.560653),
])

# For Soda-lime glass: approximate Sellmeier coefficients
SODA_LIME_GLASS = Material(1.52, "soda-lime glass", sellmeier=[
    (0.87328, 0.00676),
    (0.30467, 0.02573),
])

AIR = Material(1.0, "air")
