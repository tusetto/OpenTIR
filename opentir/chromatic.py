"""
opentir.chromatic
~~~~~~~~~~~~~~~~~~
Tools for chromatic aberration simulation.

Provides:
- `wavelength_samples(n)` : generates n wavelengths uniformly spaced
  across the visible spectrum (380–720 nm)
- `wavelength_to_rgb(nm)` : converts a wavelength to an approximate
  sRGB color for plotting (based on the classic Bruton/Cie cie mapping)
- `chromatic_rays(source, wavelengths)` : expands a list of base rays
  into one copy per wavelength, each carrying its own wavelength_nm
"""

import numpy as np


# Visible range used throughout OpenTIR
VIS_MIN_NM = 380.0
VIS_MAX_NM = 720.0


def wavelength_samples(n):
    """
    Return n wavelengths (nm) uniformly spaced across the visible
    spectrum (380–720 nm).  n=1 returns only the reference d-line
    (589.3 nm), same as the non-chromatic mode.
    """
    if n <= 1:
        return [589.3]
    return list(np.linspace(VIS_MIN_NM, VIS_MAX_NM, n))


def wavelength_to_rgb(nm):
    """
    Convert a visible-light wavelength (nm) to an approximate sRGB
    tuple (r, g, b) in [0, 1].

    Based on the algorithm by Dan Bruton (physics.sfasu.edu/astro/color)
    with gamma correction.  Not colorimetrically exact, but visually
    accurate enough for ray-plot coloring.
    """
    nm = float(nm)
    if nm < 380 or nm > 720:
        return (0.5, 0.5, 0.5)  # outside visible: gray

    if 380 <= nm < 440:
        r = -(nm - 440) / 60.0
        g = 0.0
        b = 1.0
    elif 440 <= nm < 490:
        r = 0.0
        g = (nm - 440) / 50.0
        b = 1.0
    elif 490 <= nm < 510:
        r = 0.0
        g = 1.0
        b = -(nm - 510) / 20.0
    elif 510 <= nm < 580:
        r = (nm - 510) / 70.0
        g = 1.0
        b = 0.0
    elif 580 <= nm < 645:
        r = 1.0
        g = -(nm - 645) / 65.0
        b = 0.0
    else:  # 645–720
        r = 1.0
        g = 0.0
        b = 0.0

    # intensity attenuation at the edges of the visible range
    if 380 <= nm < 420:
        factor = 0.3 + 0.7 * (nm - 380) / 40.0
    elif 700 < nm <= 720:
        factor = 0.3 + 0.7 * (720 - nm) / 20.0
    else:
        factor = 1.0

    gamma = 0.80
    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma
    return (float(r), float(g), float(b))


def chromatic_rays(base_rays, wavelengths):
    """
    Expand a list of base Rays into one copy per wavelength.

    base_rays   : list of Ray objects (from LEDSource.generate_rays())
    wavelengths : list of wavelengths in nm

    Returns a list of Ray objects, each with its own wavelength_nm attribute.
    The power is divided equally among the wavelengths.
    """
    from .optics import Ray
    result = []
    n_wl = len(wavelengths)
    for wl in wavelengths:
        for ray in base_rays:
            new_ray = Ray(
                origin=ray.origin.copy(),
                direction=ray.direction.copy(),
                power=ray.power / n_wl,  # divide power among wavelengths
                medium=ray.medium,
                wavelength_nm=wl,
            )
            result.append(new_ray)
    return result
