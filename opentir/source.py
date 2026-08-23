"""
opentir.source
~~~~~~~~~~~~~~
Simple LED source models for release 0.1.

A source emits a bundle of Ray objects in the (z, r) cross-section
plane, consistent with the axisymmetric convention used throughout
OpenTIR (the full 3D behaviour is recovered by revolving the traced
section around the optical z axis).
"""

import numpy as np
from .optics import Ray
from .materials import AIR


class LEDSource:
    def __init__(self, position, axis_deg=90.0, half_angle_deg=60.0,
                 n_rays=41, distribution="lambertian", medium=None):
        """
        position       : [z, r] emission point (typically r = 0, on axis)
        axis_deg       : direction of the emission axis, in degrees,
                         measured from +z (90 = straight toward +r)
        half_angle_deg : half-angle of the emission cone, in degrees
        n_rays         : number of rays to generate
        distribution   : 'uniform' (equal angular spacing) or
                         'lambertian' (cosine-weighted emission)
        medium         : Material the source emits into (default: air).
                         Use a glass/PMMA material to model a chip
                         encapsulated directly in the optic (release 0.2).
        """
        self.position = np.array(position, dtype=float)
        self.axis = np.radians(axis_deg)
        self.half_angle = np.radians(half_angle_deg)
        self.n_rays = n_rays
        self.distribution = distribution
        self.medium = medium if medium is not None else AIR

    def generate_rays(self):
        if self.distribution == "uniform":
            angles = np.linspace(-self.half_angle, self.half_angle, self.n_rays)
        elif self.distribution == "lambertian":
            s_max = np.sin(self.half_angle)
            s = np.linspace(-s_max, s_max, self.n_rays)
            angles = np.arcsin(np.clip(s, -1, 1))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        rays = []
        for a in angles:
            theta = self.axis + a
            direction = np.array([np.cos(theta), np.sin(theta)])
            power = np.cos(a) if self.distribution == "lambertian" else 1.0
            rays.append(Ray(self.position, direction, power=max(power, 0), medium=self.medium))
        return rays
