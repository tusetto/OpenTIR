"""
opentir.source
~~~~~~~~~~~~~~
LED source models — release 0.6.2.

Supports three Light Emitting Surface (LES) geometries:
  'point'   – classic point source at a single (z, r) position
  'square'  – rectangular die of side `les_size` mm; sampled on a
               regular n_les × n_les grid of sub-sources
  'circle'  – circular die of diameter `les_size` mm; sub-sources
               placed on concentric rings (Sunflower / Vogel spiral)

Each sub-source emits the same angular cone with the same angular
distribution (Lambertian or uniform). The total emitted power is
normalised to 1.0 regardless of the number of sub-sources so that
efficiency figures are always comparable.

In the 2D (z, r) meridian plane the die extends along the r axis;
z position is fixed at `position[0]` for all sub-sources.
The r-coordinate of each sub-source point becomes the emission origin
in the section plane.
"""

import numpy as np
from .optics import Ray
from .materials import AIR


class LEDSource:
    def __init__(self, position, axis_deg=90.0, half_angle_deg=60.0,
                 n_rays=41, distribution="lambertian", medium=None,
                 les_shape="point", les_size=0.0, n_les=5):
        """
        position       : [z, r] emission centre (typically r = 0, on axis)
        axis_deg       : emission axis direction in degrees from +z
                         (90 = straight toward +r)
        half_angle_deg : emission cone half-angle (degrees)
        n_rays         : angular rays per sub-source
        distribution   : 'uniform' or 'lambertian'
        medium         : emitting medium (default: air)
        les_shape      : 'point', 'square', or 'circle'
        les_size       : side length (square) or diameter (circle) in mm;
                         ignored for les_shape='point'
        n_les          : number of sub-source samples per side (square) or
                         total sub-source count (circle).  For 'point' this
                         is ignored.  Minimum 1.
        """
        self.position     = np.array(position, dtype=float)
        self.axis         = np.radians(axis_deg)
        self.half_angle   = np.radians(half_angle_deg)
        self.n_rays       = int(n_rays)
        self.distribution = distribution
        self.medium       = medium if medium is not None else AIR
        self.les_shape    = les_shape
        self.les_size     = float(les_size)
        self.n_les        = max(1, int(n_les))

    # ── sub-source positions ──────────────────────────────────────────────────

    def _sub_source_positions(self):
        """
        Return a list of r-offsets (mm) for the sub-sources in the
        meridian plane.  The z coordinate is always self.position[0].
        """
        shape = self.les_shape
        if shape == "point" or self.les_size <= 0:
            return [0.0]

        half = self.les_size / 2.0

        if shape == "square":
            # n_les points along r; the die extends from -half to +half
            n = max(1, self.n_les)
            if n == 1:
                return [0.0]
            return np.linspace(-half, half, n).tolist()

        elif shape == "circle":
            # Vogel (sunflower) spiral — good uniform coverage
            # total n_les points inside a circle of radius = half
            n = max(1, self.n_les)
            if n == 1:
                return [0.0]
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            r_offsets = []
            for i in range(n):
                r_norm = np.sqrt((i + 0.5) / n)   # radius in [0, 1]
                theta_i = i * golden_angle          # azimuthal angle
                # project onto the meridian plane: use the x-component
                r_offsets.append(half * r_norm * np.cos(theta_i))
            return r_offsets

        else:
            raise ValueError(f"Unknown les_shape: '{shape}'. "
                             "Use 'point', 'square', or 'circle'.")

    # ── ray generation ────────────────────────────────────────────────────────

    def _angular_samples(self):
        """Return (angles, weights) arrays for one sub-source."""
        n = self.n_rays
        if self.distribution == "uniform":
            angles  = np.linspace(-self.half_angle, self.half_angle, n)
            weights = np.ones(n)
        elif self.distribution == "lambertian":
            s_max  = np.sin(self.half_angle)
            s      = np.linspace(-s_max, s_max, n)
            angles  = np.arcsin(np.clip(s, -1.0, 1.0))
            weights = np.cos(angles)
            weights = np.maximum(weights, 0.0)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        return angles, weights

    def generate_rays(self):
        """
        Generate all rays for the full LES.

        Total emitted power is normalised to 1.0 so efficiency figures
        are always in [0, 1] regardless of n_rays or n_les.
        """
        r_offsets         = self._sub_source_positions()
        angles, weights   = self._angular_samples()
        n_sub             = len(r_offsets)
        total_weight      = weights.sum() * n_sub  # for normalisation

        rays = []
        z0   = self.position[0]
        r0   = self.position[1]

        for r_off in r_offsets:
            origin = np.array([z0, r0 + r_off], dtype=float)
            for a, w in zip(angles, weights):
                theta     = self.axis + a
                direction = np.array([np.cos(theta), np.sin(theta)])
                power     = w / total_weight if total_weight > 0 else 1.0
                rays.append(Ray(origin, direction,
                                power=power, medium=self.medium))

        return rays

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def total_rays(self):
        """Total number of rays that will be generated."""
        return len(self._sub_source_positions()) * self.n_rays

    def les_description(self):
        if self.les_shape == "point" or self.les_size <= 0:
            return "puntiforme"
        elif self.les_shape == "square":
            return f"quadrata {self.les_size:.2f}×{self.les_size:.2f} mm"
        elif self.les_shape == "circle":
            return f"circolare ⌀{self.les_size:.2f} mm"
        return self.les_shape
