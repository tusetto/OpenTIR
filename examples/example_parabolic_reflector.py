"""
Example - Release 0.1
======================
LED point source at the focus of a parabolic mirror, plus a flat target
plane, traced with pure geometric ray tracing (no refraction yet).

Run:
    python examples/example_parabolic_reflector.py
"""

import numpy as np
import matplotlib.pyplot as plt

from opentir import (
    Segment,
    Profile,
    Surface,
    OpticalSystem,
    LEDSource,
    plot_system,
    plot_illuminance,
)


def build_parabolic_mirror(focal_length=10.0, aperture_radius=25.0, n=60):
    """
    Parabola with focus at the origin, axis along +z:
        z = r^2 / (4f) - f
    Sampled as a polyline (Segments) for release 0.1.
    """
    r = np.linspace(0, aperture_radius, n)
    z = r ** 2 / (4 * focal_length) - focal_length
    elements = []
    for i in range(n - 1):
        elements.append(Segment([z[i], r[i]], [z[i + 1], r[i + 1]], name="parabolic_mirror"))
    return Profile(elements, name="parabolic_mirror")


def main():
    focal_length = 10.0
    profile = build_parabolic_mirror(focal_length=focal_length, aperture_radius=25.0)

    system = OpticalSystem()
    for el in profile.elements:
        system.add(Surface(el, kind="mirror", name="parabolic_mirror"))

    # flat target plane far downstream, perpendicular to the axis
    z_target = 200.0
    target = Segment([z_target, 0], [z_target, 30], name="target_plane")
    system.add(Surface(target, kind="target", name="target_plane"))

    # LED at the focus, wide emission cone
    source = LEDSource(position=[0.0, 0.0], axis_deg=90.0,
                        half_angle_deg=80.0, n_rays=61,
                        distribution="lambertian")
    rays = source.generate_rays()

    traces = system.trace_many(rays)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_system(system, traces, ax=axes[0])
    plot_illuminance(traces, target_name="target_plane", ax=axes[1])
    plt.tight_layout()
    plt.savefig("example_parabolic_reflector.png", dpi=150)
    print("Saved plot to example_parabolic_reflector.png")


if __name__ == "__main__":
    main()
