"""
Example - Release 0.2
======================
LED chip encapsulated in PMMA under a FLAT exit window. This is the
classic "escape cone" problem in LED/photonic encapsulation: with a
flat interface, only rays within the critical angle from the normal
can exit; everything beyond it undergoes total internal reflection
(TIR) and is lost (unless recycled by the package). It is the reason
domed (spherical) encapsulation, with the chip near the center of
curvature, is preferred in real LED optics.

Critical angle for PMMA (n=1.49) -> air (n=1.0):
    theta_c = asin(1/1.49) ~= 42.1 deg

Run:
    python examples/example_flat_window_tir.py
"""

import numpy as np
import matplotlib.pyplot as plt

from opentir import (
    Segment,
    Surface, OpticalSystem,
    LEDSource,
    PMMA, AIR,
    critical_angle,
    plot_system, plot_illuminance,
)


def main():
    z_window = 5.0
    r_window = 20.0

    system = OpticalSystem()

    # Flat PMMA -> air interface. Normal points along +z (toward 'out').
    window = Segment([z_window, 0], [z_window, r_window], name="flat_window")
    system.add(Surface(window, kind="refract", name="flat_window",
                        material_in=PMMA, material_out=AIR))

    # Opaque side wall of the package (keeps rays that TIR back down
    # from escaping sideways to infinity, more representative of a
    # real encapsulated package)
    side_wall = Segment([0, r_window], [z_window, r_window], name="side_wall")
    system.add(Surface(side_wall, kind="block", name="side_wall"))

    # target plane far downstream, in air
    z_target = 150.0
    target = Segment([z_target, 0], [z_target, 80], name="target_plane")
    system.add(Surface(target, kind="target", name="target_plane"))

    # LED chip encapsulated in PMMA, Lambertian emission
    source = LEDSource(position=[0.0, 0.0], axis_deg=90.0,
                        half_angle_deg=89.0, n_rays=121,
                        distribution="lambertian", medium=PMMA)
    rays = source.generate_rays()

    traces = system.trace_many(rays, max_bounces=15, min_power=1e-3)

    theta_c = np.degrees(critical_angle(PMMA.n, AIR.n))
    total_in = sum(r.power for r in rays)
    total_hit = sum(t["power"] for t in traces if t["hits"])
    print(f"Critical angle PMMA -> air: {theta_c:.1f} deg")
    print(f"Rays traced: {len(rays)}  |  Branches after Fresnel/TIR splitting: {len(traces)}")
    print(f"Total emitted power: {total_in:.2f}  |  Power reaching target: {total_hit:.2f} "
          f"({100 * total_hit / total_in:.1f}%)  <- most is lost to TIR beyond {theta_c:.1f} deg")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_system(system, traces, ax=axes[0],
                title="OpenTIR 0.2 - flat PMMA window: Fresnel + TIR escape cone")
    plot_illuminance(traces, target_name="target_plane", ax=axes[1])
    plt.tight_layout()
    plt.savefig("example_flat_window_tir.png", dpi=150)
    print("Saved plot to example_flat_window_tir.png")


if __name__ == "__main__":
    main()
