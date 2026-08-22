"""
Example - Release 0.3
======================
Exact single-surface Cartesian oval collimating lens: a point LED
source at the origin, refracting through ONE PMMA surface, exits
exactly parallel to the optical axis for every emission angle within
the design range. This is the closed-form building block behind the
full SMS (Simultaneous Multiple Surface) method: full 2-surface SMS
lets you additionally control an EXTENDED source (two edge points, not
just one), at the cost of a much harder numerical construction that is
still being refined (see opentir.sms.design_sms_collimator, marked
experimental).

The design is verified independently: rays are launched from the
source at several angles and traced through the RESULTING profile
with the general ray-tracing engine, then the output angle relative
to the optical axis is measured directly (should be ~0 deg).

Run:
    python examples/example_cartesian_oval.py
"""

import numpy as np
import matplotlib.pyplot as plt

from opentir import (
    Surface, OpticalSystem, Ray, Segment,
    PMMA, AIR,
    design_cartesian_oval_collimator, build_cartesian_oval_surface,
    plot_system,
)


def main():
    vertex_distance = 3.0   # mm, on-axis thickness
    theta_max_deg = 45.0    # max collection half-angle (limite fisico ~47.8 deg per PMMA)

    points = design_cartesian_oval_collimator(
        n_source=AIR.n, n_lens=PMMA.n,
        vertex_distance=vertex_distance, theta_max_deg=theta_max_deg,
        n_points=60,
    )
    surfaces = build_cartesian_oval_surface(points, AIR, PMMA, name="cartesian_oval")

    system = OpticalSystem()
    for s in surfaces:
        system.add(s)

    z_target = 100.0
    target = Segment([z_target, 0], [z_target, points[:, 1].max() * 2 + 5], name="target_plane")
    system.add(Surface(target, kind="target", name="target_plane"))

    print("Verifica indipendente (angolo di uscita atteso: 0 deg = collimato):")
    test_thetas_deg = [0, 10, 20, 30, 40]
    traces_for_plot = []
    for theta_deg in test_thetas_deg:
        theta = np.radians(theta_deg)
        direction = np.array([np.cos(theta), np.sin(theta)])
        ray = Ray([0.0, 0.0], direction, power=1.0, medium=AIR)
        branches = system.trace_ray(ray, max_bounces=4, min_power=1e-4)
        # keep only the main transmitted branch for the plot (drop the
        # ~4% Fresnel-reflected branch, which is not the collimated beam)
        best = max(branches, key=lambda b: b["power"])
        traces_for_plot.append(best)
        path = best["path"]
        if len(path) >= 2:
            out_dir = np.array(path[-1]) - np.array(path[-2])
            out_dir = out_dir / np.linalg.norm(out_dir)
            angle_deg = np.degrees(np.arctan2(out_dir[1], out_dir[0]))
            print(f"  theta_emissione={theta_deg:5.1f} deg  ->  "
                  f"uscita a {angle_deg:6.3f} deg dall'asse  (potenza {best['power']:.3f})")

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_system(system, traces_for_plot, ax=ax,
                title="OpenTIR 0.3 - lente collimatrice a ovale cartesiano (esatta)")
    plt.tight_layout()
    plt.savefig("example_cartesian_oval.png", dpi=150)
    print("\nSaved plot to example_cartesian_oval.png")


if __name__ == "__main__":
    main()
