"""
Example - Aspheric / conic / free-form surfaces
=================================================
Demonstrates the general conic+aspheric profile generator, covering
every classic lens/mirror surface family from a single formula:

    sphere (k=0), parabola (k=-1), hyperbola (k<-1), ellipse (k>-1, k!=0)

plus a fully free-form profile built from arbitrary points.

Run:
    python examples/example_conic_shapes.py
"""

import numpy as np
import matplotlib.pyplot as plt

from opentir import build_conic_profile, build_freeform_profile

R = 15.0
r_max = 8.0

shapes = {
    "Sfera (k=0)": 0.0,
    "Parabola (k=-1)": -1.0,
    "Iperbole (k=-2.5)": -2.5,
    "Ellisse (k=1.5)": 1.5,
}

fig, ax = plt.subplots(figsize=(8, 6))
for label, k in shapes.items():
    points = build_conic_profile(vertex=[0, 0], R=R, k=k, r_max=r_max, n_points=100)
    ax.plot(points[:, 0], points[:, 1], label=label, linewidth=2)

# esempio di superficie a forma libera (punti arbitrari, non da formula)
freeform_pts = build_freeform_profile([[0, 0], [1.5, 3], [4, 6], [8, 8.5], [13, 9]])
ax.plot(freeform_pts[:, 0], freeform_pts[:, 1], "--", color="black",
        linewidth=2, label="Forma libera (punti)")

ax.set_xlabel("z (asse ottico)")
ax.set_ylabel("r (radiale)")
ax.set_title("Famiglie di superfici coniche + forma libera, stesso R e apertura")
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.legend()
ax.grid(True, linewidth=0.3)
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig("example_conic_shapes.png", dpi=150)
print("Saved plot to example_conic_shapes.png")
