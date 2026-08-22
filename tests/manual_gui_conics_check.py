"""Manual headless smoke check for the new conic/freeform geometry
types in the GUI. Not collected by pytest (see tests/manual_gui_smoke_check.py
for why this naming pattern is used)."""
import numpy as np
from opentir.gui import OpenTIRApp

app = OpenTIRApp()

# Parabolic mirror via the new 'conic' geometry type (k=-1)
app.surfaces.append({
    "name": "parabola_mirror", "kind": "mirror", "geom_type": "conic",
    "vertex": [0.0, 0.0], "R": 20.0, "k": -1.0, "r_max": 15.0,
    "coeffs": [0.0, 0.0], "n_points": 50, "flip_z": False, "outward": "+z",
})
# Hyperbolic refractive surface (k=-2), PMMA/air
app.surfaces.append({
    "name": "hyperbola_lens", "kind": "refract", "geom_type": "conic",
    "vertex": [30.0, 0.0], "R": 10.0, "k": -2.0, "r_max": 6.0,
    "coeffs": [0.0, 0.0], "n_points": 50, "flip_z": False, "outward": "+z",
    "material_in": "PMMA (n=1.49)", "material_out": "Aria (n=1.00)",
})
# Freeform mirror
app.surfaces.append({
    "name": "freeform_mirror", "kind": "mirror", "geom_type": "freeform",
    "points": [[50.0, 0.0], [55.0, 5.0], [58.0, 10.0], [58.0, 15.0]],
    "outward": "+z",
})
# Target
app.surfaces.append({
    "name": "target1", "kind": "target", "geom_type": "segment",
    "p1": [200.0, 0.0], "p2": [200.0, 40.0],
})
app._refresh_tree()

app.src_z.set("0.0")
app.src_r.set("0.0")
app.src_axis.set("90.0")
app.src_half_angle.set("30.0")
app.src_n_rays.set("21")
app.src_distribution.set("lambertian")
app.src_medium.set("Aria (n=1.00)")

app._run_simulation()
print("STATS:", app.stats_label.cget("text"))

# Sanity: verify build_surface_objects expands conic/freeform into many segments
from opentir.gui import build_surface_objects
parab_surfs = build_surface_objects(app.surfaces[0])
assert len(parab_surfs) == 49, f"attesi 49 segmenti, trovati {len(parab_surfs)}"
hyp_surfs = build_surface_objects(app.surfaces[1])
assert len(hyp_surfs) == 49
free_surfs = build_surface_objects(app.surfaces[2])
assert len(free_surfs) == 3
print("EXPANSION OK:", len(parab_surfs), len(hyp_surfs), len(free_surfs))

app.destroy()
print("SMOKE TEST PASSED")
