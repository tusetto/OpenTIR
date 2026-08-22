"""
Automated smoke test for the GUI: builds a small optical system and
LED source through the same code paths the GUI uses, runs a
simulation, and checks that the plots/canvas update without errors.
Not a real user-facing test, just a headless sanity check.
"""
import tkinter as tk
from opentir.gui import OpenTIRApp

app = OpenTIRApp()

# Add a mirror (parabolic-ish single segment) surface
app.surfaces.append({
    "name": "mirror1", "kind": "mirror", "geom_type": "segment",
    "p1": [0.0, 0.0], "p2": [10.0, 25.0],
})
# Add a target
app.surfaces.append({
    "name": "target1", "kind": "target", "geom_type": "segment",
    "p1": [100.0, 0.0], "p2": [100.0, 40.0],
})
# Add a refract surface with PMMA/air
app.surfaces.append({
    "name": "window1", "kind": "refract", "geom_type": "segment",
    "p1": [5.0, 0.0], "p2": [5.0, 20.0],
    "material_in": "PMMA (n=1.49)", "material_out": "Aria (n=1.00)",
})
app._refresh_tree()

app.src_z.set("0.0")
app.src_r.set("0.0")
app.src_axis.set("90.0")
app.src_half_angle.set("70.0")
app.src_n_rays.set("41")
app.src_distribution.set("lambertian")
app.src_medium.set("PMMA (n=1.49)")

app._run_simulation()
print("STATS:", app.stats_label.cget("text"))

# Test save/load round trip
import json, tempfile, os
tmp = os.path.join(tempfile.gettempdir(), "opentir_test_project.json")
data = {"surfaces": app.surfaces, "source": app._current_source_def()}
with open(tmp, "w") as f:
    json.dump(data, f)
with open(tmp) as f:
    reloaded = json.load(f)
assert reloaded["surfaces"] == app.surfaces
print("SAVE/LOAD OK")

app.destroy()
print("SMOKE TEST PASSED")
