"""Manual headless smoke test for release 0.4 features:
- Chromatic aberration (tick on/off, N wavelengths)
- Zoom/pan toolbar and scroll handler present
- Reset view button works
"""
from opentir.gui import OpenTIRApp

app = OpenTIRApp()

# parabolic mirror + target
app.surfaces.append({
    "name": "specchio", "kind": "mirror", "geom_type": "conic",
    "vertex": [0.0, 0.0], "R": 30.0, "k": -1.0, "r_max": 15.0,
    "coeffs": [0.0, 0.0], "n_points": 60, "flip_z": False, "outward": "+z",
})
app.surfaces.append({
    "name": "lente", "kind": "refract", "geom_type": "conic",
    "vertex": [32.0, 0.0], "R": 12.0, "k": 0.0, "r_max": 8.0,
    "coeffs": [0.0, 0.0], "n_points": 40, "flip_z": False, "outward": "+z",
    "material_in": "PMMA (n=1.49)", "material_out": "Aria (n=1.00)",
})
app.surfaces.append({
    "name": "target", "kind": "target", "geom_type": "segment",
    "p1": [200.0, 0.0], "p2": [200.0, 40.0],
})
app._refresh_tree()
app.src_z.set("15.0")
app.src_r.set("0.0")
app.src_axis.set("90.0")
app.src_half_angle.set("40.0")
app.src_n_rays.set("21")
app.src_distribution.set("lambertian")
app.src_medium.set("Aria (n=1.00)")

# --- Test 1: monochromatic (default) ---
assert not app.chromatic_var.get(), "default deve essere non-cromatico"
app._run_simulation()
stats1 = app.stats_label.cget("text")
assert "Raggi base" in stats1
assert "Lunghezze d'onda" not in stats1
print("TEST 1 (monocromatico):", stats1[:80])

# --- Test 2: chromatic enabled, 7 colours ---
app.chromatic_var.set(True)
app._toggle_chromatic()
app.n_wavelengths.set("7")
app._run_simulation()
stats2 = app.stats_label.cget("text")
assert "Lunghezze d'onda: 7" in stats2
print("TEST 2 (cromatico 7):", stats2[:80])

# --- Test 3: chromatic with 3 colours ---
app.n_wavelengths.set("3")
app._run_simulation()
stats3 = app.stats_label.cget("text")
assert "Lunghezze d'onda: 3" in stats3
print("TEST 3 (cromatico 3):", stats3[:80])

# --- Test 4: zoom/pan handlers exist ---
assert hasattr(app, "toolbar"), "toolbar di navigazione deve essere presente"
assert hasattr(app, "_on_scroll"), "_on_scroll deve esistere"
assert hasattr(app, "_on_right_click"), "_on_right_click deve esistere"
assert hasattr(app, "_reset_view"), "_reset_view deve esistere"
app._reset_view()
print("TEST 4 (toolbar e reset): OK")

# --- Test 5: back to monochromatic ---
app.chromatic_var.set(False)
app._toggle_chromatic()
app._run_simulation()
stats5 = app.stats_label.cget("text")
assert "Lunghezze d'onda" not in stats5
print("TEST 5 (ritorno monocromatico): OK")

app.destroy()
print("\nALL SMOKE TESTS PASSED")
