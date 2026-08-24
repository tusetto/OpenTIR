"""
opentir.gui
~~~~~~~~~~~~
Desktop GUI (Tkinter) for OpenTIR, release 0.2.x.

Two panels:
  - "Sistema ottico": define/edit optical surfaces (segments or arcs,
    with kind mirror/target/block/refract and materials) and the LED
    source parameters. Projects can be saved/loaded as JSON.
  - "Simulazione": run the ray tracer and inspect the results
    (ray-trace plot + illuminance histogram + summary stats), embedded
    via matplotlib's Tkinter backend.
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .geometry import Segment, Arc
from .materials import Material, AIR, PMMA, POLYCARBONATE, BK7_GLASS, SODA_LIME_GLASS
from .optics import Surface, OpticalSystem, critical_angle
from .source import LEDSource
from .visualize import plot_system, plot_illuminance
from .profiles import build_conic_profile, build_freeform_profile, profile_to_surfaces
from .chromatic import wavelength_samples, chromatic_rays

MATERIAL_PRESETS = {
    "Aria (n=1.00)": AIR,
    "PMMA (n=1.49)": PMMA,
    "Policarbonato (n=1.585)": POLYCARBONATE,
    "Vetro BK7 (n=1.517)": BK7_GLASS,
    "Vetro soda-lime (n=1.52)": SODA_LIME_GLASS,
    "Personalizzato...": None,
}

SURFACE_KINDS = ["mirror", "target", "block", "refract"]
GEOM_TYPES = ["segment", "arc", "conic", "freeform"]
CONIC_PRESETS = {
    "Sfera (k=0)": 0.0,
    "Parabola (k=-1)": -1.0,
    "Iperbole (k=-2)": -2.0,
    "Ellisse (k=-0.5)": -0.5,
    "Personalizzato...": None,
}
DISTRIBUTIONS = ["lambertian", "uniform"]


def _material_from_choice(name_var, custom_var):
    name = name_var.get()
    if name in MATERIAL_PRESETS and MATERIAL_PRESETS[name] is not None:
        return MATERIAL_PRESETS[name]
    try:
        n = float(custom_var.get())
    except ValueError:
        n = 1.0
    return Material(n=n, name=f"n={n}")


class SurfaceForm(ttk.LabelFrame):
    """Form to create/edit a single optical surface."""

    def __init__(self, master, on_save):
        super().__init__(master, text="Definizione superficie")
        self.on_save = on_save
        self.editing_index = None

        row = 0
        ttk.Label(self, text="Nome").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        self.name_var = tk.StringVar(value="superficie1")
        ttk.Entry(self, textvariable=self.name_var, width=20).grid(row=row, column=1, columnspan=3, sticky="w")

        row += 1
        ttk.Label(self, text="Tipo").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        self.kind_var = tk.StringVar(value="mirror")
        kind_box = ttk.Combobox(self, textvariable=self.kind_var, values=SURFACE_KINDS,
                                 state="readonly", width=12)
        kind_box.grid(row=row, column=1, sticky="w")
        kind_box.bind("<<ComboboxSelected>>", lambda e: self._update_material_visibility())

        ttk.Label(self, text="Geometria").grid(row=row, column=2, sticky="w", padx=4)
        self.geom_var = tk.StringVar(value="segment")
        geom_box = ttk.Combobox(self, textvariable=self.geom_var, values=GEOM_TYPES,
                                 state="readonly", width=10)
        geom_box.grid(row=row, column=3, sticky="w")
        geom_box.bind("<<ComboboxSelected>>", lambda e: self._update_geom_fields())

        row += 1
        self.geom_frame = ttk.Frame(self)
        self.geom_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=4)

        row += 1
        self.material_frame = ttk.Frame(self)
        self.material_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=4)

        row += 1
        btns = ttk.Frame(self)
        btns.grid(row=row, column=0, columnspan=4, sticky="w", pady=6)
        ttk.Button(btns, text="Nuova", command=self.clear).pack(side="left", padx=2)
        ttk.Button(btns, text="Salva superficie", command=self._save).pack(side="left", padx=2)

        # segment fields
        self.seg_vars = {k: tk.StringVar(value=v) for k, v in
                          [("z1", "0.0"), ("r1", "0.0"), ("z2", "10.0"), ("r2", "10.0")]}
        # arc fields
        self.arc_vars = {k: tk.StringVar(value=v) for k, v in
                          [("cz", "0.0"), ("cr", "0.0"), ("radius", "10.0"),
                           ("theta1", "0.0"), ("theta2", "90.0")]}
        # conic (sphere/parabola/hyperbole/ellipse/general aspheric) fields
        self.conic_vars = {k: tk.StringVar(value=v) for k, v in
                            [("vz", "0.0"), ("vr", "0.0"), ("R", "10.0"), ("k", "-1.0"),
                             ("rmax", "10.0"), ("A4", "0.0"), ("A6", "0.0"), ("npoints", "80")]}
        self.conic_preset_var = tk.StringVar(value="Parabola (k=-1)")
        self.conic_flip_var = tk.BooleanVar(value=False)
        self.conic_outward_var = tk.StringVar(value="+z")
        # freeform fields
        self.freeform_text = None
        self.freeform_outward_var = tk.StringVar(value="+z")

        # material fields
        self.mat_in_var = tk.StringVar(value="PMMA (n=1.49)")
        self.mat_in_custom = tk.StringVar(value="1.50")
        self.mat_out_var = tk.StringVar(value="Aria (n=1.00)")
        self.mat_out_custom = tk.StringVar(value="1.00")

        self._update_geom_fields()
        self._update_material_visibility()

    def _update_geom_fields(self):
        for w in self.geom_frame.winfo_children():
            w.destroy()
        geom = self.geom_var.get()
        if geom == "segment":
            labels = [("z1", "0.0"), ("r1", "0.0"), ("z2", "10.0"), ("r2", "10.0")]
            for i, (key, _) in enumerate(labels):
                ttk.Label(self.geom_frame, text=f"{key} [mm]").grid(row=0, column=2 * i, sticky="w", padx=2)
                ttk.Entry(self.geom_frame, textvariable=self.seg_vars[key], width=8).grid(
                    row=0, column=2 * i + 1, padx=2)
        elif geom == "arc":
            labels = [("cz", "centro z"), ("cr", "centro r"), ("radius", "raggio"),
                      ("theta1", "theta1 [deg]"), ("theta2", "theta2 [deg]")]
            for i, (key, label) in enumerate(labels):
                ttk.Label(self.geom_frame, text=f"{label}").grid(row=0, column=2 * i, sticky="w", padx=2)
                ttk.Entry(self.geom_frame, textvariable=self.arc_vars[key], width=7).grid(
                    row=0, column=2 * i + 1, padx=2)
        elif geom == "conic":
            ttk.Label(self.geom_frame, text="Tipo").grid(row=0, column=0, sticky="w", padx=2)
            preset_box = ttk.Combobox(self.geom_frame, textvariable=self.conic_preset_var,
                                       values=list(CONIC_PRESETS.keys()), state="readonly", width=16)
            preset_box.grid(row=0, column=1, padx=2)
            preset_box.bind("<<ComboboxSelected>>", lambda e: self._apply_conic_preset())

            labels_row1 = [("vz", "vertice z"), ("vr", "vertice r"), ("R", "raggio curv. R"), ("k", "k")]
            for i, (key, label) in enumerate(labels_row1):
                ttk.Label(self.geom_frame, text=label).grid(row=1, column=2 * i, sticky="w", padx=2)
                e = ttk.Entry(self.geom_frame, textvariable=self.conic_vars[key], width=7)
                e.grid(row=1, column=2 * i + 1, padx=2)
                if key == "k":
                    self._conic_k_entry = e

            labels_row2 = [("rmax", "r_max"), ("A4", "A4"), ("A6", "A6"), ("npoints", "n. punti")]
            for i, (key, label) in enumerate(labels_row2):
                ttk.Label(self.geom_frame, text=label).grid(row=2, column=2 * i, sticky="w", padx=2)
                ttk.Entry(self.geom_frame, textvariable=self.conic_vars[key], width=7).grid(
                    row=2, column=2 * i + 1, padx=2)

            ttk.Checkbutton(self.geom_frame, text="Curva verso -z (flip)",
                             variable=self.conic_flip_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
            ttk.Label(self.geom_frame, text="Lato esterno verso").grid(row=3, column=2, sticky="w", padx=2)
            ttk.Combobox(self.geom_frame, textvariable=self.conic_outward_var, values=["+z", "-z"],
                         state="readonly", width=5).grid(row=3, column=3, padx=2)
            self._apply_conic_preset()
        else:  # freeform
            ttk.Label(self.geom_frame,
                      text="Punti (uno per riga, formato: z,r)").grid(row=0, column=0, columnspan=2, sticky="w")
            self.freeform_text = tk.Text(self.geom_frame, width=24, height=6)
            self.freeform_text.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
            self.freeform_text.insert("1.0", "0.0,0.0\n2.0,3.0\n5.0,6.0\n9.0,8.0")
            ttk.Label(self.geom_frame, text="Lato esterno verso").grid(row=0, column=2, sticky="w", padx=6)
            ttk.Combobox(self.geom_frame, textvariable=self.freeform_outward_var, values=["+z", "-z"],
                         state="readonly", width=5).grid(row=0, column=3, padx=2)

    def _apply_conic_preset(self):
        preset = CONIC_PRESETS.get(self.conic_preset_var.get())
        if preset is not None:
            self.conic_vars["k"].set(str(preset))

    def _update_material_visibility(self):
        for w in self.material_frame.winfo_children():
            w.destroy()
        if self.kind_var.get() != "refract":
            return
        ttk.Label(self.material_frame, text="Materiale interno (lato opposto normale)").grid(
            row=0, column=0, sticky="w", padx=2)
        cb_in = ttk.Combobox(self.material_frame, textvariable=self.mat_in_var,
                              values=list(MATERIAL_PRESETS.keys()), state="readonly", width=20)
        cb_in.grid(row=0, column=1, padx=2)
        ttk.Entry(self.material_frame, textvariable=self.mat_in_custom, width=6).grid(row=0, column=2, padx=2)

        ttk.Label(self.material_frame, text="Materiale esterno (lato normale)").grid(
            row=1, column=0, sticky="w", padx=2)
        cb_out = ttk.Combobox(self.material_frame, textvariable=self.mat_out_var,
                               values=list(MATERIAL_PRESETS.keys()), state="readonly", width=20)
        cb_out.grid(row=1, column=1, padx=2)
        ttk.Entry(self.material_frame, textvariable=self.mat_out_custom, width=6).grid(row=1, column=2, padx=2)

    def clear(self):
        self.editing_index = None
        self.name_var.set(f"superficie_{np.random.randint(1000)}")

    def load(self, index, surf_def):
        self.editing_index = index
        self.name_var.set(surf_def["name"])
        self.kind_var.set(surf_def["kind"])
        self.geom_var.set(surf_def["geom_type"])
        self._update_geom_fields()
        gt = surf_def["geom_type"]
        if gt == "segment":
            self.seg_vars["z1"].set(str(surf_def["p1"][0]))
            self.seg_vars["r1"].set(str(surf_def["p1"][1]))
            self.seg_vars["z2"].set(str(surf_def["p2"][0]))
            self.seg_vars["r2"].set(str(surf_def["p2"][1]))
        elif gt == "arc":
            self.arc_vars["cz"].set(str(surf_def["center"][0]))
            self.arc_vars["cr"].set(str(surf_def["center"][1]))
            self.arc_vars["radius"].set(str(surf_def["radius"]))
            self.arc_vars["theta1"].set(str(surf_def["theta1_deg"]))
            self.arc_vars["theta2"].set(str(surf_def["theta2_deg"]))
        elif gt == "conic":
            self.conic_preset_var.set("Personalizzato...")
            self.conic_vars["vz"].set(str(surf_def["vertex"][0]))
            self.conic_vars["vr"].set(str(surf_def["vertex"][1]))
            self.conic_vars["R"].set(str(surf_def["R"]))
            self.conic_vars["k"].set(str(surf_def["k"]))
            self.conic_vars["rmax"].set(str(surf_def["r_max"]))
            coeffs = surf_def.get("coeffs", [0.0, 0.0])
            self.conic_vars["A4"].set(str(coeffs[0] if len(coeffs) > 0 else 0.0))
            self.conic_vars["A6"].set(str(coeffs[1] if len(coeffs) > 1 else 0.0))
            self.conic_vars["npoints"].set(str(surf_def.get("n_points", 80)))
            self.conic_flip_var.set(surf_def.get("flip_z", False))
            self.conic_outward_var.set(surf_def.get("outward", "+z"))
        else:  # freeform
            self.freeform_text.delete("1.0", "end")
            lines = "\n".join(f"{p[0]},{p[1]}" for p in surf_def.get("points", []))
            self.freeform_text.insert("1.0", lines)
            self.freeform_outward_var.set(surf_def.get("outward", "+z"))
        self._update_material_visibility()
        if surf_def["kind"] == "refract":
            self.mat_in_var.set(surf_def.get("material_in", "PMMA (n=1.49)"))
            self.mat_out_var.set(surf_def.get("material_out", "Aria (n=1.00)"))

    def _save(self):
        try:
            surf_def = {"name": self.name_var.get(), "kind": self.kind_var.get(),
                        "geom_type": self.geom_var.get()}
            gt = self.geom_var.get()
            if gt == "segment":
                surf_def["p1"] = [float(self.seg_vars["z1"].get()), float(self.seg_vars["r1"].get())]
                surf_def["p2"] = [float(self.seg_vars["z2"].get()), float(self.seg_vars["r2"].get())]
            elif gt == "arc":
                surf_def["center"] = [float(self.arc_vars["cz"].get()), float(self.arc_vars["cr"].get())]
                surf_def["radius"] = float(self.arc_vars["radius"].get())
                surf_def["theta1_deg"] = float(self.arc_vars["theta1"].get())
                surf_def["theta2_deg"] = float(self.arc_vars["theta2"].get())
            elif gt == "conic":
                surf_def["vertex"] = [float(self.conic_vars["vz"].get()), float(self.conic_vars["vr"].get())]
                surf_def["R"] = float(self.conic_vars["R"].get())
                surf_def["k"] = float(self.conic_vars["k"].get())
                surf_def["r_max"] = float(self.conic_vars["rmax"].get())
                surf_def["coeffs"] = [float(self.conic_vars["A4"].get()), float(self.conic_vars["A6"].get())]
                surf_def["n_points"] = int(self.conic_vars["npoints"].get())
                surf_def["flip_z"] = bool(self.conic_flip_var.get())
                surf_def["outward"] = self.conic_outward_var.get()
            else:  # freeform
                pts = []
                raw = self.freeform_text.get("1.0", "end").strip()
                for line_no, line in enumerate(raw.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.replace(";", ",").split(",")]
                    if len(parts) != 2:
                        raise ValueError(f"riga {line_no} non valida: '{line}' (atteso 'z,r')")
                    pts.append([float(parts[0]), float(parts[1])])
                if len(pts) < 2:
                    raise ValueError("servono almeno 2 punti per una superficie a forma libera")
                surf_def["points"] = pts
                surf_def["outward"] = self.freeform_outward_var.get()
            if self.kind_var.get() == "refract":
                surf_def["material_in"] = self.mat_in_var.get()
                surf_def["material_out"] = self.mat_out_var.get()
                if self.mat_in_var.get() == "Personalizzato...":
                    surf_def["material_in_n"] = float(self.mat_in_custom.get())
                if self.mat_out_var.get() == "Personalizzato...":
                    surf_def["material_out_n"] = float(self.mat_out_custom.get())
        except ValueError as exc:
            messagebox.showerror("Errore", f"Valore numerico non valido: {exc}")
            return
        self.on_save(self.editing_index, surf_def)
        self.clear()


def build_surface_objects(surf_def):
    """Turn a GUI surface dict into a list of opentir Surface instances
    (a single-element list for segment/arc, a chain of segments for
    conic/freeform profiles)."""
    material_in = material_out = None
    if surf_def["kind"] == "refract":
        material_in = _preset_or_custom(surf_def, "material_in")
        material_out = _preset_or_custom(surf_def, "material_out")

    gt = surf_def["geom_type"]
    if gt == "segment":
        geom = Segment(surf_def["p1"], surf_def["p2"], name=surf_def["name"])
        return [Surface(geom, kind=surf_def["kind"], name=surf_def["name"],
                         material_in=material_in, material_out=material_out)]
    elif gt == "arc":
        geom = Arc(surf_def["center"], surf_def["radius"],
                   np.radians(surf_def["theta1_deg"]), np.radians(surf_def["theta2_deg"]),
                   name=surf_def["name"])
        return [Surface(geom, kind=surf_def["kind"], name=surf_def["name"],
                         material_in=material_in, material_out=material_out)]
    elif gt == "conic":
        points = build_conic_profile(
            vertex=surf_def["vertex"], R=surf_def["R"], k=surf_def["k"],
            r_max=surf_def["r_max"], coeffs=surf_def.get("coeffs", ()),
            n_points=surf_def.get("n_points", 80), flip_z=surf_def.get("flip_z", False),
        )
        outward = (1.0, 0.0) if surf_def.get("outward", "+z") == "+z" else (-1.0, 0.0)
        return profile_to_surfaces(points, kind=surf_def["kind"],
                                    material_in=material_in, material_out=material_out,
                                    outward_direction=outward, name=surf_def["name"])
    else:  # freeform
        points = build_freeform_profile(surf_def["points"])
        outward = (1.0, 0.0) if surf_def.get("outward", "+z") == "+z" else (-1.0, 0.0)
        return profile_to_surfaces(points, kind=surf_def["kind"],
                                    material_in=material_in, material_out=material_out,
                                    outward_direction=outward, name=surf_def["name"])


def _preset_or_custom(surf_def, key):
    name = surf_def.get(key, "Aria (n=1.00)")
    if name == "Personalizzato...":
        n = surf_def.get(f"{key}_n", 1.0)
        return Material(n=n, name=f"n={n}")
    return MATERIAL_PRESETS.get(name, AIR)


class OpenTIRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenTIR - Editor e Simulatore ottiche LED")
        self.geometry("1150x760")

        self.surfaces = []  # list of surf_def dicts

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.system_tab = ttk.Frame(notebook)
        self.sim_tab = ttk.Frame(notebook)
        notebook.add(self.system_tab, text="Sistema ottico")
        notebook.add(self.sim_tab, text="Simulazione")

        self._build_system_tab()
        self._build_sim_tab()

    # ------------------------------------------------------------------
    # Tab 1: system definition
    # ------------------------------------------------------------------
    def _build_system_tab(self):
        left = ttk.Frame(self.system_tab)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        right = ttk.Frame(self.system_tab)
        right.pack(side="right", fill="y", padx=8, pady=8)

        ttk.Label(left, text="Superfici definite", font=("", 10, "bold")).pack(anchor="w")
        columns = ("name", "kind", "geom")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=12)
        for c, label in zip(columns, ["Nome", "Tipo", "Geometria"]):
            self.tree.heading(c, text=label)
            self.tree.column(c, width=140)
        self.tree.pack(fill="both", expand=True, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_surface)

        btn_row = ttk.Frame(left)
        btn_row.pack(anchor="w", pady=4)
        ttk.Button(btn_row, text="Elimina selezionata", command=self._delete_selected).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Salva progetto...", command=self._save_project).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Carica progetto...", command=self._load_project).pack(side="left", padx=2)

        self.form = SurfaceForm(left, on_save=self._on_surface_saved)
        self.form.pack(fill="x", pady=8)

        # LED source panel
        src_frame = ttk.LabelFrame(right, text="Sorgente LED")
        src_frame.pack(fill="x", pady=4)

        self.src_z = tk.StringVar(value="0.0")
        self.src_r = tk.StringVar(value="0.0")
        self.src_axis = tk.StringVar(value="90.0")
        self.src_half_angle = tk.StringVar(value="80.0")
        self.src_n_rays = tk.StringVar(value="61")
        self.src_distribution = tk.StringVar(value="lambertian")
        self.src_medium = tk.StringVar(value="Aria (n=1.00)")

        fields = [
            ("Posizione z [mm]", self.src_z), ("Posizione r [mm]", self.src_r),
            ("Asse emissione [deg]", self.src_axis), ("Semi-angolo cono [deg]", self.src_half_angle),
            ("N. raggi", self.src_n_rays),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(src_frame, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(src_frame, textvariable=var, width=10).grid(row=i, column=1, padx=4)

        r = len(fields)
        ttk.Label(src_frame, text="Distribuzione").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(src_frame, textvariable=self.src_distribution, values=DISTRIBUTIONS,
                     state="readonly", width=12).grid(row=r, column=1, padx=4)

        r += 1
        ttk.Label(src_frame, text="Mezzo di emissione").grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(src_frame, textvariable=self.src_medium,
                     values=[k for k in MATERIAL_PRESETS if k != "Personalizzato..."],
                     state="readonly", width=16).grid(row=r, column=1, padx=4)

    def _on_surface_saved(self, index, surf_def):
        if index is None:
            self.surfaces.append(surf_def)
        else:
            self.surfaces[index] = surf_def
        self._refresh_tree()

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, s in enumerate(self.surfaces):
            self.tree.insert("", "end", iid=str(i), values=(s["name"], s["kind"], s["geom_type"]))

    def _on_select_surface(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        index = int(sel[0])
        self.form.load(index, self.surfaces[index])

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        index = int(sel[0])
        del self.surfaces[index]
        self.form.clear()
        self._refresh_tree()

    def _current_source_def(self):
        return {
            "position": [float(self.src_z.get()), float(self.src_r.get())],
            "axis_deg": float(self.src_axis.get()),
            "half_angle_deg": float(self.src_half_angle.get()),
            "n_rays": int(self.src_n_rays.get()),
            "distribution": self.src_distribution.get(),
            "medium": self.src_medium.get(),
        }

    def _save_project(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("OpenTIR project", "*.json")])
        if not path:
            return
        try:
            data = {"surfaces": self.surfaces, "source": self._current_source_def()}
        except ValueError as exc:
            messagebox.showerror("Errore", f"Parametri sorgente non validi: {exc}")
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Salvato", f"Progetto salvato in {path}")

    def _load_project(self):
        path = filedialog.askopenfilename(filetypes=[("OpenTIR project", "*.json")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.surfaces = data.get("surfaces", [])
        self._refresh_tree()
        src = data.get("source", {})
        if src:
            self.src_z.set(str(src["position"][0]))
            self.src_r.set(str(src["position"][1]))
            self.src_axis.set(str(src["axis_deg"]))
            self.src_half_angle.set(str(src["half_angle_deg"]))
            self.src_n_rays.set(str(src["n_rays"]))
            self.src_distribution.set(src["distribution"])
            self.src_medium.set(src["medium"])

    # ------------------------------------------------------------------
    # Tab 2: simulation
    # ------------------------------------------------------------------
    def _build_sim_tab(self):
        top = ttk.Frame(self.sim_tab)
        top.pack(fill="x", padx=8, pady=8)

        self.max_bounces = tk.StringVar(value="15")
        self.min_power = tk.StringVar(value="0.001")

        ttk.Label(top, text="Max rimbalzi").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(top, textvariable=self.max_bounces, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(top, text="Potenza minima ramo").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Entry(top, textvariable=self.min_power, width=8).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Esegui simulazione", command=self._run_simulation).grid(
            row=0, column=4, padx=12)

        self.stats_label = ttk.Label(top, text="", foreground="#1a5276")
        self.stats_label.grid(row=1, column=0, columnspan=5, sticky="w", pady=6)

        self.figure = Figure(figsize=(11, 5.2), dpi=100)
        self.ax_system = self.figure.add_subplot(1, 2, 1)
        self.ax_illum = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.sim_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _run_simulation(self):
        if not self.surfaces:
            messagebox.showwarning("Attenzione", "Definisci almeno una superficie nella scheda 'Sistema ottico'.")
            return
        try:
            system = OpticalSystem()
            for s in self.surfaces:
                for surf_obj in build_surface_objects(s):
                    system.add(surf_obj)

            src_def = self._current_source_def()
            medium = MATERIAL_PRESETS.get(src_def["medium"], AIR)
            source = LEDSource(position=src_def["position"], axis_deg=src_def["axis_deg"],
                                half_angle_deg=src_def["half_angle_deg"], n_rays=src_def["n_rays"],
                                distribution=src_def["distribution"], medium=medium)
            rays = source.generate_rays()

            max_bounces = int(self.max_bounces.get())
            min_power = float(self.min_power.get())
            traces = system.trace_many(rays, max_bounces=max_bounces, min_power=min_power)
        except Exception as exc:
            messagebox.showerror("Errore nella simulazione", str(exc))
            return

        self.ax_system.clear()
        self.ax_illum.clear()
        plot_system(system, traces, ax=self.ax_system)
        target_names = [s["name"] for s in self.surfaces if s["kind"] == "target"]
        plot_illuminance(traces, target_name=target_names[0] if target_names else None, ax=self.ax_illum)
        self.canvas.draw()

        total_in = sum(r.power for r in rays)
        total_hit = sum(t["power"] for t in traces if t["hits"])
        pct = 100 * total_hit / total_in if total_in else 0.0
        msg = (f"Raggi emessi: {len(rays)}  |  Rami dopo Fresnel/TIR: {len(traces)}  |  "
               f"Potenza emessa: {total_in:.2f}  |  Potenza sul target: {total_hit:.2f} ({pct:.1f}%)")
        self.stats_label.config(text=msg)


def main():
    app = OpenTIRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
