"""
opentir.gui
~~~~~~~~~~~~
Desktop GUI — CustomTkinter, release 0.7.1.

Unified interface without tabs.
LED source panel opens as a popup window.
Auto-update simulation when parameters change.
Fixed: surface selection with clickable frames.
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox
import time

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from .geometry import Segment, Arc
from .materials import Material, AIR, PMMA, POLYCARBONATE, BK7_GLASS, SODA_LIME_GLASS
from .optics import Surface, OpticalSystem
from .source import LEDSource
from .visualize import plot_system, plot_illuminance
from .profiles import build_conic_profile, build_freeform_profile, profile_to_surfaces, conic_sag
from .chromatic import wavelength_samples, wavelength_to_rgb, chromatic_rays
from .export_dxf import export_dxf, get_hit_points
from .lee import compute_lee, plot_lee_pie, plot_lee_bar

# ── appearance ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_LABEL  = ("Segoe UI", 11)
FONT_BOLD   = ("Segoe UI", 11, "bold")
FONT_SMALL  = ("Segoe UI", 10)
FONT_TITLE  = ("Segoe UI", 12, "bold")
ACCENT      = "#1f6aa5"
BG_CARD     = "#2b2b2b"
BG_DARK     = "#1e1e1e"
BG_SELECTED = "#1f6aa5"
BG_HOVER    = "#3a3a3a"

# ── material presets ─────────────────────────────────────────────────────────
MATERIAL_PRESETS = {
    "Aria (n=1.00)":          AIR,
    "PMMA (n=1.49)":          PMMA,
    "Policarbonato (n=1.585)": POLYCARBONATE,
    "Vetro BK7 (n=1.517)":    BK7_GLASS,
    "Vetro soda-lime (n=1.52)": SODA_LIME_GLASS,
    "Personalizzato...":      None,
}

SURFACE_KINDS = ["mirror", "target", "block", "refract"]
GEOM_TYPES    = ["segment", "arc", "conic", "freeform"]
DISTRIBUTIONS = ["lambertian", "uniform"]

MATERIAL_FILL_COLOR = {
    "Aria (n=1.00)":           "#888888",
    "PMMA (n=1.49)":           "#4ab0f0",
    "Policarbonato (n=1.585)": "#b070e0",
    "Vetro BK7 (n=1.517)":    "#40c880",
    "Vetro soda-lime (n=1.52)":"#d0c040",
    "Personalizzato...":       "#909090",
}
LENS_FILL_ALPHA = 0.35

# ── helpers ──────────────────────────────────────────────────────────────────
def _lbl(parent, text, font=None, **kw):
    return ctk.CTkLabel(parent, text=text, font=font or FONT_LABEL, **kw)

def _entry(parent, textvariable, width=100, **kw):
    return ctk.CTkEntry(parent, textvariable=textvariable,
                        width=width, font=FONT_SMALL, **kw)

def _btn(parent, text, command, width=120, **kw):
    return ctk.CTkButton(parent, text=text, command=command,
                         width=width, font=FONT_LABEL, **kw)

def _combo(parent, variable, values, width=120, state="readonly", **kw):
    return ctk.CTkComboBox(parent, variable=variable, values=values,
                           width=width, font=FONT_SMALL,
                           state=state, **kw)

def _preset_or_custom(surf_def, key):
    name = surf_def.get(key, "Aria (n=1.00)")
    if name == "Personalizzato...":
        n = surf_def.get(f"{key}_n", 1.0)
        return Material(n=n, name=f"n={n}")
    return MATERIAL_PRESETS.get(name, AIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Lens geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sag_at_r(surf_def, r):
    gt = surf_def["geom_type"]
    if gt == "segment":
        return surf_def["p1"][0]
    elif gt == "conic":
        vz   = surf_def["vertex"][0]
        R    = surf_def["R"]
        k    = surf_def["k"]
        coef = surf_def.get("coeffs", ())
        flip = surf_def.get("flip_z", False)
        s    = float(conic_sag(np.array([r]), R, k, coef)[0])
        return vz + (-s if flip else s)
    return 0.0


def build_lens_surface_defs(lens_def):
    r_max    = float(lens_def["r_max"])
    mat_name = lens_def["material"]
    origin_z = float(lens_def["origin_z"])
    t_edge   = float(lens_def["t_edge"])
    n_pts    = int(lens_def.get("n_points", 80))

    fg = dict(lens_def["fronte_geom"])
    fg["geom_type"] = fg.get("geom_type", "conic")
    if fg["geom_type"] == "conic":
        fg["vertex"] = [origin_z, 0.0]
        fg.setdefault("coeffs", [0.0, 0.0])
        fg["r_max"] = r_max
        R_f_raw = fg.get("R", 0.0)
        if R_f_raw < 0:
            fg["flip_z"] = True
            fg["R"]      = abs(R_f_raw)
        else:
            fg.setdefault("flip_z", False)
    elif fg["geom_type"] == "segment":
        fg["p1"] = [origin_z, -r_max]
        fg["p2"] = [origin_z,  r_max]

    sag_f_edge = _sag_at_r(fg, r_max)
    sag_f_0    = origin_z

    rg = dict(lens_def["retro_geom"])
    rg["geom_type"] = rg.get("geom_type", "conic")
    rg.setdefault("coeffs", [0.0, 0.0])
    rg["r_max"] = r_max

    if rg["geom_type"] == "conic":
        R_r_raw = rg.get("R", 15.0)
        k_r     = rg.get("k", 0.0)
        coef_r  = rg.get("coeffs", [0.0, 0.0])

        if R_r_raw < 0:
            flip_r = True
            R_r    = abs(R_r_raw)
        else:
            flip_r = rg.get("flip_z", True)
            R_r    = R_r_raw

        sag_r_edge = float(conic_sag(np.array([r_max]), R_r, k_r, coef_r)[0])
        if flip_r:
            z_retro_v = sag_f_edge + t_edge + sag_r_edge
        else:
            z_retro_v = sag_f_edge + t_edge - sag_r_edge
        rg["R"]      = R_r
        rg["vertex"] = [z_retro_v, 0.0]
        rg["flip_z"] = flip_r
        rg["outward"] = rg.get("outward", "+z")

    elif rg["geom_type"] == "segment":
        z_retro_v = sag_f_edge + t_edge
        rg["p1"] = [z_retro_v, -r_max]
        rg["p2"] = [z_retro_v,  r_max]

    t_center = z_retro_v - sag_f_0
    lens_def["_t_center"] = round(t_center, 4)

    origin_r     = float(lens_def.get("origin_r", 0.0))
    rotation_deg = float(lens_def.get("rotation_deg", 0.0))

    fronte_def = {
        "name":         f"{lens_def['name']}_fronte",
        "kind":         "refract",
        "geom_type":    fg["geom_type"],
        "material_in":  "Aria (n=1.00)",
        "material_out": mat_name,
        "origin_r":     origin_r,
        "rotation_deg": rotation_deg,
    }
    fronte_def.update({k: v for k, v in fg.items() if k not in ("geom_type",)})

    retro_def = {
        "name":         f"{lens_def['name']}_retro",
        "kind":         "refract",
        "geom_type":    rg["geom_type"],
        "material_in":  mat_name,
        "material_out": "Aria (n=1.00)",
        "origin_r":     origin_r,
        "rotation_deg": rotation_deg,
    }
    retro_def.update({k: v for k, v in rg.items() if k not in ("geom_type",)})

    return fronte_def, retro_def


# ═══════════════════════════════════════════════════════════════════════════════
# LensForm — dialog for creating/editing a compound lens
# ═══════════════════════════════════════════════════════════════════════════════

class LensForm(ctk.CTkToplevel):
    GEOM_OPTIONS = ["conic", "segment"]

    def __init__(self, master, on_save, lens_def=None):
        super().__init__(master)
        self.title("Definizione lente")
        self.geometry("780x580")
        self.resizable(False, False)
        self.grab_set()
        self.on_save  = on_save

        self.lens_name  = tk.StringVar(value="lente1")
        self.mat_var    = tk.StringVar(value="PMMA (n=1.49)")
        self.origin_z   = tk.StringVar(value="5.0")
        self.origin_r   = tk.StringVar(value="0.0")
        self.rotation   = tk.StringVar(value="0.0")
        self.r_max      = tk.StringVar(value="10.0")
        self.t_edge     = tk.StringVar(value="1.0")
        self.n_points   = tk.StringVar(value="80")

        self.f_geom = tk.StringVar(value="conic")
        self.f_R    = tk.StringVar(value="0")
        self.f_k    = tk.StringVar(value="0.0")
        self.f_A4   = tk.StringVar(value="0.0")
        self.f_A6   = tk.StringVar(value="0.0")

        self.r_geom = tk.StringVar(value="conic")
        self.r_R    = tk.StringVar(value="-15.0")
        self.r_k    = tk.StringVar(value="0.0")
        self.r_A4   = tk.StringVar(value="0.0")
        self.r_A6   = tk.StringVar(value="0.0")
        self.r_flip = tk.BooleanVar(value=True)

        if lens_def:
            self._populate(lens_def)

        self._build_ui()

        for v in (self.origin_z, self.origin_r, self.rotation, self.r_max, self.t_edge,
                  self.f_R, self.f_k, self.f_A4, self.f_A6,
                  self.r_R, self.r_k, self.r_A4, self.r_A6):
            v.trace_add("write", self._update_t_center)
        self.r_flip.trace_add("write", self._update_t_center)
        self._update_t_center()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        _lbl(top, "Nome lente:").grid(row=0, column=0, **pad, sticky="w")
        _entry(top, self.lens_name, width=130).grid(row=0, column=1, **pad)

        _lbl(top, "Materiale:").grid(row=0, column=2, **pad, sticky="w")
        mat_names = [k for k in MATERIAL_PRESETS if k != "Personalizzato..."]
        _combo(top, self.mat_var, mat_names, width=200).grid(row=0, column=3, **pad)

        _lbl(top, "r_max [mm]:").grid(row=0, column=4, **pad, sticky="w")
        _entry(top, self.r_max, width=65).grid(row=0, column=5, **pad)

        _lbl(top, "N. punti:").grid(row=0, column=6, **pad, sticky="w")
        _entry(top, self.n_points, width=55).grid(row=0, column=7, **pad)

        dim = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        dim.pack(fill="x", padx=10, pady=4)

        _lbl(dim, "Origine z [mm]").grid(row=0, column=0, **pad, sticky="w")
        _entry(dim, self.origin_z, width=70).grid(row=0, column=1, **pad)

        _lbl(dim, "Origine r [mm]").grid(row=0, column=2, **pad, sticky="w")
        _entry(dim, self.origin_r, width=70).grid(row=0, column=3, **pad)

        _lbl(dim, "Rotazione [°]").grid(row=0, column=4, **pad, sticky="w")
        _entry(dim, self.rotation, width=70).grid(row=0, column=5, **pad)

        _lbl(dim, "Spessore bordo [mm]").grid(row=1, column=0, **pad, sticky="w")
        _entry(dim, self.t_edge, width=70).grid(row=1, column=1, **pad)

        _lbl(dim, "Spessore centro [mm]").grid(row=1, column=2, **pad, sticky="w")
        self._tc_lbl = _lbl(dim, "—", font=FONT_BOLD, text_color="#58a6ff")
        self._tc_lbl.grid(row=1, column=3, **pad)

        surfaces_row = ctk.CTkFrame(self, fg_color="transparent")
        surfaces_row.pack(fill="both", expand=True, padx=10, pady=4)

        self._build_surface_card(surfaces_row, side="left",
                                 title="Fronte (prima superficie)",
                                 geom_var=self.f_geom,
                                 R_var=self.f_R, k_var=self.f_k,
                                 A4_var=self.f_A4, A6_var=self.f_A6,
                                 flip_var=None)
        self._build_surface_card(surfaces_row, side="right",
                                 title="Retro (seconda superficie)",
                                 geom_var=self.r_geom,
                                 R_var=self.r_R, k_var=self.r_k,
                                 A4_var=self.r_A4, A6_var=self.r_A6,
                                 flip_var=self.r_flip)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(4, 10))
        _btn(btn_row, "💾  Salva lente", self._save, width=160).pack(side="left", padx=6)
        _btn(btn_row, "✕  Annulla", self.destroy, width=120,
             fg_color="gray30", hover_color="gray40").pack(side="left", padx=4)

    def _build_surface_card(self, parent, side, title,
                             geom_var, R_var, k_var, A4_var, A6_var, flip_var):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        card.pack(side=side, fill="both", expand=True, padx=4, pady=4)

        _lbl(card, title, font=FONT_BOLD).grid(
            row=0, column=0, columnspan=4, padx=8, pady=(8,4), sticky="w")

        _lbl(card, "Tipo:", font=FONT_SMALL).grid(row=1, column=0, padx=8, pady=3, sticky="w")
        _combo(card, geom_var, self.GEOM_OPTIONS, width=100).grid(row=1, column=1, padx=4)

        for i, (label, var) in enumerate([
            ("R [mm]", R_var), ("k:", k_var), ("A4:", A4_var), ("A6:", A6_var),
        ]):
            r = 2 + i
            _lbl(card, label, font=FONT_SMALL).grid(row=r, column=0, padx=8, pady=3, sticky="w")
            _entry(card, var, width=100).grid(row=r, column=1, padx=4)

    def _make_geom_dict(self, geom_var, R_var, k_var, A4_var, A6_var, flip_var=None):
        gtype = geom_var.get()
        R_val = float(R_var.get())
        if gtype == "segment" or R_val == 0:
            return {"geom_type": "segment"}
        flip = R_val < 0
        return {
            "geom_type": "conic",
            "R":     abs(R_val),
            "k":     float(k_var.get()),
            "coeffs":[float(A4_var.get()), float(A6_var.get())],
            "flip_z": flip,
            "outward": "+z",
        }

    def _update_t_center(self, *_):
        try:
            lens_def = self._make_lens_def()
            build_lens_surface_defs(lens_def)
            tc = lens_def.get("_t_center", None)
            if tc is not None:
                txt = f"{tc:.3f} mm"
                col = "#58a6ff" if tc > 0 else "#ff6b6b"
                self._tc_lbl.configure(text=txt, text_color=col)
        except Exception:
            self._tc_lbl.configure(text="errore", text_color="#ff6b6b")

    def _make_lens_def(self):
        return {
            "type":        "lens",
            "name":        self.lens_name.get(),
            "material":    self.mat_var.get(),
            "origin_z":    float(self.origin_z.get()),
            "origin_r":   float(self.origin_r.get()),
            "rotation_deg": float(self.rotation.get()),
            "r_max":       float(self.r_max.get()),
            "t_edge":      float(self.t_edge.get()),
            "n_points":    int(self.n_points.get()),
            "fronte_geom": self._make_geom_dict(
                self.f_geom, self.f_R, self.f_k, self.f_A4, self.f_A6),
            "retro_geom":  self._make_geom_dict(
                self.r_geom, self.r_R, self.r_k, self.r_A4, self.r_A6, self.r_flip),
        }

    def _save(self):
        try:
            t_edge = float(self.t_edge.get())
            if t_edge <= 0:
                raise ValueError("Lo spessore sul bordo deve essere > 0")
            lens_def = self._make_lens_def()
            build_lens_surface_defs(lens_def)
            if lens_def.get("_t_center", 1) <= 0:
                raise ValueError("Spessore al centro ≤ 0: aumenta t_edge")
        except Exception as exc:
            messagebox.showerror("Errore geometria lente", str(exc))
            return
        self.on_save(lens_def)
        self.destroy()

    def _populate(self, lens_def):
        self.lens_name.set(lens_def.get("name", "lente1"))
        self.mat_var.set(lens_def.get("material", "PMMA (n=1.49)"))
        self.origin_z.set(str(lens_def.get("origin_z", 5.0)))
        self.origin_r.set(str(lens_def.get("origin_r", 0.0)))
        self.rotation.set(str(lens_def.get("rotation_deg", 0.0)))
        self.r_max.set(str(lens_def.get("r_max", 10.0)))
        self.t_edge.set(str(lens_def.get("t_edge", 1.0)))
        self.n_points.set(str(lens_def.get("n_points", 80)))
        fg = lens_def.get("fronte_geom", {})
        self.f_geom.set(fg.get("geom_type", "conic"))
        f_R_val = fg.get("R", 0)
        f_flip  = fg.get("flip_z", False)
        if f_flip and f_R_val > 0:
            f_R_val = -f_R_val
        self.f_R.set(str(f_R_val))
        self.f_k.set(str(fg.get("k", 0.0)))
        if fg.get("coeffs"):
            self.f_A4.set(str(fg["coeffs"][0]))
            self.f_A6.set(str(fg["coeffs"][1]))
        rg = lens_def.get("retro_geom", {})
        self.r_geom.set(rg.get("geom_type", "conic"))
        r_R_val = rg.get("R", 15.0)
        r_flip  = rg.get("flip_z", True)
        if r_flip and r_R_val > 0:
            r_R_val = -r_R_val
        self.r_R.set(str(r_R_val))
        self.r_k.set(str(rg.get("k", 0.0)))
        if rg.get("coeffs"):
            self.r_A4.set(str(rg["coeffs"][0]))
            self.r_A6.set(str(rg["coeffs"][1]))


# ══════════════════════════════════════════════════════════════════════════════
# SurfaceForm — dialog for creating/editing a single surface
# ═══════════════════════════════════════════════════════════════════════════════

class SurfaceForm(ctk.CTkToplevel):
    def __init__(self, master, on_save, surf_def=None):
        super().__init__(master)
        self.title("Definizione superficie")
        self.geometry("700x500")
        self.resizable(False, False)
        self.grab_set()
        self.on_save = on_save

        self.name_var = tk.StringVar(value="superficie1")
        self.kind_var = tk.StringVar(value="mirror")
        self.geom_var = tk.StringVar(value="segment")

        self.seg_vars = {k: tk.StringVar(value=v) for k, v in
                         [("z1","0.0"),("r1","-10.0"),("z2","10.0"),("r2","10.0")]}
        self.arc_vars = {k: tk.StringVar(value=v) for k, v in
                         [("cz","0.0"),("cr","0.0"),("radius","10.0"),
                          ("theta1","0.0"),("theta2","90.0")]}
        self.conic_vars = {k: tk.StringVar(value=v) for k, v in
                           [("vz","0.0"),("vr","0.0"),("R","10.0"),("k","-1.0"),
                            ("rmax","10.0"),("A4","0.0"),("A6","0.0"),("npoints","80")]}
        self.conic_flip_var   = tk.BooleanVar(value=False)
        self.conic_outward_var = tk.StringVar(value="+z")
        self.freeform_outward_var = tk.StringVar(value="+z")

        self.mat_in_var     = tk.StringVar(value="PMMA (n=1.49)")
        self.mat_out_var    = tk.StringVar(value="Aria (n=1.00)")

        if surf_def:
            self._populate(surf_def)

        self._build_ui()
        self._update_geom()
        self._update_material()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        top.pack(fill="x", padx=10, pady=10)

        _lbl(top, "Nome").grid(row=0, column=0, **pad, sticky="w")
        _entry(top, self.name_var, width=160).grid(row=0, column=1, **pad)

        _lbl(top, "Tipo").grid(row=0, column=2, **pad, sticky="w")
        _combo(top, self.kind_var, SURFACE_KINDS, width=120,
               command=lambda v: self._update_material()).grid(row=0, column=3, **pad)

        _lbl(top, "Geometria").grid(row=0, column=4, **pad, sticky="w")
        _combo(top, self.geom_var, GEOM_TYPES, width=120,
               command=lambda v: self._update_geom()).grid(row=0, column=5, **pad)

        self.geom_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        self.geom_frame.pack(fill="x", padx=10, pady=4)

        self.mat_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        self.mat_frame.pack(fill="x", padx=10, pady=4)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=10)
        _btn(btn_row, "  Salva superficie", self._save, width=180).pack(side="left", padx=4)
        _btn(btn_row, "✕  Annulla", self.destroy, width=120,
             fg_color="gray30", hover_color="gray40").pack(side="left", padx=4)

    def _update_geom(self, *_):
        for w in self.geom_frame.winfo_children():
            w.destroy()
        gt = self.geom_var.get()
        pad = {"padx": 10, "pady": 4}

        if gt == "segment":
            _lbl(self.geom_frame, "Segmento (linea retta)", font=FONT_BOLD).grid(
                row=0, column=0, columnspan=4, **pad, sticky="w")
            for i, (k, lbl) in enumerate([("z1","z1 [mm]"),("r1","r1 [mm]"),
                                          ("z2","z2 [mm]"),("r2","r2 [mm]")]):
                _lbl(self.geom_frame, lbl, font=FONT_SMALL).grid(row=1, column=i, **pad, sticky="w")
                _entry(self.geom_frame, self.seg_vars[k], width=80).grid(row=2, column=i, **pad)
        elif gt == "arc":
            _lbl(self.geom_frame, "Arco circolare", font=FONT_BOLD).grid(
                row=0, column=0, columnspan=5, **pad, sticky="w")
            for i, (k, lbl) in enumerate([("cz","Centro z"),("cr","Centro r"),
                                          ("radius","Raggio"),("theta1","θ1 [°]"),("theta2","θ2 [°]")]):
                _lbl(self.geom_frame, lbl, font=FONT_SMALL).grid(row=1, column=i, **pad, sticky="w")
                _entry(self.geom_frame, self.arc_vars[k], width=70).grid(row=2, column=i, **pad)
        elif gt == "conic":
            _lbl(self.geom_frame, "Conica / Asferica", font=FONT_BOLD).grid(
                row=0, column=0, columnspan=4, **pad, sticky="w")
            labels = [("vz","Vertex z"),("vr","Vertex r"),("R","R [mm]"),
                      ("k","k"),("rmax","r_max"),("A4","A4"),("A6","A6"),("npoints","N. punti")]
            for i, (k, lbl) in enumerate(labels):
                _lbl(self.geom_frame, lbl, font=FONT_SMALL).grid(row=1, column=i, **pad, sticky="w")
                _entry(self.geom_frame, self.conic_vars[k], width=70).grid(row=2, column=i, **pad)
            _lbl(self.geom_frame, "Outward:", font=FONT_SMALL).grid(row=3, column=0, **pad, sticky="w")
            _combo(self.geom_frame, self.conic_outward_var, ["+z", "-z"], width=80).grid(row=3, column=1, **pad)
        elif gt == "freeform":
            _lbl(self.geom_frame, "Freeform (punti z,r)", font=FONT_BOLD).grid(
                row=0, column=0, columnspan=2, **pad, sticky="w")
            _lbl(self.geom_frame, "Outward:", font=FONT_SMALL).grid(row=1, column=0, **pad, sticky="w")
            _combo(self.geom_frame, self.freeform_outward_var, ["+z", "-z"], width=80).grid(row=1, column=1, **pad)

    def _update_material(self, *_):
        for w in self.mat_frame.winfo_children():
            w.destroy()
        if self.kind_var.get() != "refract":
            return
        pad = {"padx": 10, "pady": 4}
        _lbl(self.mat_frame, "Materiale interno", font=FONT_BOLD).grid(row=0, column=0, **pad, sticky="w")
        _lbl(self.mat_frame, "Materiale esterno", font=FONT_BOLD).grid(row=0, column=2, **pad, sticky="w")
        mat_names = [k for k in MATERIAL_PRESETS if k != "Personalizzato..."]
        _combo(self.mat_frame, self.mat_in_var, mat_names, width=180).grid(row=1, column=0, **pad)
        _combo(self.mat_frame, self.mat_out_var, mat_names, width=180).grid(row=1, column=2, **pad)

    def _save(self):
        try:
            surf_def = {
                "name": self.name_var.get(),
                "kind": self.kind_var.get(),
                "geom_type": self.geom_var.get(),
            }
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
                surf_def["flip_z"] = self.conic_flip_var.get()
                surf_def["outward"] = self.conic_outward_var.get()
            elif gt == "freeform":
                surf_def["points"] = []
                surf_def["outward"] = self.freeform_outward_var.get()

            if self.kind_var.get() == "refract":
                surf_def["material_in"] = self.mat_in_var.get()
                surf_def["material_out"] = self.mat_out_var.get()

            self.on_save(surf_def)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Errore", str(exc))

    def _populate(self, surf_def):
        self.name_var.set(surf_def.get("name", "superficie1"))
        self.kind_var.set(surf_def.get("kind", "mirror"))
        self.geom_var.set(surf_def.get("geom_type", "segment"))
        if "p1" in surf_def:
            self.seg_vars["z1"].set(str(surf_def["p1"][0]))
            self.seg_vars["r1"].set(str(surf_def["p1"][1]))
            self.seg_vars["z2"].set(str(surf_def["p2"][0]))
            self.seg_vars["r2"].set(str(surf_def["p2"][1]))
        if "center" in surf_def:
            self.arc_vars["cz"].set(str(surf_def["center"][0]))
            self.arc_vars["cr"].set(str(surf_def["center"][1]))
            self.arc_vars["radius"].set(str(surf_def["radius"]))
            self.arc_vars["theta1"].set(str(surf_def["theta1_deg"]))
            self.arc_vars["theta2"].set(str(surf_def["theta2_deg"]))
        if "vertex" in surf_def:
            self.conic_vars["vz"].set(str(surf_def["vertex"][0]))
            self.conic_vars["vr"].set(str(surf_def["vertex"][1]))
            self.conic_vars["R"].set(str(surf_def["R"]))
            self.conic_vars["k"].set(str(surf_def["k"]))
            self.conic_vars["rmax"].set(str(surf_def["r_max"]))
            if surf_def.get("coeffs"):
                self.conic_vars["A4"].set(str(surf_def["coeffs"][0]))
                self.conic_vars["A6"].set(str(surf_def["coeffs"][1]))
            self.conic_vars["npoints"].set(str(surf_def.get("n_points", 80)))
        if "material_in" in surf_def:
            self.mat_in_var.set(surf_def["material_in"])
            self.mat_out_var.set(surf_def["material_out"])


# ═══════════════════════════════════════════════════════════════════════════════
# ClickableSurfaceItem — frame for each surface in the list
# ═══════════════════════════════════════════════════════════════════════════════

class ClickableSurfaceItem(ctk.CTkFrame):
    """A clickable frame representing a surface/lens in the list."""
    
    def __init__(self, parent, index, surf_data, on_click, on_delete, on_edit):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=6)
        self.index = index
        self.surf_data = surf_data
        self.on_click = on_click
        self.on_delete = on_delete
        self.on_edit = on_edit
        self.is_selected = False
        
        self._build_ui()
        
        # Bind click events
        self.bind("<Button-1>", lambda e: self._handle_click())
        self.label.bind("<Button-1>", lambda e: self._handle_click())
        self.detail_label.bind("<Button-1>", lambda e: self._handle_click())
    
    def _build_ui(self):
        # Main content
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(side="left", fill="both", expand=True, padx=8, pady=4)
        
        # Title line
        if self.surf_data.get("type") == "lens":
            mat = self.surf_data.get("material", "PMMA")
            name = self.surf_data.get("name", "lente")
            tc = self.surf_data.get("_t_center", "?")
            title = f"🔷 {name} [{mat}] t={tc}mm"
        else:
            name = self.surf_data.get("name", "superficie")
            kind = self.surf_data.get("kind", "?")
            geom = self.surf_data.get("geom_type", "?")
            title = f" {name} ({kind}/{geom})"
        
        self.label = ctk.CTkLabel(content_frame, text=title, 
                                  font=FONT_SMALL, anchor="w")
        self.label.pack(fill="x")
        
        # Detail line
        if self.surf_data.get("type") == "lens":
            fg = self.surf_data.get("fronte_geom", {})
            rg = self.surf_data.get("retro_geom", {})
            detail = f"Fronte: {fg.get('geom_type','?')} R={fg.get('R','?')} | Retro: {rg.get('geom_type','?')} R={rg.get('R','?')}"
        else:
            detail = f"Geometria: {self.surf_data.get('geom_type', '?')}"
        
        self.detail_label = ctk.CTkLabel(content_frame, text=detail,
                                         font=("Segoe UI", 9), 
                                         text_color="gray", anchor="w")
        self.detail_label.pack(fill="x")
    
    def _handle_click(self):
        self.on_click(self.index)
    
    def _handle_delete(self, event):
        event.stopPropagation()
        self.on_delete(self.index)
    
    def _handle_edit(self, event):
        event.stopPropagation()
        self.on_edit(self.index)
    
    def set_selected(self, selected):
        self.is_selected = selected
        if selected:
            self.configure(fg_color=BG_SELECTED)
            self.label.configure(text_color="white")
            self.detail_label.configure(text_color="lightgray")
        else:
            self.configure(fg_color=BG_CARD)
            self.label.configure(text_color="white")
            self.detail_label.configure(text_color="gray")


# ═══════════════════════════════════════════════════════════════════════════════
# LEDSourcePanel — popup window for LED source definition
# ═══════════════════════════════════════════════════════════════════════════════

class LEDSourcePanel(ctk.CTkToplevel):
    def __init__(self, master, on_save, source_def=None):
        super().__init__(master)
        self.title("Sorgente LED")
        self.geometry("420x720")
        self.resizable(True, True)
        self.grab_set()
        self.on_save = on_save

        self.src_z = tk.StringVar(value="0.0")
        self.src_r = tk.StringVar(value="0.0")
        self.src_axis = tk.StringVar(value="0.0")
        self.src_half_angle = tk.StringVar(value="60.0")
        self.src_n_rays = tk.StringVar(value="81")
        self.src_distribution = tk.StringVar(value="lambertian")
        self.src_medium = tk.StringVar(value="Aria (n=1.00)")
        self.src_les_shape = tk.StringVar(value="point")
        self.src_les_size = tk.StringVar(value="1.25")
        self.src_les_n = tk.StringVar(value="5")
        self.enable_chromatic = tk.BooleanVar(value=False)

        if source_def:
            self._populate(source_def)

        self._build_ui()
        self._on_les_change()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        pos_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        pos_frame.pack(fill="x", padx=10, pady=10)
        _lbl(pos_frame, "Posizione [mm]", font=FONT_BOLD).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        _lbl(pos_frame, "Asse z", font=FONT_SMALL).grid(row=1, column=0, sticky="w", **pad)
        _entry(pos_frame, self.src_z, width=80).grid(row=1, column=1, **pad)
        _lbl(pos_frame, "Asse r", font=FONT_SMALL).grid(row=2, column=0, sticky="w", **pad)
        _entry(pos_frame, self.src_r, width=80).grid(row=2, column=1, **pad)

        ang_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        ang_frame.pack(fill="x", padx=10, pady=8)
        _lbl(ang_frame, "Angoli [°]", font=FONT_BOLD).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        _lbl(ang_frame, "Asse emissione", font=FONT_SMALL).grid(row=1, column=0, sticky="w", **pad)
        _entry(ang_frame, self.src_axis, width=80).grid(row=1, column=1, **pad)
        _lbl(ang_frame, "Semi-angolo", font=FONT_SMALL).grid(row=2, column=0, sticky="w", **pad)
        _entry(ang_frame, self.src_half_angle, width=80).grid(row=2, column=1, **pad)

        rays_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        rays_frame.pack(fill="x", padx=10, pady=8)
        _lbl(rays_frame, "Raggi", font=FONT_BOLD).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        _lbl(rays_frame, "N. raggi angolari", font=FONT_SMALL).grid(row=1, column=0, sticky="w", **pad)
        _entry(rays_frame, self.src_n_rays, width=80).grid(row=1, column=1, **pad)
        _lbl(rays_frame, "Distribuzione", font=FONT_SMALL).grid(row=2, column=0, sticky="w", **pad)
        _combo(rays_frame, self.src_distribution, DISTRIBUTIONS, width=80).grid(row=2, column=1, **pad)

        med_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        med_frame.pack(fill="x", padx=10, pady=8)
        _lbl(med_frame, "Mezzo emissione", font=FONT_BOLD).grid(row=0, column=0, sticky="w", **pad)
        mat_names = [k for k in MATERIAL_PRESETS if k != "Personalizzato..."]
        _combo(med_frame, self.src_medium, mat_names, width=180).grid(row=1, column=0, **pad)

        les_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        les_frame.pack(fill="x", padx=10, pady=8)
        _lbl(les_frame, "LES (Light Emitting Surface)", font=FONT_BOLD).grid(row=0, column=0, sticky="w", **pad)
        _lbl(les_frame, "Forma LES", font=FONT_SMALL).grid(row=1, column=0, sticky="w", **pad)
        _combo(les_frame, self.src_les_shape, ["point", "square", "circle"], width=180,
               command=lambda v: self._on_les_change()).grid(row=1, column=0, **pad)

        self._les_size_lbl = _lbl(les_frame, "Dimensione [mm]", font=FONT_SMALL)
        self._les_size_entry = _entry(les_frame, self.src_les_size, width=80)
        self._les_n_lbl = _lbl(les_frame, "N. sotto-sorgenti", font=FONT_SMALL)
        self._les_n_entry = _entry(les_frame, self.src_les_n, width=80)

        chrom_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        chrom_frame.pack(fill="x", padx=10, pady=8)
        _lbl(chrom_frame, "Aberrazione cromatica", font=FONT_BOLD).grid(row=0, column=0, sticky="w", **pad)
        ctk.CTkCheckBox(chrom_frame, text="Simula dispersione cromatica",
                       variable=self.enable_chromatic, font=FONT_SMALL).grid(row=1, column=0, sticky="w", **pad)

        self._total_rays_lbl = _lbl(self, "Raggi totali: 0", font=FONT_SMALL, text_color="#58a6ff")
        self._total_rays_lbl.pack(pady=8)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        _btn(btn_frame, "✓  Applica", self._apply, width=120).pack(side="left", padx=5)
        _btn(btn_frame, "✕  Annulla", self.destroy, width=120,
             fg_color="gray30", hover_color="gray40").pack(side="left", padx=5)

    def _on_les_change(self):
        shape = self.src_les_shape.get()
        if shape == "point":
            self._les_size_lbl.grid_remove()
            self._les_size_entry.grid_remove()
            self._les_n_lbl.grid_remove()
            self._les_n_entry.grid_remove()
        else:
            self._les_size_lbl.grid(row=2, column=0, sticky="w", padx=10, pady=6)
            self._les_size_entry.grid(row=2, column=0, padx=10, pady=6)
            self._les_n_lbl.grid(row=3, column=0, sticky="w", padx=10, pady=6)
            self._les_n_entry.grid(row=3, column=0, padx=10, pady=6)
        self._update_total_rays()

    def _update_total_rays(self):
        try:
            n_rays = int(self.src_n_rays.get())
            shape = self.src_les_shape.get()
            if shape == "point":
                total = n_rays
            else:
                n_les = max(1, int(self.src_les_n.get()))
                total = n_rays * n_les
            self._total_rays_lbl.configure(text=f"Raggi totali: {total}")
        except:
            pass

    def _apply(self):
        try:
            source_def = {
                "position": [float(self.src_z.get()), float(self.src_r.get())],
                "axis_deg": float(self.src_axis.get()),
                "half_angle_deg": float(self.src_half_angle.get()),
                "n_rays": int(self.src_n_rays.get()),
                "distribution": self.src_distribution.get(),
                "medium": self.src_medium.get(),
                "les_shape": self.src_les_shape.get(),
                "les_size": float(self.src_les_size.get()) if self.src_les_shape.get() != "point" else 0.0,
                "les_n": int(self.src_les_n.get()) if self.src_les_shape.get() != "point" else 1,
                "enable_chromatic": self.enable_chromatic.get(),
            }
            self.on_save(source_def)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Errore", f"Parametri non validi:\n{e}")

    def _populate(self, source_def):
        self.src_z.set(str(source_def["position"][0]))
        self.src_r.set(str(source_def["position"][1]))
        self.src_axis.set(str(source_def["axis_deg"]))
        self.src_half_angle.set(str(source_def["half_angle_deg"]))
        self.src_n_rays.set(str(source_def["n_rays"]))
        self.src_distribution.set(source_def["distribution"])
        self.src_medium.set(source_def["medium"])
        self.src_les_shape.set(source_def.get("les_shape", "point"))
        self.src_les_size.set(str(source_def.get("les_size", 1.25)))
        self.src_les_n.set(str(source_def.get("les_n", 5)))
        self.enable_chromatic.set(source_def.get("enable_chromatic", False))


# ═══════════════════════════════════════════════════════════════════════════════
# Surface object builders
# ═══════════════════════════════════════════════════════════════════════════════

def _mirror_surface_object(surf):
    geom = surf.geometry
    if not isinstance(geom, Segment):
        return None
    if surf.kind not in ("target", "block"):
        return None
    p1, p2 = geom.p1, geom.p2
    if min(p1[1], p2[1]) < -1e-9:
        return None
    mp1 = np.array([p1[0], -p1[1]])
    mp2 = np.array([p2[0], -p2[1]])
    mirrored_geom = Segment(mp1, mp2, name=geom.name)
    return Surface(mirrored_geom, kind=surf.kind, name=surf.name,
                   material_in=surf.material_in, material_out=surf.material_out)


def _build_symmetric_system(surfaces_in):
    system = OpticalSystem()
    for surf in surfaces_in:
        system.add(surf)
        mirror = _mirror_surface_object(surf)
        if mirror is not None:
            system.add(mirror)
    return system


def build_surface_objects(surf_def):
    material_in = material_out = None
    if surf_def["kind"] == "refract":
        material_in  = _preset_or_custom(surf_def, "material_in")
        material_out = _preset_or_custom(surf_def, "material_out")

    gt = surf_def["geom_type"]
    if gt == "segment":
        geom = Segment(surf_def["p1"], surf_def["p2"], name=surf_def["name"])
        return [Surface(geom, kind=surf_def["kind"], name=surf_def["name"],
                        material_in=material_in, material_out=material_out)]
    elif gt == "arc":
        geom = Arc(surf_def["center"], surf_def["radius"],
                   np.radians(surf_def["theta1_deg"]),
                   np.radians(surf_def["theta2_deg"]),
                   name=surf_def["name"])
        return [Surface(geom, kind=surf_def["kind"], name=surf_def["name"],
                        material_in=material_in, material_out=material_out)]
    elif gt == "conic":
        pts_half = build_conic_profile(
            vertex=surf_def["vertex"], R=surf_def["R"], k=surf_def["k"],
            r_max=surf_def["r_max"], coeffs=surf_def.get("coeffs", ()),
            n_points=surf_def.get("n_points", 80),
            flip_z=surf_def.get("flip_z", False))
        _origin_r = surf_def.get("origin_r", 0.0)
        _rot_deg  = surf_def.get("rotation_deg", 0.0)
        points = np.vstack([pts_half[::-1] * [1, -1], pts_half[1:]])
        if _origin_r != 0.0 or _rot_deg != 0.0:
            _c, _s = np.cos(np.radians(_rot_deg)), np.sin(np.radians(_rot_deg))
            pts_t = points.copy()
            pts_t[:, 1] += _origin_r
            zz, rr = pts_t[:, 0], pts_t[:, 1]
            points = np.column_stack([zz * _c - rr * _s, zz * _s + rr * _c])
        outward = (1.0, 0.0) if surf_def.get("outward", "+z") == "+z" else (-1.0, 0.0)
        return profile_to_surfaces(points, kind=surf_def["kind"],
                                   material_in=material_in, material_out=material_out,
                                   outward_direction=outward, name=surf_def["name"])
    else:
        points = build_freeform_profile(surf_def["points"])
        outward = (1.0, 0.0) if surf_def.get("outward", "+z") == "+z" else (-1.0, 0.0)
        return profile_to_surfaces(points, kind=surf_def["kind"],
                                   material_in=material_in, material_out=material_out,
                                   outward_direction=outward, name=surf_def["name"])


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application — Unified Interface v0.7.1
# ═══════════════════════════════════════════════════════════════════════════════

class OpenTIRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("OpenTIR — LED Optical Simulator v0.7.1")
        self.geometry("1400x850")

        self.surfaces = []
        self.selected_index = None
        self.surface_items = []
        
        self.source_def = {
            "position": [0.0, 0.0],
            "axis_deg": 0.0,
            "half_angle_deg": 60.0,
            "n_rays": 81,
            "distribution": "lambertian",
            "medium": "Aria (n=1.00)",
            "les_shape": "point",
            "les_size": 1.25,
            "les_n": 5,
        }
        self._last_traces = None
        self._last_total_power = None
        self._last_lee = None
        self._last_source = None
        self._lens_fill_data = []
        self._plot_system = None

        self._update_scheduled = False
        self._update_delay = 0.5

        self._build_ui()

    def _build_ui(self):
        # Top control bar
        top_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=60)
        top_bar.pack(fill="x", padx=8, pady=(8, 4))
        top_bar.pack_propagate(False)

        _btn(top_bar, " Sorgente LED", self._open_led_source, width=140,
             fg_color="#2a5a8a", hover_color="#3a7aaa").pack(side="left", padx=6, pady=8)

        sim_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        sim_frame.pack(side="left", padx=10)

        self.max_bounces = tk.StringVar(value="15")
        self.min_power = tk.StringVar(value="0.001")
        self.refl_length = tk.StringVar(value="20")

        for label, var, w in [
            ("Max rimbalzi", self.max_bounces, 50),
            ("Potenza min", self.min_power, 60),
            ("Lungh. riflessi", self.refl_length, 70),
        ]:
            ctk.CTkLabel(sim_frame, text=label, font=FONT_SMALL).pack(side="left", padx=(8, 2))
            _entry(sim_frame, var, width=w).pack(side="left", padx=2)

        self.auto_update_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(top_bar, text="Auto-aggiornamento",
                       variable=self.auto_update_var, font=FONT_SMALL).pack(side="left", padx=10)

        _btn(top_bar, "▶ Esegui", self._run_simulation, width=100,
             fg_color="#2a8a2a", hover_color="#3aaa3a").pack(side="left", padx=6)
        _btn(top_bar, " Reset vista", self._reset_view, width=110,
             fg_color="gray30", hover_color="gray40").pack(side="left", padx=4)

        # Main content area
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Left panel — surfaces
        left_panel = ctk.CTkFrame(main_frame, fg_color=BG_CARD, corner_radius=8, width=350)
        left_panel.pack(side="left", fill="y", padx=(0, 4))
        left_panel.pack_propagate(False)

        self._build_surface_panel(left_panel)

        # Center panel — ray trace
        center_panel = ctk.CTkFrame(main_frame, fg_color=BG_DARK, corner_radius=8)
        center_panel.pack(side="left", fill="both", expand=True, padx=4)

        self.figure = Figure(facecolor="#1e1e1e", dpi=100)
        self.ax_system = self.figure.add_subplot(1, 1, 1)
        self._style_ax(self.ax_system)
        self.ax_system.set_title("OpenTIR – ray trace")

        self.canvas = FigureCanvasTkAgg(self.figure, master=center_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self.toolbar = NavigationToolbar2Tk(self.canvas, center_panel, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_right_click)

        # Right panel — analysis
        right_panel = ctk.CTkFrame(main_frame, fg_color=BG_CARD, corner_radius=8, width=420)
        right_panel.pack(side="right", fill="y", padx=(4, 0))
        right_panel.pack_propagate(False)

        self._build_analysis_panel(right_panel)

        # Bottom stats bar
        stats_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=40)
        stats_bar.pack(fill="x", padx=8, pady=(0, 8))
        stats_bar.pack_propagate(False)

        self.stats_label = _lbl(stats_bar, "Pronta per la simulazione",
                               font=FONT_SMALL, text_color="#58a6ff")
        self.stats_label.pack(padx=8, pady=6)

    def _build_surface_panel(self, parent):
        # Header
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(10, 4))
        _lbl(header, "Superfici definite", font=FONT_BOLD).pack(side="left", anchor="w")
        
        # Add buttons
        add_frame = ctk.CTkFrame(parent, fg_color="transparent")
        add_frame.pack(fill="x", padx=8, pady=4)
        _btn(add_frame, "+ Aggiungi lente", self._add_lens, width=140,
             fg_color="#2a5a8a").pack(side="left", padx=2)
        _btn(add_frame, "+ Superficie", self._add_surface, width=120).pack(side="left", padx=2)

        # Scrollable frame for surface items
        self.surface_scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.surface_scroll.pack(fill="both", expand=True, padx=8, pady=4)

        # Action buttons
        action_frame = ctk.CTkFrame(parent, fg_color="transparent")
        action_frame.pack(fill="x", padx=8, pady=4)
        _btn(action_frame, "✏️ Modifica", self._edit_selected, width=120,
             fg_color="#2a5a8a").pack(side="left", padx=2)
        _btn(action_frame, "🗑 Elimina", self._delete_selected, width=120,
             fg_color="#8a2a2a").pack(side="left", padx=2)

        # Project buttons
        proj_frame = ctk.CTkFrame(parent, fg_color="transparent")
        proj_frame.pack(fill="x", padx=8, pady=4)
        _btn(proj_frame, "💾 Salva progetto", self._save_project, width=140).pack(side="left", padx=2)
        _btn(proj_frame, "📂 Carica", self._load_project, width=100).pack(side="left", padx=2)

    def _build_analysis_panel(self, parent):
        # Illuminance histogram
        hist_frame = ctk.CTkFrame(parent, fg_color=BG_DARK, corner_radius=8, height=220)
        hist_frame.pack(fill="x", padx=8, pady=8)
        hist_frame.pack_propagate(False)

        _lbl(hist_frame, "Illuminamento sul target", font=FONT_BOLD).pack(padx=8, pady=(6, 2), anchor="w")

        self.fig_illum = Figure(facecolor="#1e1e1e", dpi=80, figsize=(4, 2.5))
        self.ax_illum = self.fig_illum.add_subplot(1, 1, 1)
        self._style_ax(self.ax_illum)
        self.ax_illum.set_title("Illuminance distribution")

        self.canvas_illum = FigureCanvasTkAgg(self.fig_illum, master=hist_frame)
        self.canvas_illum.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Radial distribution
        iso_frame = ctk.CTkFrame(parent, fg_color=BG_DARK, corner_radius=8, height=220)
        iso_frame.pack(fill="x", padx=8, pady=4)
        iso_frame.pack_propagate(False)

        _lbl(iso_frame, "Distribuzione radiale", font=FONT_BOLD).pack(padx=8, pady=(6, 2), anchor="w")

        self.fig_iso = Figure(facecolor="#1e1e1e", dpi=80, figsize=(4, 2.5))
        self.ax_iso = self.fig_iso.add_subplot(1, 1, 1)
        self._style_ax(self.ax_iso)
        self.ax_iso.set_title("Radial distribution")

        self.canvas_iso = FigureCanvasTkAgg(self.fig_iso, master=iso_frame)
        self.canvas_iso.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # LEE breakdown
        lee_frame = ctk.CTkFrame(parent, fg_color=BG_DARK, corner_radius=8)
        lee_frame.pack(fill="both", expand=True, padx=8, pady=4)

        _lbl(lee_frame, "LEE Breakdown", font=FONT_BOLD).pack(padx=8, pady=(6, 2), anchor="w")

        self.fig_lee = Figure(facecolor="#1e1e1e", dpi=80, figsize=(4, 3))
        self.ax_lee = self.fig_lee.add_subplot(1, 1, 1)
        self._style_ax(self.ax_lee)
        self.ax_lee.set_title("LEE Breakdown")

        self.canvas_lee = FigureCanvasTkAgg(self.fig_lee, master=lee_frame)
        self.canvas_lee.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Export buttons
        exp_frame = ctk.CTkFrame(parent, fg_color="transparent")
        exp_frame.pack(fill="x", padx=8, pady=4)
        _btn(exp_frame, "💾 Esporta DXF", self._export_dxf, width=120,
             fg_color="#2a5a2a").pack(side="left", padx=2)
        _btn(exp_frame, "🌡 Isofote", self._show_isophote, width=100,
             fg_color="#5a3a2a").pack(side="left", padx=2)
        _btn(exp_frame, "📊 LEE", self._show_lee, width=80,
             fg_color="#3a2a5a").pack(side="left", padx=2)

    def _style_ax(self, ax):
        ax.set_facecolor("#252526")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

    # ── Surface Management ─────────────────────────────────────────────────────
    
    def _on_surface_click(self, index):
        """Handle click on a surface item."""
        self.selected_index = index
        for i, item in enumerate(self.surface_items):
            item.set_selected(i == index)
    
    def _on_surface_delete(self, index):
        """Handle delete button click."""
        if 0 <= index < len(self.surfaces):
            self.surfaces.pop(index)
            self.selected_index = None
            self._refresh_surface_list()
            if self.auto_update_var.get():
                self._schedule_auto_update()
    
    def _on_surface_edit(self, index):
        """Handle edit button click."""
        if 0 <= index < len(self.surfaces):
            self._edit_surface_at(index)
    
    def _delete_selected(self):
        """Delete the currently selected surface."""
        if self.selected_index is None:
            messagebox.showinfo("Info", "Seleziona una superficie dalla lista")
            return
        self._on_surface_delete(self.selected_index)
    
    def _edit_selected(self):
        """Edit the currently selected surface."""
        if self.selected_index is None:
            messagebox.showinfo("Info", "Seleziona una superficie dalla lista")
            return
        self._on_surface_edit(self.selected_index)
    
    def _edit_surface_at(self, index):
        """Open edit dialog for surface at given index."""
        surf = self.surfaces[index]
        if surf.get("type") == "lens":
            def on_save(lens_def):
                lens_def["type"] = "lens"
                self.surfaces[index] = lens_def
                self._refresh_surface_list()
                if self.auto_update_var.get():
                    self._schedule_auto_update()
            LensForm(self, on_save, surf)
        else:
            def on_save(surf_def):
                self.surfaces[index] = surf_def
                self._refresh_surface_list()
                if self.auto_update_var.get():
                    self._schedule_auto_update()
            SurfaceForm(self, on_save, surf)

    def _add_lens(self):
        def on_save(lens_def):
            lens_def["type"] = "lens"
            self.surfaces.append(lens_def)
            self._refresh_surface_list()
            if self.auto_update_var.get():
                self._schedule_auto_update()
        LensForm(self, on_save)

    def _add_surface(self):
        def on_save(surf_def):
            self.surfaces.append(surf_def)
            self._refresh_surface_list()
            if self.auto_update_var.get():
                self._schedule_auto_update()
        SurfaceForm(self, on_save)

    def _refresh_surface_list(self):
        """Rebuild the surface list with clickable items."""
        # Clear existing items
        for item in self.surface_items:
            item.destroy()
        self.surface_items.clear()
        
        # Create new items
        for i, surf in enumerate(self.surfaces):
            item = ClickableSurfaceItem(
                self.surface_scroll,
                index=i,
                surf_data=surf,
                on_click=self._on_surface_click,
                on_delete=self._on_surface_delete,
                on_edit=self._on_surface_edit
            )
            item.pack(fill="x", pady=2)
            self.surface_items.append(item)
        
        # Update selection
        if self.selected_index is not None and self.selected_index < len(self.surfaces):
            self.surface_items[self.selected_index].set_selected(True)
        else:
            self.selected_index = None

    # ── LED Source Panel ───────────────────────────────────────────────────────
    def _open_led_source(self):
        LEDSourcePanel(self, self._save_led_source, self.source_def)

    def _save_led_source(self, source_def):
        self.source_def = source_def
        if self.auto_update_var.get():
            self._schedule_auto_update()

    # ── Auto-update ───────────────────────────────────────────────────────────
    def _schedule_auto_update(self):
        if not self.auto_update_var.get():
            return
        if not self._update_scheduled:
            self._update_scheduled = True
            self.after(int(self._update_delay * 1000), self._process_auto_update)

    def _process_auto_update(self):
        self._update_scheduled = False
        self._run_simulation()

    def _reset_view(self):
        for ax in [self.ax_system, self.ax_illum, self.ax_iso, self.ax_lee]:
            ax.relim()
            ax.autoscale()
        self.canvas.draw_idle()
        self.canvas_illum.draw_idle()
        self.canvas_iso.draw_idle()
        self.canvas_lee.draw_idle()

    # ── Simulation ─────────────────────────────────────────────────────────────
    def _run_simulation(self):
        if not self.surfaces:
            self.stats_label.configure(text="Definisci almeno una superficie", text_color="#ff6b6b")
            return

        try:
            raw_surfaces = []
            self._lens_fill_data = []

            for s in self.surfaces:
                if s.get("type") == "lens":
                    fd, rd = build_lens_surface_defs(dict(s))
                    for surf_def in (fd, rd):
                        for obj in build_surface_objects(surf_def):
                            raw_surfaces.append(obj)

                    r_max = float(s["r_max"])
                    n_pts = int(s.get("n_points", 80))
                    mat = s.get("material", "PMMA (n=1.49)")
                    color = MATERIAL_FILL_COLOR.get(mat, "#90c8f0")

                    fg = s["fronte_geom"]
                    if fg.get("geom_type", "conic") == "conic" and float(fg.get("R", 0)) != 0:
                        f_pts = build_conic_profile(
                            vertex=fd["vertex"], R=fd["R"], k=fd.get("k", 0),
                            r_max=r_max, coeffs=fd.get("coeffs", ()),
                            n_points=n_pts, flip_z=fd.get("flip_z", False))
                    else:
                        f_pts = np.array([[fd["p1"][0], 0], [fd["p2"][0], r_max]])

                    rg = s["retro_geom"]
                    if rg.get("geom_type", "conic") == "conic" and float(rg.get("R", 0)) != 0:
                        r_pts = build_conic_profile(
                            vertex=rd["vertex"], R=rd["R"], k=rd.get("k", 0),
                            r_max=r_max, coeffs=rd.get("coeffs", ()),
                            n_points=n_pts, flip_z=rd.get("flip_z", True))
                    else:
                        r_pts = np.array([[rd["p1"][0], 0], [rd["p2"][0], r_max]])

                    self._lens_fill_data.append((f_pts, r_pts, color))
                else:
                    for obj in build_surface_objects(s):
                        raw_surfaces.append(obj)

            system = _build_symmetric_system(raw_surfaces)

            self._plot_system = OpticalSystem()
            for surf in raw_surfaces:
                self._plot_system.add(surf)

            medium = MATERIAL_PRESETS.get(self.source_def["medium"], AIR)
            source = LEDSource(
                position=self.source_def["position"],
                axis_deg=self.source_def["axis_deg"],
                half_angle_deg=self.source_def["half_angle_deg"],
                n_rays=self.source_def["n_rays"],
                distribution=self.source_def["distribution"],
                medium=medium,
                les_shape=self.source_def.get("les_shape", "point"),
                les_size=self.source_def.get("les_size", 0.0),
                n_les=self.source_def.get("les_n", 1),
            )

            base_rays = source.generate_rays()
            self._last_source = source

            max_b = int(self.max_bounces.get())
            min_p = float(self.min_power.get())
            refl_len = float(self.refl_length.get())

            traces = system.trace_many(base_rays, max_bounces=max_b, min_power=min_p)
            total_in = sum(r.power for r in base_rays)

        except Exception as exc:
            messagebox.showerror("Errore simulazione", str(exc))
            import traceback
            traceback.print_exc()
            return

        self._last_traces = traces
        self._last_total_power = total_in

        self._redraw_plots(refl_len)

        total_hit = sum(t["power"] for t in traces if t["hits"])
        pct = 100 * total_hit / total_in if total_in else 0.0
        src = getattr(self, "_last_source", None)
        les_txt = f"  |  LES: {src.les_description()}" if src else ""

        self.stats_label.configure(
            text=f"Raggi: {len(base_rays)}  |  Rami: {len(traces)}  |  η: {pct:.1f}%{les_txt}",
            text_color="#58a6ff")

    def _redraw_plots(self, reflected_length=20.0):
        # Ray trace
        self.ax_system.clear()
        self._style_ax(self.ax_system)
        self.ax_system.set_title("OpenTIR – ray trace")

        plot_system(self._plot_system, self._last_traces, ax=self.ax_system,
                   reflected_length=reflected_length, linewidth_power=True)
        self._draw_lens_fills(self.ax_system)

        # Illuminance
        self.ax_illum.clear()
        self._style_ax(self.ax_illum)
        self.ax_illum.set_title("Illuminance distribution")

        targets = [s["name"] for s in self.surfaces if s.get("kind") == "target"]
        _, eff = plot_illuminance(
            self._last_traces,
            target_name=targets[0] if targets else None,
            ax=self.ax_illum)

        # Radial distribution
        self.ax_iso.clear()
        self._style_ax(self.ax_iso)
        self.ax_iso.set_title("Radial distribution")

        r_hits = []
        powers = []
        for trace in self._last_traces:
            for surf, point in trace["hits"]:
                if not targets or surf.name == targets[0]:
                    r_hits.append(abs(point[1]))
                    powers.append(trace["power"])

        if r_hits:
            r_hits = np.array(r_hits)
            powers = np.array(powers)
            self.ax_iso.bar(r_hits, powers, width=0.5, alpha=0.7, color="orange")
            self.ax_iso.set_xlabel("r [mm]")
            self.ax_iso.set_ylabel("Power")

        # LEE breakdown
        self.ax_lee.clear()
        self._style_ax(self.ax_lee)
        self.ax_lee.set_title("LEE Breakdown")

        targets = [s["name"] for s in self.surfaces if s.get("kind") == "target"]
        self._last_lee = compute_lee(
            self._last_traces, self._last_total_power,
            target_name=targets[0] if targets else None)

        plot_lee_pie(self._last_lee, ax=self.ax_lee, dark=True)

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw_idle()
        self.fig_illum.tight_layout()
        self.canvas_illum.draw_idle()
        self.fig_iso.tight_layout()
        self.canvas_iso.draw_idle()
        self.fig_lee.tight_layout()
        self.canvas_lee.draw_idle()

    def _draw_lens_fills(self, ax):
        if not hasattr(self, "_lens_fill_data"):
            return
        for f_pts, r_pts, color in self._lens_fill_data:
            upper_z = np.concatenate([f_pts[:, 0], r_pts[::-1, 0]])
            upper_r = np.concatenate([f_pts[:, 1], r_pts[::-1, 1]])
            ax.fill(upper_z, upper_r, color=color, alpha=LENS_FILL_ALPHA, zorder=1)
            ax.fill(upper_z, -upper_r, color=color, alpha=LENS_FILL_ALPHA, zorder=1)

    # ── Event handlers ─────────────────────────────────────────────────────────
    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        factor = 1.15 if event.button == "down" else 1/1.15
        xd, yd = event.xdata, event.ydata
        if xd is None or yd is None:
            return
        ax.set_xlim([xd + (x - xd) * factor for x in ax.get_xlim()])
        ax.set_ylim([yd + (y - yd) * factor for y in ax.get_ylim()])
        self.canvas.draw_idle()

    def _on_right_click(self, event):
        if event.button != 3 or event.inaxes is None:
            return
        event.inaxes.relim()
        event.inaxes.autoscale()
        self.canvas.draw_idle()

    # ─ Export and analysis ────────────────────────────────────────────────────
    def _export_dxf(self):
        if not self._last_traces:
            messagebox.showwarning("Attenzione", "Esegui prima una simulazione.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF Drawing", "*.dxf")])
        if not path:
            return

        try:
            refl_len = float(self.refl_length.get())
            lens_fill = getattr(self, "_lens_fill_data", None)
            if lens_fill:
                lens_fill = [(f, r, c, f"lente_{i}")
                            for i, (f, r, c) in enumerate(lens_fill)]

            export_dxf(
                path=path,
                system=self._plot_system,
                traces=self._last_traces,
                lens_fill_data=lens_fill,
                reflected_length=refl_len,
                symmetric=True,
                title="OpenTIR – exported optical system")

            messagebox.showinfo("Esportato", f"File DXF salvato in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Errore export DXF", str(exc))

    def _show_isophote(self):
        if not self._last_traces:
            messagebox.showwarning("Attenzione", "Esegui prima una simulazione.")
            return

        targets = [s["name"] for s in self.surfaces if s.get("kind") == "target"]
        r_hits, powers = get_hit_points(
            self._last_traces,
            target_name=targets[0] if targets else None)

        if len(r_hits) < 3:
            messagebox.showwarning("Dati insufficienti", "Nessun raggio ha raggiunto il target.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Isofote – distribuzione di illuminamento")
        win.geometry("700x600")

        fig_iso = Figure(figsize=(7, 5.5), facecolor="#1e1e1e", dpi=100)
        ax_iso = fig_iso.add_subplot(1, 1, 1)
        ax_iso.set_facecolor("#252526")
        ax_iso.tick_params(colors="white")
        ax_iso.xaxis.label.set_color("white")
        ax_iso.yaxis.label.set_color("white")
        ax_iso.title.set_color("white")

        n_az = 64
        thetas = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
        x_all, y_all, w_all = [], [], []
        for r, p in zip(r_hits, powers):
            xs = r * np.cos(thetas)
            ys = r * np.sin(thetas)
            x_all.extend(xs)
            y_all.extend(ys)
            w_all.extend([p / n_az] * n_az)

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        w_all = np.array(w_all)

        lim = max(abs(r_hits).max() * 1.05, 1.0)
        n_bins = 60
        edges = np.linspace(-lim, lim, n_bins + 1)
        H, xedg, yedg = np.histogram2d(x_all, y_all, bins=edges, weights=w_all)
        xc = 0.5 * (xedg[:-1] + xedg[1:])
        yc = 0.5 * (yedg[:-1] + yedg[1:])
        XX, YY = np.meshgrid(xc, yc)
        ZZ = gaussian_filter(H.T, sigma=1.5)

        cf = ax_iso.contourf(XX, YY, ZZ, levels=12, cmap="inferno")
        cs = ax_iso.contour(XX, YY, ZZ, levels=12, colors="white", linewidths=0.5, alpha=0.6)
        ax_iso.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"{v:.2e}", colors="white")

        fig_iso.colorbar(cf, ax=ax_iso, pad=0.02).set_label("Illuminamento [a.u.]", color="white")
        ax_iso.set_xlabel("x [mm]", color="white")
        ax_iso.set_ylabel("y [mm]", color="white")
        ax_iso.set_title("Isofote", color="white")
        ax_iso.set_aspect("equal")

        canvas_iso = FigureCanvasTkAgg(fig_iso, master=win)
        canvas_iso.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        NavigationToolbar2Tk(canvas_iso, win, pack_toolbar=False).pack(side="bottom", fill="x", padx=10)

    def _show_lee(self):
        if not self._last_traces or self._last_lee is None:
            messagebox.showwarning("Attenzione", "Esegui prima una simulazione.")
            return

        win = ctk.CTkToplevel(self)
        win.title("LEE Breakdown")
        win.geometry("700x500")

        fig_lee = Figure(figsize=(7, 4.5), facecolor="#1e1e1e", dpi=100)
        ax_pie = fig_lee.add_subplot(1, 2, 1)
        ax_bar = fig_lee.add_subplot(1, 2, 2)
        ax_pie.set_facecolor("#252526")
        ax_bar.set_facecolor("#252526")

        plot_lee_pie(self._last_lee, ax=ax_pie, dark=True)
        plot_lee_bar(self._last_lee, ax=ax_bar, dark=True)
        fig_lee.tight_layout(pad=2.0)

        canvas_lee = FigureCanvasTkAgg(fig_lee, master=win)
        canvas_lee.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        NavigationToolbar2Tk(canvas_lee, win, pack_toolbar=False).pack(side="bottom", fill="x", padx=10)

    # ── Project management ─────────────────────────────────────────────────────
    def _save_project(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("OpenTIR project", "*.json")])
        if not path:
            return

        try:
            data = {
                "surfaces": self.surfaces,
                "source": self.source_def
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Salvato", f"Progetto salvato in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Errore", str(exc))

    def _load_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("OpenTIR project", "*.json")])
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.surfaces = data.get("surfaces", [])
            self.source_def = data.get("source", self.source_def)
            self.selected_index = None

            self._refresh_surface_list()

            if self.auto_update_var.get():
                self._schedule_auto_update()

            messagebox.showinfo("Caricato", f"Progetto caricato da:\n{path}")
        except Exception as exc:
            messagebox.showerror("Errore", f"Impossibile caricare il progetto:\n{exc}")


def main():
    app = OpenTIRApp()
    app.mainloop()


if __name__ == "__main__":
    main()