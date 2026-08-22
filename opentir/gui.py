"""
opentir.gui
~~~~~~~~~~~~
Desktop GUI — CustomTkinter, release 0.5.

Modern dark-mode interface built with CustomTkinter (ctk).
All optical logic is unchanged; only the UI layer has been rewritten.

Tabs:
  - Sistema ottico : surface editor + LED source definition
  - Simulazione    : ray-trace plot, illuminance histogram, chromatic
                     aberration, zoom/pan, radial filter + efficiency
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .geometry import Segment, Arc
from .materials import Material, AIR, PMMA, POLYCARBONATE, BK7_GLASS, SODA_LIME_GLASS
from .optics import Surface, OpticalSystem
from .source import LEDSource
from .visualize import plot_system, plot_illuminance
from .profiles import build_conic_profile, build_freeform_profile, profile_to_surfaces
from .chromatic import wavelength_samples, wavelength_to_rgb, chromatic_rays
from .export_dxf import export_dxf, get_hit_points
from .lee import compute_lee, plot_lee_pie, plot_lee_bar

# ── appearance ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_LABEL  = ("Segoe UI", 12)
FONT_BOLD   = ("Segoe UI", 12, "bold")
FONT_SMALL  = ("Segoe UI", 11)
FONT_TITLE  = ("Segoe UI", 13, "bold")
ACCENT      = "#1f6aa5"
BG_CARD     = "#2b2b2b"
BG_DARK     = "#1e1e1e"

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

# colour fill used when drawing a lens cross-section
MATERIAL_FILL_COLOR = {
    "Aria (n=1.00)":           "#888888",
    "PMMA (n=1.49)":           "#4ab0f0",   # azzurro
    "Policarbonato (n=1.585)": "#b070e0",   # viola
    "Vetro BK7 (n=1.517)":    "#40c880",   # verde
    "Vetro soda-lime (n=1.52)":"#d0c040",   # giallo
    "Personalizzato...":       "#909090",
}
LENS_FILL_ALPHA = 0.35

CONIC_PRESETS = {
    "Sfera (k=0)":     0.0,
    "Parabola (k=-1)": -1.0,
    "Iperbole (k=-2)": -2.0,
    "Ellisse (k=-0.5)": -0.5,
    "Personalizzato...": None,
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _lbl(parent, text, font=None, **kw):
    return ctk.CTkLabel(parent, text=text, font=font or FONT_LABEL, **kw)


def _entry(parent, textvariable, width=100, **kw):
    return ctk.CTkEntry(parent, textvariable=textvariable,
                        width=width, font=FONT_SMALL, **kw)


def _btn(parent, text, command, width=160, **kw):
    return ctk.CTkButton(parent, text=text, command=command,
                         width=width, font=FONT_LABEL, **kw)


def _combo(parent, variable, values, width=160, state="readonly", **kw):
    return ctk.CTkComboBox(parent, variable=variable, values=values,
                           width=width, font=FONT_SMALL,
                           state=state, **kw)


def _preset_or_custom(surf_def, key):
    name = surf_def.get(key, "Aria (n=1.00)")
    if name == "Personalizzato...":
        n = surf_def.get(f"{key}_n", 1.0)
        return Material(n=n, name=f"n={n}")
    return MATERIAL_PRESETS.get(name, AIR)


# ── lens geometry helpers ────────────────────────────────────────────────────

def _sag_at_r(surf_def, r):
    """Return the sag z(r) for a surface_def at a given r (mm).
    Supports geom_type: segment (flat), conic."""
    gt = surf_def["geom_type"]
    if gt == "segment":
        # flat surface: z is constant (p1[0] == p2[0])
        return surf_def["p1"][0]
    elif gt == "conic":
        from .profiles import conic_sag
        vz   = surf_def["vertex"][0]
        R    = surf_def["R"]
        k    = surf_def["k"]
        coef = surf_def.get("coeffs", ())
        flip = surf_def.get("flip_z", False)
        s    = float(conic_sag(np.array([r]), R, k, coef)[0])
        return vz + (-s if flip else s)
    else:
        return 0.0


def build_lens_surface_defs(lens_def):
    """
    Expand a lens_def dict into two surf_def dicts (fronte, retro).

    lens_def keys:
      name, material, r_max, n_points,
      fronte_geom  (dict with geom_type + shape params, WITHOUT vertex z)
      retro_geom   (dict with geom_type + shape params, WITHOUT vertex z)
      origin_z     : z of fronte vertex on axis
      t_edge       : thickness at r = r_max (> 0, user-supplied)

    The retro vertex z is computed so that the edge thickness is exactly
    t_edge: z_retro_v = origin_z + sag_fronte(r_max) + t_edge - sag_retro_at_0_origin
    Then t_center = z_retro_vertex - origin_z is calculated and stored.
    """
    r_max    = float(lens_def["r_max"])
    mat_name = lens_def["material"]
    mat_obj  = MATERIAL_PRESETS.get(mat_name, PMMA)
    origin_z = float(lens_def["origin_z"])
    t_edge   = float(lens_def["t_edge"])
    n_pts    = int(lens_def.get("n_points", 80))

    # ── build fronte surf_def (preliminary pass for sag calculation) ──────────
    fg = dict(lens_def["fronte_geom"])
    fg["geom_type"] = fg.get("geom_type", "conic")
    if fg["geom_type"] == "conic":
        fg.setdefault("vertex", [origin_z, 0.0])
        fg["vertex"] = [origin_z, 0.0]
        fg.setdefault("coeffs", [0.0, 0.0])
        fg.setdefault("outward", "+z")
        fg.setdefault("n_points", n_pts)
        fg["r_max"] = r_max
        # auto-flip dalla fronte se R negativo
        R_f_raw = fg.get("R", 0.0)
        if R_f_raw < 0:
            fg["flip_z"] = True
            fg["R"]      = abs(R_f_raw)
        else:
            fg.setdefault("flip_z", False)
    elif fg["geom_type"] == "segment":
        fg["p1"] = [origin_z, -r_max]
        fg["p2"] = [origin_z,  r_max]

    # sag of fronte at the edge
    sag_f_edge = _sag_at_r(fg, r_max)
    sag_f_0    = origin_z

    # ── build retro surf_def ─────────────────────────────────────────────────
    rg = dict(lens_def["retro_geom"])
    rg["geom_type"] = rg.get("geom_type", "conic")
    rg.setdefault("coeffs", [0.0, 0.0])
    rg.setdefault("n_points", n_pts)
    rg["r_max"] = r_max

    if rg["geom_type"] == "conic":
        from .profiles import conic_sag
        R_r_raw = rg.get("R", 15.0)
        k_r     = rg.get("k", 0.0)
        coef_r  = rg.get("coeffs", [0.0, 0.0])

        # Auto-flip: R negativo → curva verso -z (flip=True), indipendente
        # dal campo flip_z che ora è gestito automaticamente dal segno di R.
        # Manteniamo anche la compatibilità col vecchio formato (flip_z esplicito).
        if R_r_raw < 0:
            flip_r = True
            R_r    = abs(R_r_raw)
        else:
            flip_r = rg.get("flip_z", True)   # backward compat
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

    # ── t_center (read-only, for display) ────────────────────────────────────
    t_center = z_retro_v - sag_f_0
    lens_def["_t_center"] = round(t_center, 4)

    # ── assemble surf_defs ────────────────────────────────────────────────────
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
    fronte_def.update({k: v for k, v in fg.items()
                       if k not in ("geom_type",)})

    retro_def = {
        "name":         f"{lens_def['name']}_retro",
        "kind":         "refract",
        "geom_type":    rg["geom_type"],
        "material_in":  mat_name,
        "material_out": "Aria (n=1.00)",
        "origin_r":     origin_r,
        "rotation_deg": rotation_deg,
    }
    retro_def.update({k: v for k, v in rg.items()
                      if k not in ("geom_type",)})

    return fronte_def, retro_def


# ═══════════════════════════════════════════════════════════════════════════════
# LensForm — dialog-card for creating/editing a compound lens entity
# ═══════════════════════════════════════════════════════════════════════════════

class LensForm(ctk.CTkToplevel):
    """Modal window for defining a complete lens (fronte + retro)."""

    GEOM_OPTIONS = ["conic", "segment"]

    def __init__(self, master, on_save, lens_def=None):
        super().__init__(master)
        self.title("Definizione lente")
        self.geometry("780x580")
        self.resizable(False, False)
        self.grab_set()          # modal
        self.on_save  = on_save
        self._result  = None

        # ── variables ─────────────────────────────────────────────────────────
        self.lens_name  = tk.StringVar(value="lente1")
        self.mat_var    = tk.StringVar(value="PMMA (n=1.49)")
        self.origin_z   = tk.StringVar(value="5.0")
        self.origin_r   = tk.StringVar(value="0.0")
        self.rotation   = tk.StringVar(value="0.0")
        self.r_max      = tk.StringVar(value="10.0")
        self.t_edge     = tk.StringVar(value="1.0")
        self.n_points   = tk.StringVar(value="80")
        self.t_center_display = tk.StringVar(value="—")

        # fronte geometry
        self.f_geom = tk.StringVar(value="conic")
        self.f_R    = tk.StringVar(value="0")       # 0 = flat (segment)
        self.f_k    = tk.StringVar(value="0.0")
        self.f_A4   = tk.StringVar(value="0.0")
        self.f_A6   = tk.StringVar(value="0.0")

        # retro geometry
        self.r_geom = tk.StringVar(value="conic")
        self.r_R    = tk.StringVar(value="-15.0")   # negativo = flip automatico
        self.r_k    = tk.StringVar(value="0.0")
        self.r_A4   = tk.StringVar(value="0.0")
        self.r_A6   = tk.StringVar(value="0.0")
        self.r_flip = tk.BooleanVar(value=True)

        if lens_def:
            self._populate(lens_def)

        # ── layout ────────────────────────────────────────────────────────────
        self._build_ui()

        # live update of t_center whenever any param changes
        for v in (self.origin_z, self.origin_r, self.rotation, self.r_max, self.t_edge,
                  self.f_R, self.f_k, self.f_A4, self.f_A6,
                  self.r_R, self.r_k, self.r_A4, self.r_A6):
            v.trace_add("write", self._update_t_center)
        self.r_flip.trace_add("write", self._update_t_center)
        self._update_t_center()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ── header row ────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        _lbl(top, "Nome lente:").grid(row=0, column=0, **pad, sticky="w")
        _entry(top, self.lens_name, width=130).grid(row=0, column=1, **pad)

        _lbl(top, "Materiale:").grid(row=0, column=2, **pad, sticky="w")
        mat_names = [k for k in MATERIAL_PRESETS if k != "Personalizzato..."]
        _combo(top, self.mat_var, mat_names, width=200).grid(
            row=0, column=3, **pad)

        _lbl(top, "r_max [mm]:").grid(row=0, column=4, **pad, sticky="w")
        _entry(top, self.r_max, width=65).grid(row=0, column=5, **pad)

        _lbl(top, "N. punti:").grid(row=0, column=6, **pad, sticky="w")
        _entry(top, self.n_points, width=55).grid(row=0, column=7, **pad)

        # ── dimensions row ────────────────────────────────────────────────────
        dim = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        dim.pack(fill="x", padx=10, pady=4)

        _lbl(dim, "Origine z [mm]").grid(row=0, column=0, **pad, sticky="w")
        _entry(dim, self.origin_z, width=70).grid(row=0, column=1, **pad)

        _lbl(dim, "Origine r [mm]").grid(row=0, column=2, **pad, sticky="w")
        _entry(dim, self.origin_r, width=70).grid(row=0, column=3, **pad)

        _lbl(dim, "Rotazione [\u00b0]").grid(row=0, column=4, **pad, sticky="w")
        _entry(dim, self.rotation, width=70).grid(row=0, column=5, **pad)

        _lbl(dim, "Spessore bordo [mm]").grid(row=1, column=0, **pad, sticky="w")
        _entry(dim, self.t_edge, width=70).grid(row=1, column=1, **pad)

        _lbl(dim, "Spessore centro [mm]").grid(row=1, column=2, **pad, sticky="w")
        self._tc_lbl = _lbl(dim, "\u2014", font=FONT_BOLD, text_color="#58a6ff")
        self._tc_lbl.grid(row=1, column=3, **pad)
        # ── surfaces panel ────────────────────────────────────────────────────
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

        # ── buttons ───────────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(4, 10))
        _btn(btn_row, "💾  Salva lente", self._save,
             width=160).pack(side="left", padx=6)
        _btn(btn_row, "✕  Annulla", self.destroy,
             width=120, fg_color="gray30",
             hover_color="gray40").pack(side="left", padx=4)

    def _build_surface_card(self, parent, side, title,
                             geom_var, R_var, k_var, A4_var, A6_var, flip_var):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        card.pack(side=side, fill="both", expand=True,
                  padx=(0 if side=="right" else 0, 0), pady=4)
        card.pack_configure(padx=(4,4))

        _lbl(card, title, font=FONT_BOLD).grid(
            row=0, column=0, columnspan=4, padx=8, pady=(8,4), sticky="w")

        _lbl(card, "Tipo:", font=FONT_SMALL).grid(
            row=1, column=0, padx=8, pady=3, sticky="w")
        _combo(card, geom_var, self.GEOM_OPTIONS, width=100,
               command=lambda v: None).grid(row=1, column=1, padx=4)

        for i, (label, var) in enumerate([
            ("R [mm]\n(+ →+z, − →-z, 0=piano):", R_var),
            ("k:", k_var),
            ("A4:", A4_var),
            ("A6:", A6_var),
        ]):
            r = 2 + i
            _lbl(card, label, font=FONT_SMALL).grid(
                row=r, column=0, padx=8, pady=3, sticky="w")
            _entry(card, var, width=100).grid(row=r, column=1, padx=4)

        if flip_var is not None:
            # checkbox rimosso: il flip è automatico dal segno di R
            pass

    def _make_geom_dict(self, geom_var, R_var, k_var, A4_var, A6_var,
                         flip_var=None):
        """Build a geometry sub-dict for build_lens_surface_defs.

        If R < 0 the surface curves toward -z: we store abs(R) and
        set flip_z=True automatically.  The user never needs to set
        flip manually — just enter a negative R value.
        """
        gtype = geom_var.get()
        R_val = float(R_var.get())
        if gtype == "segment" or R_val == 0:
            return {"geom_type": "segment"}
        # auto-flip from negative R
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
                self.t_center_display.set(txt)
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
                self.r_geom, self.r_R, self.r_k, self.r_A4, self.r_A6,
                self.r_flip),
        }

    def _save(self):
        try:
            t_edge = float(self.t_edge.get())
            if t_edge <= 0:
                raise ValueError("Lo spessore sul bordo deve essere > 0")
            lens_def = self._make_lens_def()
            build_lens_surface_defs(lens_def)   # validate geometry
            if lens_def.get("_t_center", 1) <= 0:
                raise ValueError(
                    "Lo spessore al centro risulta ≤ 0: aumenta t_edge "
                    "o riduci la curvatura")
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
        self.f_R.set(str(fg.get("R", 0)))
        self.f_k.set(str(fg.get("k", 0.0)))
        if fg.get("coeffs"):
            self.f_A4.set(str(fg["coeffs"][0]))
            self.f_A6.set(str(fg["coeffs"][1]))
        rg = lens_def.get("retro_geom", {})
        self.r_geom.set(rg.get("geom_type", "conic"))
        # riconverti flip_z→segno di R: se flip_z=True il valore salvato
        # potrebbe essere positivo (formato vecchio) — negate per la nuova convenzione
        r_R_val = rg.get("R", 15.0)
        r_flip  = rg.get("flip_z", True)
        if r_flip and r_R_val > 0:
            r_R_val = -r_R_val   # converti vecchio formato al nuovo
        self.r_R.set(str(r_R_val))
        self.r_k.set(str(rg.get("k", 0.0)))
        if rg.get("coeffs"):
            self.r_A4.set(str(rg["coeffs"][0]))
            self.r_A6.set(str(rg["coeffs"][1]))


def _mirror_surface_object(surf):
    """
    Return the r→-r mirror of a Surface whose geometry is a Segment.
    Only applied to 'target' surfaces that span r≥0 so that hits on
    both halves of the target are registered.
    For refractive/mirror surfaces the user must define the full r range
    (from -r_max to +r_max) explicitly in the GUI.
    Returns None if not applicable.
    """
    geom = surf.geometry
    if not isinstance(geom, Segment):
        return None
    # Only mirror target and block surfaces for hit detection symmetry
    if surf.kind not in ("target", "block"):
        return None
    p1, p2 = geom.p1, geom.p2
    # if already spans r<0 don't duplicate
    if min(p1[1], p2[1]) < -1e-9:
        return None
    mp1 = np.array([p1[0], -p1[1]])
    mp2 = np.array([p2[0], -p2[1]])
    mirrored_geom = Segment(mp1, mp2, name=geom.name)
    return Surface(mirrored_geom, kind=surf.kind, name=surf.name,
                   material_in=surf.material_in, material_out=surf.material_out)


def _build_symmetric_system(surfaces_in):
    """
    Given a list of Surface objects, return an OpticalSystem where target
    and block surfaces defined only for r≥0 are duplicated for r≤0.
    Refractive and mirror surfaces must be defined by the user with the
    full r range (the GUI SurfaceForm supports negative r values).
    """
    system = OpticalSystem()
    for surf in surfaces_in:
        system.add(surf)
        mirror = _mirror_surface_object(surf)
        if mirror is not None:
            system.add(mirror)
    return system


def build_surface_objects(surf_def):
    """Turn a GUI surface dict into a list of opentir Surface instances."""
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
        # Full symmetric profile: apply offset (origin_r) and rotation if set
        import numpy as _np
        _origin_r = surf_def.get("origin_r", 0.0)
        _rot_deg  = surf_def.get("rotation_deg", 0.0)
        # build symmetric around local axis then translate+rotate
        points = _np.vstack([pts_half[::-1] * [1, -1], pts_half[1:]])
        if _origin_r != 0.0 or _rot_deg != 0.0:
            _c, _s = _np.cos(_np.radians(_rot_deg)), _np.sin(_np.radians(_rot_deg))
            # translate r by origin_r then rotate in (z,r) plane
            pts_t = points.copy()
            pts_t[:, 1] += _origin_r
            zz, rr = pts_t[:, 0], pts_t[:, 1]
            points = _np.column_stack([zz * _c - rr * _s,
                                       zz * _s + rr * _c])
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
        # Full symmetric profile: apply offset (origin_r) and rotation if set
        import numpy as _np
        _origin_r = surf_def.get("origin_r", 0.0)
        _rot_deg  = surf_def.get("rotation_deg", 0.0)
        # build symmetric around local axis then translate+rotate
        points = _np.vstack([pts_half[::-1] * [1, -1], pts_half[1:]])
        if _origin_r != 0.0 or _rot_deg != 0.0:
            _c, _s = _np.cos(_np.radians(_rot_deg)), _np.sin(_np.radians(_rot_deg))
            # translate r by origin_r then rotate in (z,r) plane
            pts_t = points.copy()
            pts_t[:, 1] += _origin_r
            zz, rr = pts_t[:, 0], pts_t[:, 1]
            points = _np.column_stack([zz * _c - rr * _s,
                                       zz * _s + rr * _c])
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


# ═══════════════════════════════════════════════════════════════════════════════
# SurfaceForm — card that lives inside the System tab
# ═══════════════════════════════════════════════════════════════════════════════
class SurfaceForm(ctk.CTkFrame):

    def __init__(self, master, on_save):
        super().__init__(master, fg_color=BG_CARD, corner_radius=10)
        self.on_save       = on_save
        self.editing_index = None
        self.freeform_text = None

        # ── row 0: name / kind / geometry ────────────────────────────────────
        r = 0
        _lbl(self, "Nome").grid(row=r, column=0, padx=8, pady=4, sticky="w")
        self.name_var = tk.StringVar(value="superficie1")
        _entry(self, self.name_var, width=160).grid(row=r, column=1, padx=4)

        _lbl(self, "Tipo").grid(row=r, column=2, padx=8, sticky="w")
        self.kind_var = tk.StringVar(value="mirror")
        _combo(self, self.kind_var, SURFACE_KINDS, width=120,
               command=lambda v: self._update_material()).grid(row=r, column=3, padx=4)

        _lbl(self, "Geometria").grid(row=r, column=4, padx=8, sticky="w")
        self.geom_var = tk.StringVar(value="segment")
        _combo(self, self.geom_var, GEOM_TYPES, width=120,
               command=lambda v: self._update_geom()).grid(row=r, column=5, padx=4)

        # ── geometry fields frame ─────────────────────────────────────────────
        r += 1
        self.geom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.geom_frame.grid(row=r, column=0, columnspan=6, sticky="ew", padx=8, pady=4)

        # ── material fields frame ─────────────────────────────────────────────
        r += 1
        self.mat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.mat_frame.grid(row=r, column=0, columnspan=6, sticky="ew", padx=8, pady=4)

        # ── buttons ──────────────────────────────────────────────────────────
        r += 1
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=r, column=0, columnspan=6, sticky="w", padx=8, pady=8)
        _btn(btn_row, "✦  Nuova", self.clear, width=120).pack(side="left", padx=4)
        _btn(btn_row, "💾  Salva superficie", self._save, width=180).pack(side="left", padx=4)

        # ── internal field variables ──────────────────────────────────────────
        self.seg_vars = {k: tk.StringVar(value=v) for k, v in
                         [("z1","0.0"),("r1","-10.0"),("z2","10.0"),("r2","10.0")]}
        self.arc_vars = {k: tk.StringVar(value=v) for k, v in
                         [("cz","0.0"),("cr","0.0"),("radius","10.0"),
                          ("theta1","0.0"),("theta2","90.0")]}
        self.conic_vars = {k: tk.StringVar(value=v) for k, v in
                           [("vz","0.0"),("vr","0.0"),("R","10.0"),("k","-1.0"),
                            ("rmax","10.0"),("A4","0.0"),("A6","0.0"),("npoints","80")]}
        self.conic_preset_var = tk.StringVar(value="Parabola (k=-1)")
        self.conic_flip_var   = tk.BooleanVar(value=False)
        self.conic_outward_var = tk.StringVar(value="+z")
        self.freeform_outward_var = tk.StringVar(value="+z")

        self.mat_in_var     = tk.StringVar(value="PMMA (n=1.49)")
        self.mat_in_custom  = tk.StringVar(value="1.50")
        self.mat_out_var    = tk.StringVar(value="Aria (n=1.00)")
        self.mat_out_custom = tk.StringVar(value="1.00")

        self._update_geom()
        self._update_material()

    # ── geometry fields ───────────────────────────────────────────────────────
    def _update_geom(self, *_):
        for w in self.geom_frame.winfo_children():
            w.destroy()
        gt = self.geom_var.get()
        if gt == "segment":
            for i, (k, lbl) in enumerate([("z1","z1"),("r1","r1"),("z2","z2"),("r2","r2")]):
                _lbl(self.geom_frame, f"{lbl} [mm]", font=FONT_SMALL).grid(
                    row=0, column=2*i, padx=(8,2), sticky="w")
                _entry(self.geom_frame, self.seg_vars[k], width=72).grid(
                    row=0, column=2*i+1, padx=(0,6))
            _lbl(self.geom_frame,
                 "ℹ  Per simmetria usa r1<0 e r2>0  (es. r1=-10, r2=+10)",
                 font=("Segoe UI", 10), text_color="#aaaaaa").grid(
                row=1, column=0, columnspan=8, sticky="w", padx=8, pady=(2,0))

        elif gt == "arc":
            for i, (k, lbl) in enumerate([("cz","c.z"),("cr","c.r"),
                                           ("radius","raggio"),
                                           ("theta1","θ1 °"),("theta2","θ2 °")]):
                _lbl(self.geom_frame, lbl, font=FONT_SMALL).grid(
                    row=0, column=2*i, padx=(8,2), sticky="w")
                _entry(self.geom_frame, self.arc_vars[k], width=68).grid(
                    row=0, column=2*i+1, padx=(0,6))

        elif gt == "conic":
            _lbl(self.geom_frame, "Preset", font=FONT_SMALL).grid(
                row=0, column=0, padx=(8,2), sticky="w")
            pb = _combo(self.geom_frame, self.conic_preset_var,
                        list(CONIC_PRESETS.keys()), width=150,
                        command=self._apply_conic_preset)
            pb.grid(row=0, column=1, padx=(0,10))

            for i, (k, lbl) in enumerate([("vz","vtx z"),("vr","vtx r"),
                                           ("R","R curv."),("k","k")]):
                _lbl(self.geom_frame, lbl, font=FONT_SMALL).grid(
                    row=0, column=2+2*i, padx=(6,2), sticky="w")
                _entry(self.geom_frame, self.conic_vars[k], width=68).grid(
                    row=0, column=3+2*i, padx=(0,4))

            for i, (k, lbl) in enumerate([("rmax","r_max"),
                                           ("A4","A4"),("A6","A6"),("npoints","N pt")]):
                _lbl(self.geom_frame, lbl, font=FONT_SMALL).grid(
                    row=1, column=2*i, padx=(8,2), sticky="w", pady=(4,0))
                _entry(self.geom_frame, self.conic_vars[k], width=68).grid(
                    row=1, column=2*i+1, padx=(0,4), pady=(4,0))

            ctk.CTkCheckBox(self.geom_frame, text="Flip (curva verso -z)",
                            variable=self.conic_flip_var,
                            font=FONT_SMALL).grid(
                row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(4,0))
            _lbl(self.geom_frame, "Lato esterno", font=FONT_SMALL).grid(
                row=2, column=3, padx=(16,2), sticky="w", pady=(4,0))
            _combo(self.geom_frame, self.conic_outward_var,
                   ["+z", "-z"], width=70).grid(
                row=2, column=4, padx=(0,4), pady=(4,0))
            self._apply_conic_preset()

        else:  # freeform
            _lbl(self.geom_frame, "Punti z,r (uno per riga):",
                 font=FONT_SMALL).grid(row=0, column=0, sticky="nw", padx=8)
            self.freeform_text = ctk.CTkTextbox(
                self.geom_frame, width=220, height=100, font=FONT_SMALL)
            self.freeform_text.grid(row=0, column=1, padx=4, pady=4)
            self.freeform_text.insert("1.0", "0.0,0.0\n2.0,3.0\n5.0,6.0\n9.0,8.0")
            _lbl(self.geom_frame, "Lato esterno", font=FONT_SMALL).grid(
                row=0, column=2, padx=(16,2), sticky="w")
            _combo(self.geom_frame, self.freeform_outward_var,
                   ["+z", "-z"], width=70).grid(row=0, column=3, padx=4)

    def _apply_conic_preset(self, *_):
        v = CONIC_PRESETS.get(self.conic_preset_var.get())
        if v is not None:
            self.conic_vars["k"].set(str(v))

    # ── material fields ───────────────────────────────────────────────────────
    def _update_material(self, *_):
        for w in self.mat_frame.winfo_children():
            w.destroy()
        if self.kind_var.get() != "refract":
            return
        for row, (label, var, cvar) in enumerate([
            ("Materiale interno (lato opposto normale)",
             self.mat_in_var,  self.mat_in_custom),
            ("Materiale esterno (lato normale)",
             self.mat_out_var, self.mat_out_custom),
        ]):
            _lbl(self.mat_frame, label, font=FONT_SMALL).grid(
                row=row, column=0, sticky="w", padx=8, pady=2)
            _combo(self.mat_frame, var,
                   list(MATERIAL_PRESETS.keys()), width=200).grid(
                row=row, column=1, padx=4)
            _entry(self.mat_frame, cvar, width=60).grid(
                row=row, column=2, padx=4)

    # ── load / clear ──────────────────────────────────────────────────────────
    def clear(self):
        self.editing_index = None
        self.name_var.set(f"superficie_{np.random.randint(1000)}")

    def load(self, index, surf_def):
        self.editing_index = index
        self.name_var.set(surf_def["name"])
        self.kind_var.set(surf_def["kind"])
        self.geom_var.set(surf_def["geom_type"])
        self._update_geom()
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
        else:
            if self.freeform_text:
                self.freeform_text.delete("1.0", "end")
                lines = "\n".join(f"{p[0]},{p[1]}"
                                  for p in surf_def.get("points", []))
                self.freeform_text.insert("1.0", lines)
            self.freeform_outward_var.set(surf_def.get("outward", "+z"))
        self._update_material()
        if surf_def["kind"] == "refract":
            self.mat_in_var.set(surf_def.get("material_in",  "PMMA (n=1.49)"))
            self.mat_out_var.set(surf_def.get("material_out", "Aria (n=1.00)"))

    # ── save ──────────────────────────────────────────────────────────────────
    def _save(self):
        try:
            surf_def = {"name": self.name_var.get(),
                        "kind": self.kind_var.get(),
                        "geom_type": self.geom_var.get()}
            gt = self.geom_var.get()
            if gt == "segment":
                surf_def["p1"] = [float(self.seg_vars["z1"].get()),
                                  float(self.seg_vars["r1"].get())]
                surf_def["p2"] = [float(self.seg_vars["z2"].get()),
                                  float(self.seg_vars["r2"].get())]
            elif gt == "arc":
                surf_def["center"] = [float(self.arc_vars["cz"].get()),
                                      float(self.arc_vars["cr"].get())]
                surf_def["radius"]    = float(self.arc_vars["radius"].get())
                surf_def["theta1_deg"] = float(self.arc_vars["theta1"].get())
                surf_def["theta2_deg"] = float(self.arc_vars["theta2"].get())
            elif gt == "conic":
                surf_def["vertex"]  = [float(self.conic_vars["vz"].get()),
                                       float(self.conic_vars["vr"].get())]
                surf_def["R"]       = float(self.conic_vars["R"].get())
                surf_def["k"]       = float(self.conic_vars["k"].get())
                surf_def["r_max"]   = float(self.conic_vars["rmax"].get())
                surf_def["coeffs"]  = [float(self.conic_vars["A4"].get()),
                                       float(self.conic_vars["A6"].get())]
                surf_def["n_points"] = int(self.conic_vars["npoints"].get())
                surf_def["flip_z"]  = bool(self.conic_flip_var.get())
                surf_def["outward"] = self.conic_outward_var.get()
            else:
                if not self.freeform_text:
                    raise ValueError("Apri prima il form freeform")
                raw = self.freeform_text.get("1.0", "end").strip()
                pts = []
                for ln, line in enumerate(raw.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.replace(";", ",").split(",")]
                    if len(parts) != 2:
                        raise ValueError(f"riga {ln} non valida: '{line}'")
                    pts.append([float(parts[0]), float(parts[1])])
                if len(pts) < 2:
                    raise ValueError("Servono almeno 2 punti")
                surf_def["points"]  = pts
                surf_def["outward"] = self.freeform_outward_var.get()

            if surf_def["kind"] == "refract":
                surf_def["material_in"]  = self.mat_in_var.get()
                surf_def["material_out"] = self.mat_out_var.get()
                if self.mat_in_var.get()  == "Personalizzato...":
                    surf_def["material_in_n"]  = float(self.mat_in_custom.get())
                if self.mat_out_var.get() == "Personalizzato...":
                    surf_def["material_out_n"] = float(self.mat_out_custom.get())
        except ValueError as exc:
            messagebox.showerror("Errore", str(exc))
            return
        self.on_save(self.editing_index, surf_def)
        self.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Main application
# ═══════════════════════════════════════════════════════════════════════════════
class OpenTIRApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("OpenTIR")
        self.geometry("1280x820")

        self.surfaces: list = []
        self._last_traces      = []
        self._last_total_power = 0.0
        self._last_lee         = None
        self._last_source      = None

        # top title bar
        title_bar = ctk.CTkFrame(self, fg_color=BG_DARK, height=44)
        title_bar.pack(fill="x", side="top")
        ctk.CTkLabel(title_bar, text="  ◉  OpenTIR  —  LED Optical Simulator",
                     font=("Segoe UI", 14, "bold"),
                     text_color="#58a6ff").pack(side="left", pady=8)
        ctk.CTkLabel(title_bar, text="v0.5  ·  dark mode",
                     font=FONT_SMALL, text_color="gray").pack(side="right", padx=16)

        self.tabview = ctk.CTkTabview(self, anchor="nw",
                                      segmented_button_fg_color=BG_DARK,
                                      segmented_button_selected_color=ACCENT)
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)

        self.tabview.add("⚙  Sistema ottico")
        self.tabview.add("📡  Simulazione")
        self._build_system_tab(self.tabview.tab("⚙  Sistema ottico"))
        self._build_sim_tab(self.tabview.tab("📡  Simulazione"))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 – Sistema ottico
    # ─────────────────────────────────────────────────────────────────────────
    def _build_system_tab(self, parent):
        # left column: surface list + toolbar
        left = ctk.CTkFrame(parent, fg_color=BG_DARK, width=520)
        left.pack(side="left", fill="both", expand=True, padx=(8,4), pady=8)

        _lbl(left, "Superfici definite", font=FONT_TITLE).pack(anchor="w", pady=(4,2))

        # surface list (plain Listbox — no CTk equivalent yet)
        list_frame = ctk.CTkFrame(left, fg_color=BG_CARD, corner_radius=8)
        list_frame.pack(fill="both", expand=True, pady=(0,4))
        self.listbox = tk.Listbox(
            list_frame,
            bg="#2b2b2b", fg="white", selectbackground=ACCENT,
            selectforeground="white", activestyle="none",
            font=("Consolas", 11), relief="flat", bd=0,
            highlightthickness=0)
        self.listbox.pack(fill="both", expand=True, padx=4, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # toolbar buttons
        tb = ctk.CTkFrame(left, fg_color="transparent")
        tb.pack(fill="x", pady=(0,6))
        for text, cmd, w in [
            ("↑", self._move_up,    40),
            ("↓", self._move_down,  40),
            ("🗑 Elimina",  self._delete_selected, 110),
            ("💾 Salva progetto", self._save_project, 150),
            ("📂 Carica",  self._load_project, 100),
        ]:
            _btn(tb, text, cmd, width=w).pack(side="left", padx=3)

        # "Aggiungi lente" button on a second row
        tb2 = ctk.CTkFrame(left, fg_color="transparent")
        tb2.pack(fill="x", pady=(0,4))
        _btn(tb2, "🔭  Aggiungi lente…", self._open_lens_form,
             width=200, fg_color="#2a5a2a",
             hover_color="#3a7a3a").pack(side="left", padx=3)

        # surface form
        self.form = SurfaceForm(left, on_save=self._on_surface_saved)
        self.form.pack(fill="x", pady=4)

        # right column: LED source
        right = ctk.CTkFrame(parent, fg_color=BG_DARK, width=320)
        right.pack(side="right", fill="y", padx=(4,8), pady=8)

        src_card = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=10)
        src_card.pack(fill="x", pady=4)
        _lbl(src_card, "  Sorgente LED", font=FONT_TITLE).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8,4))

        self.src_z            = tk.StringVar(value="0.0")
        self.src_r            = tk.StringVar(value="0.0")
        self.src_axis         = tk.StringVar(value="90.0")
        self.src_half_angle   = tk.StringVar(value="80.0")
        self.src_n_rays       = tk.StringVar(value="61")
        self.src_distribution = tk.StringVar(value="lambertian")
        self.src_medium       = tk.StringVar(value="Aria (n=1.00)")
        self.src_les_shape    = tk.StringVar(value="point")
        self.src_les_size     = tk.StringVar(value="1.0")
        self.src_les_n        = tk.StringVar(value="5")

        fields = [
            ("Posizione z [mm]", self.src_z),
            ("Posizione r [mm]", self.src_r),
            ("Asse emissione [°]", self.src_axis),
            ("Semi-angolo [°]", self.src_half_angle),
            ("N. raggi angolari", self.src_n_rays),
        ]
        for i, (label, var) in enumerate(fields, start=1):
            _lbl(src_card, label, font=FONT_SMALL).grid(
                row=i, column=0, sticky="w", padx=12, pady=3)
            _entry(src_card, var, width=110).grid(row=i, column=1, padx=8, pady=3)

        r = len(fields) + 1
        _lbl(src_card, "Distribuzione", font=FONT_SMALL).grid(
            row=r, column=0, sticky="w", padx=12, pady=3)
        _combo(src_card, self.src_distribution, DISTRIBUTIONS, width=130).grid(
            row=r, column=1, padx=8, pady=3)
        r += 1
        _lbl(src_card, "Mezzo emissione", font=FONT_SMALL).grid(
            row=r, column=0, sticky="w", padx=12, pady=3)
        _combo(src_card, self.src_medium,
               [k for k in MATERIAL_PRESETS if k != "Personalizzato..."],
               width=175).grid(row=r, column=1, padx=8, pady=(3, 6))

        # ── LES section ──────────────────────────────────────────────────────
        les_card = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=10)
        les_card.pack(fill="x", pady=(6, 4))
        _lbl(les_card, "  LES (Light Emitting Surface)",
             font=FONT_TITLE).grid(row=0, column=0, columnspan=2,
                                    sticky="w", padx=8, pady=(8, 4))

        _lbl(les_card, "Forma LES", font=FONT_SMALL).grid(
            row=1, column=0, sticky="w", padx=12, pady=3)
        les_cb = _combo(les_card, self.src_les_shape,
                        ["point", "square", "circle"], width=120,
                        command=self._on_les_shape_change)
        les_cb.grid(row=1, column=1, padx=8, pady=3)

        _lbl(les_card, "Dimensione [mm]", font=FONT_SMALL).grid(
            row=2, column=0, sticky="w", padx=12, pady=3)
        self._les_size_entry = _entry(les_card, self.src_les_size, width=80)
        self._les_size_entry.grid(row=2, column=1, padx=8, pady=3)

        _lbl(les_card, "N. sotto-sorgenti", font=FONT_SMALL).grid(
            row=3, column=0, sticky="w", padx=12, pady=3)
        self._les_n_entry = _entry(les_card, self.src_les_n, width=80)
        self._les_n_entry.grid(row=3, column=1, padx=8, pady=3)

        self.src_les_total_lbl = _lbl(les_card, "Raggi totali: 61",
                                       font=FONT_SMALL, text_color="gray")
        self.src_les_total_lbl.grid(row=4, column=0, columnspan=2,
                                     sticky="w", padx=12, pady=(2, 8))

        # update total rays label when params change
        for v in (self.src_n_rays, self.src_les_n, self.src_les_shape):
            v.trace_add("write", self._update_les_total)
        self._update_les_total()
        self._on_les_shape_change("point")

    # ── lens form ─────────────────────────────────────────────────────────────
    def _open_lens_form(self, lens_def=None):
        def on_save(ld):
            build_lens_surface_defs(ld)   # populate _t_center
            if lens_def is None:
                self.surfaces.append(ld)
            else:
                idx = self.surfaces.index(lens_def)
                self.surfaces[idx] = ld
            self._refresh_list()
        LensForm(self, on_save=on_save, lens_def=lens_def)

    # ── surface list helpers ───────────────────────────────────────────────────
    def _on_surface_saved(self, index, surf_def):
        if index is None:
            self.surfaces.append(surf_def)
        else:
            self.surfaces[index] = surf_def
        self._refresh_list()

    def _refresh_list(self):
        self.listbox.delete(0, "end")
        kind_icon = {"mirror": "🪞", "target": "🎯",
                     "block": "■",   "refract": "💎", "lens": "🔭"}
        self._listbox_index_map = []   # maps listbox row → surfaces index

        for si, s in enumerate(self.surfaces):
            if s.get("type") == "lens":
                mat = s.get("material", "PMMA (n=1.49)")
                tc  = s.get("_t_center", "?")
                te  = s.get("t_edge", "?")
                self.listbox.insert(
                    "end",
                    f"  🔭  {s['name']}  [{mat}]  "
                    f"t_edge={te} mm  t_center={tc} mm")
                self.listbox.itemconfig("end", fg="#90c8f0")
                self._listbox_index_map.append((si, "lens_header"))

                # two indented child rows
                fd, rd = build_lens_surface_defs(dict(s))
                for tag, label in [("fronte", fd["name"]),
                                    ("retro",  rd["name"])]:
                    self.listbox.insert(
                        "end", f"      ├─  {label}  [{s.get('fronte_geom' if tag=='fronte' else 'retro_geom', {}).get('geom_type','conic')}]")
                    self.listbox.itemconfig("end", fg="#6aa8d0")
                    self._listbox_index_map.append((si, tag))
            else:
                icon = kind_icon.get(s.get("kind", ""), "·")
                gt   = s.get("geom_type", "")
                self.listbox.insert(
                    "end",
                    f"  {icon}  {s['name']}  [{s.get('kind','')} / {gt}]")
                self._listbox_index_map.append((si, "surface"))

    def _on_select(self, _=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        row = sel[0]
        if not hasattr(self, "_listbox_index_map") or row >= len(self._listbox_index_map):
            return
        si, tag = self._listbox_index_map[row]
        s = self.surfaces[si]
        if tag == "lens_header":
            # double-click on lens header opens the lens editor
            self._open_lens_form(lens_def=s)
        elif tag == "surface":
            self.form.load(si, s)

    def _delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        row = sel[0]
        if not hasattr(self, "_listbox_index_map") or row >= len(self._listbox_index_map):
            return
        si, _ = self._listbox_index_map[row]
        del self.surfaces[si]
        self.form.clear()
        self._refresh_list()

    def _move_up(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        row = sel[0]
        if not hasattr(self, "_listbox_index_map") or row >= len(self._listbox_index_map):
            return
        si, _ = self._listbox_index_map[row]
        if si == 0:
            return
        self.surfaces[si-1], self.surfaces[si] = self.surfaces[si], self.surfaces[si-1]
        self._refresh_list()
        # reselect the moved item's header row
        for r, (s_idx, tag) in enumerate(self._listbox_index_map):
            if s_idx == si-1 and tag in ("lens_header", "surface"):
                self.listbox.selection_set(r)
                break

    def _move_down(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        row = sel[0]
        if not hasattr(self, "_listbox_index_map") or row >= len(self._listbox_index_map):
            return
        si, _ = self._listbox_index_map[row]
        if si >= len(self.surfaces)-1:
            return
        self.surfaces[si], self.surfaces[si+1] = self.surfaces[si+1], self.surfaces[si]
        self._refresh_list()
        for r, (s_idx, tag) in enumerate(self._listbox_index_map):
            if s_idx == si+1 and tag in ("lens_header", "surface"):
                self.listbox.selection_set(r)
                break

    def _on_les_shape_change(self, value=None):
        shape = self.src_les_shape.get()
        is_point = (shape == "point")
        state = "disabled" if is_point else "normal"
        self._les_size_entry.configure(state=state)
        self._les_n_entry.configure(state=state)
        self._update_les_total()

    def _update_les_total(self, *_):
        try:
            n_ang  = int(self.src_n_rays.get())
            n_les  = int(self.src_les_n.get())
            shape  = self.src_les_shape.get()
            n_sub  = 1 if shape == "point" else n_les
            total  = n_ang * n_sub
            self.src_les_total_lbl.configure(
                text=f"Raggi totali: {total}", text_color="#58a6ff")
        except ValueError:
            self.src_les_total_lbl.configure(
                text="Raggi totali: —", text_color="gray")

    def _current_source_def(self):
        return {
            "position":       [float(self.src_z.get()), float(self.src_r.get())],
            "axis_deg":       float(self.src_axis.get()),
            "half_angle_deg": float(self.src_half_angle.get()),
            "n_rays":         int(self.src_n_rays.get()),
            "distribution":   self.src_distribution.get(),
            "medium":         self.src_medium.get(),
            "les_shape":      self.src_les_shape.get(),
            "les_size":       float(self.src_les_size.get()),
            "les_n":          int(self.src_les_n.get()),
        }

    def _save_project(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("OpenTIR project", "*.json")])
        if not path:
            return
        try:
            data = {"surfaces": self.surfaces,
                    "source":   self._current_source_def()}
        except ValueError as exc:
            messagebox.showerror("Errore", str(exc))
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Salvato", f"Progetto salvato in:\n{path}")

    def _load_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("OpenTIR project", "*.json")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.surfaces = data.get("surfaces", [])
        self._refresh_list()
        src = data.get("source", {})
        if src:
            self.src_z.set(str(src["position"][0]))
            self.src_r.set(str(src["position"][1]))
            self.src_axis.set(str(src["axis_deg"]))
            self.src_half_angle.set(str(src["half_angle_deg"]))
            self.src_n_rays.set(str(src["n_rays"]))
            self.src_distribution.set(src["distribution"])
            self.src_medium.set(src["medium"])
            self.src_les_shape.set(src.get("les_shape", "point"))
            self.src_les_size.set(str(src.get("les_size", 1.0)))
            self.src_les_n.set(str(src.get("les_n", 5)))
            self._on_les_shape_change()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 – Simulazione
    # ─────────────────────────────────────────────────────────────────────────
    def _build_sim_tab(self, parent):

        # ── top control bar ───────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8, height=48)
        ctrl.pack(fill="x", padx=8, pady=(8,4))

        self.max_bounces = tk.StringVar(value="15")
        self.min_power   = tk.StringVar(value="0.001")
        self.refl_length = tk.StringVar(value="20")

        for label, var, w in [
            ("Max rimbalzi", self.max_bounces, 50),
            ("Potenza min.", self.min_power,   70),
            ("Lungh. rimbalzi [mm]", self.refl_length, 50),
        ]:
            ctk.CTkFrame(ctrl, fg_color="transparent", width=2).pack(side="left")
            _lbl(ctrl, label, font=FONT_SMALL).pack(side="left", padx=(10,2))
            _entry(ctrl, var, width=w).pack(side="left")

        ctk.CTkFrame(ctrl, fg_color="gray30",
                     width=1, height=28).pack(side="left", padx=12, pady=8)

        self.chromatic_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(ctrl, text="Aberrazione cromatica",
                        variable=self.chromatic_var,
                        command=self._toggle_chromatic,
                        font=FONT_SMALL).pack(side="left", padx=6)
        _lbl(ctrl, "N. colori", font=FONT_SMALL).pack(side="left", padx=(8,2))
        self.n_wavelengths = tk.StringVar(value="7")
        self.wl_entry = _entry(ctrl, self.n_wavelengths, width=44)
        self.wl_entry.pack(side="left")
        self.wl_entry.configure(state="disabled")

        ctk.CTkFrame(ctrl, fg_color="gray30",
                     width=1, height=28).pack(side="left", padx=12, pady=8)

        _btn(ctrl, "▶  Esegui",
             self._run_simulation, width=120).pack(side="left", padx=4)
        _btn(ctrl, "⌂  Reset vista",
             self._reset_view, width=120,
             fg_color="gray30", hover_color="gray40").pack(side="left", padx=4)
        _btn(ctrl, "💾  Esporta DXF",
             self._export_dxf, width=140,
             fg_color="#2a5a2a", hover_color="#3a7a3a").pack(side="left", padx=4)
        _btn(ctrl, "🌡  Isofote target",
             self._show_isophote, width=150,
             fg_color="#5a3a2a", hover_color="#7a4a3a").pack(side="left", padx=4)
        _btn(ctrl, "📊  LEE Breakdown",
             self._show_lee, width=150,
             fg_color="#3a2a5a", hover_color="#4a3a7a").pack(side="left", padx=4)

        self.stats_label = _lbl(ctrl, "", font=FONT_SMALL,
                                 text_color="#58a6ff")
        self.stats_label.pack(side="left", padx=12)

        # ── matplotlib canvas ─────────────────────────────────────────────────
        fig_frame = ctk.CTkFrame(parent, fg_color=BG_DARK)
        fig_frame.pack(fill="both", expand=True, padx=8, pady=(0,2))

        self.figure    = Figure(facecolor="#1e1e1e", dpi=100)
        self.ax_system = self.figure.add_subplot(1, 2, 1)
        self.ax_illum  = self.figure.add_subplot(1, 2, 2)
        for ax in (self.ax_system, self.ax_illum):
            ax.set_facecolor("#252526")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
        self.figure.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, fig_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.canvas.mpl_connect("scroll_event",       self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_right_click)

        # ── radial filter bar ─────────────────────────────────────────────────
        rng = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8, height=44)
        rng.pack(fill="x", padx=8, pady=(0,6))

        _lbl(rng, "Filtro radiale target:", font=FONT_SMALL).pack(
            side="left", padx=10)

        _lbl(rng, "r min", font=FONT_SMALL).pack(side="left", padx=(6,2))
        self.rmin_var = tk.DoubleVar(value=0.0)
        self.rmin_slider = ctk.CTkSlider(
            rng, from_=0, to=50, variable=self.rmin_var,
            width=160, command=self._on_range_slide)
        self.rmin_slider.pack(side="left", padx=4)
        self.rmin_lbl = _lbl(rng, "0.0", font=FONT_SMALL, width=44)
        self.rmin_lbl.pack(side="left")

        _lbl(rng, "r max", font=FONT_SMALL).pack(side="left", padx=(12,2))
        self.rmax_var = tk.DoubleVar(value=50.0)
        self.rmax_slider = ctk.CTkSlider(
            rng, from_=0, to=50, variable=self.rmax_var,
            width=160, command=self._on_range_slide)
        self.rmax_slider.pack(side="left", padx=4)
        self.rmax_lbl = _lbl(rng, "50.0", font=FONT_SMALL, width=44)
        self.rmax_lbl.pack(side="left")

        ctk.CTkFrame(rng, fg_color="gray30",
                     width=1, height=28).pack(side="left", padx=12, pady=6)

        _lbl(rng, "Efficienza:", font=FONT_SMALL).pack(side="left", padx=4)
        self.eff_lbl = _lbl(rng, "—", font=FONT_BOLD, text_color="#58a6ff")
        self.eff_lbl.pack(side="left", padx=4)

        _btn(rng, "Aggiorna", self._update_illuminance,
             width=110, fg_color="gray30",
             hover_color="gray40").pack(side="left", padx=10)

    # ── simulation helpers ────────────────────────────────────────────────────
    def _toggle_chromatic(self):
        state = "normal" if self.chromatic_var.get() else "disabled"
        self.wl_entry.configure(state=state)

    def _reset_view(self):
        self.ax_system.relim(); self.ax_system.autoscale()
        self.ax_illum.relim();  self.ax_illum.autoscale()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        factor = 1.15 if event.button == "down" else 1/1.15
        xd, yd = event.xdata, event.ydata
        if xd is None or yd is None:
            return
        ax.set_xlim([xd + (x-xd)*factor for x in ax.get_xlim()])
        ax.set_ylim([yd + (y-yd)*factor for y in ax.get_ylim()])
        self.canvas.draw_idle()

    def _on_right_click(self, event):
        if event.button != 3 or event.inaxes is None:
            return
        event.inaxes.relim(); event.inaxes.autoscale()
        self.canvas.draw_idle()

    def _on_range_slide(self, _=None):
        rmin = round(self.rmin_var.get(), 1)
        rmax = round(self.rmax_var.get(), 1)
        self.rmin_lbl.configure(text=f"{rmin:.1f}")
        self.rmax_lbl.configure(text=f"{rmax:.1f}")
        if self._last_traces:
            self._update_illuminance()

    def _update_illuminance(self):
        if not self._last_traces:
            return
        rmin = self.rmin_var.get()
        rmax = self.rmax_var.get()
        targets = [s["name"] for s in self.surfaces if s["kind"] == "target"]
        self.ax_illum.clear()
        self._style_ax(self.ax_illum)
        _, eff = plot_illuminance(
            self._last_traces,
            target_name=targets[0] if targets else None,
            r_min=rmin if rmin > 0 else None, r_max=rmax,
            ax=self.ax_illum)
        self.eff_lbl.configure(text=f"{eff*100:.1f} %")
        self.canvas.draw_idle()

    def _style_ax(self, ax):
        ax.set_facecolor("#252526")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

    # ── run ───────────────────────────────────────────────────────────────────
    def _run_simulation(self):
        if not self.surfaces:
            messagebox.showwarning(
                "Attenzione",
                "Definisci almeno una superficie nella scheda Sistema ottico.")
            return
        try:
            raw_surfaces = []   # collect all Surface objects before symmetrising
            # collect lens fill data for the plot (fronte_pts, retro_pts, color)
            self._lens_fill_data = []
            for s in self.surfaces:
                if s.get("type") == "lens":
                    fd, rd = build_lens_surface_defs(dict(s))
                    for surf_def in (fd, rd):
                        for obj in build_surface_objects(surf_def):
                            raw_surfaces.append(obj)
                    # store profile points for fill rendering
                    from .profiles import build_conic_profile
                    r_max  = float(s["r_max"])
                    n_pts  = int(s.get("n_points", 80))
                    mat    = s.get("material", "PMMA (n=1.49)")
                    color  = MATERIAL_FILL_COLOR.get(mat, "#90c8f0")
                    # fronte profile points
                    fg = s["fronte_geom"]
                    if fg.get("geom_type", "conic") == "conic" and float(fg.get("R", 0)) != 0:
                        f_pts = build_conic_profile(
                            vertex=fd["vertex"], R=fd["R"], k=fd.get("k",0),
                            r_max=r_max, coeffs=fd.get("coeffs",()),
                            n_points=n_pts, flip_z=fd.get("flip_z", False))
                    else:
                        f_pts = np.array([[fd["p1"][0], 0],
                                          [fd["p2"][0], r_max]])
                    # retro profile points
                    rg = s["retro_geom"]
                    if rg.get("geom_type", "conic") == "conic" and float(rg.get("R", 0)) != 0:
                        r_pts = build_conic_profile(
                            vertex=rd["vertex"], R=rd["R"], k=rd.get("k",0),
                            r_max=r_max, coeffs=rd.get("coeffs",()),
                            n_points=n_pts, flip_z=rd.get("flip_z", True))
                    else:
                        r_pts = np.array([[rd["p1"][0], 0],
                                          [rd["p2"][0], r_max]])
                    self._lens_fill_data.append((f_pts, r_pts, color))
                else:
                    for obj in build_surface_objects(s):
                        raw_surfaces.append(obj)

            # build symmetric system: every r≥0-only Segment surface gets
            # a mirrored counterpart for r≤0 so the physics is symmetric
            system = _build_symmetric_system(raw_surfaces)
            # keep a version without mirrored surfaces for plotting
            self._plot_system = OpticalSystem()
            for surf in raw_surfaces:
                self._plot_system.add(surf)

            src_def = self._current_source_def()
            medium  = MATERIAL_PRESETS.get(src_def["medium"], AIR)
            source  = LEDSource(
                position      = src_def["position"],
                axis_deg      = src_def["axis_deg"],
                half_angle_deg= src_def["half_angle_deg"],
                n_rays        = src_def["n_rays"],
                distribution  = src_def["distribution"],
                medium        = medium,
                les_shape     = src_def.get("les_shape", "point"),
                les_size      = src_def.get("les_size", 0.0),
                n_les         = src_def.get("les_n", 1),
            )
            base_rays = source.generate_rays()
            self._last_source = source

            max_b    = int(self.max_bounces.get())
            min_p    = float(self.min_power.get())
            refl_len = float(self.refl_length.get())
            chromatic = self.chromatic_var.get()

            if chromatic:
                n_wl = max(1, int(self.n_wavelengths.get()))
                wls  = wavelength_samples(n_wl)
                traces = []
                for wl, ray in chromatic_rays(base_rays, wls):
                    traces.extend(system.trace_ray(ray, max_bounces=max_b,
                                                    min_power=min_p))
                total_in = sum(r.power for r in base_rays)
            else:
                traces   = system.trace_many(base_rays,
                                             max_bounces=max_b, min_power=min_p)
                total_in = sum(r.power for r in base_rays)

        except Exception as exc:
            messagebox.showerror("Errore simulazione", str(exc))
            return

        # ── redraw ────────────────────────────────────────────────────────────
        self.ax_system.clear(); self.ax_illum.clear()
        self._style_ax(self.ax_system); self._style_ax(self.ax_illum)

        plot_surf = getattr(self, "_plot_system", system)
        if chromatic:
            self._plot_chromatic(plot_surf, traces, refl_len)
        else:
            plot_system(plot_surf, traces, ax=self.ax_system,
                        reflected_length=refl_len)

        # lens fill (drawn after surfaces, before rays would be on top)
        self._draw_lens_fills(self.ax_system)

        # update slider range
        r_hits = [pt[1] for t in traces for _, pt in t["hits"]]
        if r_hits:
            lo, hi = min(r_hits), max(r_hits)
            pad = max((hi-lo)*0.1, 1.0)
            for sl, var, lbl, val in [
                (self.rmin_slider, self.rmin_var, self.rmin_lbl, lo-pad),
                (self.rmax_slider, self.rmax_var, self.rmax_lbl, hi+pad),
            ]:
                sl.configure(from_=lo-pad, to=hi+pad)
                var.set(val)
                lbl.configure(text=f"{val:.1f}")

        self._last_traces      = traces
        self._last_total_power = total_in

        targets = [s["name"] for s in self.surfaces
                   if s.get("kind") == "target"]

        # LEE breakdown
        self._last_lee = compute_lee(
            traces, total_in,
            target_name=targets[0] if targets else None)

        _, eff = plot_illuminance(
            traces,
            target_name=targets[0] if targets else None,
            r_min=None, r_max=None,
            ax=self.ax_illum)
        self.eff_lbl.configure(text=f"{eff*100:.1f} %")

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

        total_hit = sum(t["power"] for t in traces if t["hits"])
        pct = 100 * total_hit / total_in if total_in else 0.0
        chrom_txt = (f"  |  λ: {n_wl}" if chromatic else "")
        src = getattr(self, "_last_source", None)
        les_txt = f"  |  LES: {src.les_description()}" if src else ""
        self.stats_label.configure(
            text=(f"Raggi: {len(base_rays)}  |  Rami: {len(traces)}"
                  f"{chrom_txt}  |  η: {pct:.1f}%{les_txt}"))

    # ── DXF export ────────────────────────────────────────────────────────────
    def _export_dxf(self):
        if not self._last_traces:
            messagebox.showwarning(
                "Attenzione",
                "Esegui prima una simulazione per avere dati da esportare.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF Drawing", "*.dxf")],
            title="Esporta sistema ottico in DXF")
        if not path:
            return
        try:
            refl_len   = float(self.refl_length.get())
            plot_surf  = getattr(self, "_plot_system", None)
            if plot_surf is None:
                plot_surf = OpticalSystem()
                for s in self.surfaces:
                    if s.get("type") == "lens":
                        fd, rd = build_lens_surface_defs(dict(s))
                        for s_def in (fd, rd):
                            for obj in build_surface_objects(s_def):
                                plot_surf.add(obj)
                    else:
                        for obj in build_surface_objects(s):
                            plot_surf.add(obj)
            lens_fill = getattr(self, "_lens_fill_data", None)
            if lens_fill:
                lens_fill = [(f, r, c, f"lente_{i}")
                             for i, (f, r, c) in enumerate(lens_fill)]
            export_dxf(
                path=path,
                system=plot_surf,
                traces=self._last_traces,
                lens_fill_data=lens_fill,
                reflected_length=refl_len,
                symmetric=True,
                title="OpenTIR – exported optical system")
            messagebox.showinfo("Esportato", f"File DXF salvato in:\n{path}")
        except Exception as exc:
            messagebox.showerror("Errore export DXF", str(exc))

    # ── isophote window ───────────────────────────────────────────────────────
    def _show_isophote(self):
        if not self._last_traces:
            messagebox.showwarning(
                "Attenzione",
                "Esegui prima una simulazione.")
            return

        targets = [s["name"] for s in self.surfaces if s.get("kind") == "target"]
        r_hits, powers = get_hit_points(
            self._last_traces,
            target_name=targets[0] if targets else None)

        if len(r_hits) < 3:
            messagebox.showwarning(
                "Dati insufficienti",
                "Nessun raggio ha raggiunto il target.\n"
                "Aumenta il numero di raggi o controlla la geometria.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Isofote – distribuzione di illuminamento sul target")
        win.geometry("860x680")
        self._iso_colorbar = None

        # ── controls ─────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(win, fg_color=BG_CARD, corner_radius=8, height=44)
        ctrl.pack(fill="x", padx=10, pady=(10, 4))

        self._iso_n_levels   = tk.StringVar(value="12")
        self._iso_n_bins     = tk.StringVar(value="60")
        self._iso_colormap   = tk.StringVar(value="inferno")
        self._iso_show_lines = tk.BooleanVar(value=True)
        self._iso_show_pts   = tk.BooleanVar(value=False)

        for label, var, w in [
            ("Livelli isofote:", self._iso_n_levels, 50),
            ("Bins griglia:",    self._iso_n_bins,   50),
        ]:
            _lbl(ctrl, label, font=FONT_SMALL).pack(side="left", padx=(10, 2))
            _entry(ctrl, var, width=w).pack(side="left", padx=(0, 8))

        _lbl(ctrl, "Colormap:", font=FONT_SMALL).pack(side="left", padx=(6, 2))
        cmaps = ["inferno", "hot", "plasma", "viridis", "magma",
                 "YlOrRd", "jet", "turbo"]
        _combo(ctrl, self._iso_colormap, cmaps, width=110,
               command=lambda v: self._redraw_isophote(
                   ax_iso, r_hits, powers, canvas_iso)).pack(side="left", padx=(0, 8))

        ctk.CTkCheckBox(ctrl, text="Linee isofote",
                        variable=self._iso_show_lines, font=FONT_SMALL,
                        command=lambda: self._redraw_isophote(
                            ax_iso, r_hits, powers, canvas_iso)).pack(
            side="left", padx=6)
        ctk.CTkCheckBox(ctrl, text="Punti impatto",
                        variable=self._iso_show_pts, font=FONT_SMALL,
                        command=lambda: self._redraw_isophote(
                            ax_iso, r_hits, powers, canvas_iso)).pack(
            side="left", padx=6)
        _btn(ctrl, "Aggiorna", width=100,
             command=lambda: self._redraw_isophote(
                 ax_iso, r_hits, powers, canvas_iso),
             fg_color="gray30", hover_color="gray40").pack(side="left", padx=8)
        _btn(ctrl, "💾 Salva PNG",
             command=lambda: self._save_isophote_png(fig_iso),
             width=120).pack(side="left", padx=4)

        # ── figure ────────────────────────────────────────────────────────────
        fig_iso = Figure(figsize=(8, 5.5), facecolor="#1e1e1e", dpi=100)
        ax_iso  = fig_iso.add_subplot(1, 1, 1)

        canvas_iso = FigureCanvasTkAgg(fig_iso, master=win)
        canvas_iso.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 6))
        NavigationToolbar2Tk(canvas_iso, win, pack_toolbar=False).pack(
            side="bottom", fill="x", padx=10, pady=(0, 4))

        # stats label
        r_min, r_max = float(r_hits.min()), float(r_hits.max())
        total_pw = float(powers.sum())
        stats_txt = (f"Hit: {len(r_hits)}  |  r ∈ [{r_min:.2f}, {r_max:.2f}] mm"
                     f"  |  Potenza totale sul target: {total_pw:.3f}")
        _lbl(win, stats_txt, font=FONT_SMALL, text_color="#58a6ff").pack(pady=4)

        self._redraw_isophote(ax_iso, r_hits, powers, canvas_iso)

    def _redraw_isophote(self, ax, r_hits, powers, canvas):
        """Rebuild the isophote plot with current settings."""
        ax.clear()
        ax.set_facecolor("#252526")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

        try:
            n_levels = max(3, int(self._iso_n_levels.get()))
            n_bins   = max(10, int(self._iso_n_bins.get()))
            cmap     = self._iso_colormap.get()
        except ValueError:
            canvas.draw_idle()
            return

        # Build 2D illuminance map using the axisymmetric geometry:
        # in the full 3D solid, the target is a disc; in the 2D section
        # we have r-values only.  We synthesise a fake 2D scatter by
        # reflecting hits across the axis (x ← r, y ← 0 ± small jitter
        # to avoid degenerate triangulation), then bin into a 2D grid
        # and use contourf for smooth isophotes.

        # Build a 2D grid: x = r (radial), y = azimuthal angle (0..2π)
        # For each hit at radius r_i with power p_i, spread uniformly
        # in azimuth proportional to power (correct for axisymmetric source).
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

        # 2D histogram → smooth grid
        lim = max(abs(r_hits).max() * 1.05, 1.0)
        edges = np.linspace(-lim, lim, n_bins + 1)
        H, xedg, yedg = np.histogram2d(x_all, y_all, bins=edges, weights=w_all)
        xc = 0.5 * (xedg[:-1] + xedg[1:])
        yc = 0.5 * (yedg[:-1] + yedg[1:])
        XX, YY = np.meshgrid(xc, yc)
        ZZ = H.T   # transpose to match (x, y) orientation

        # smooth slightly for cleaner isophotes
        from scipy.ndimage import gaussian_filter
        ZZ = gaussian_filter(ZZ, sigma=1.5)

        cf = ax.contourf(XX, YY, ZZ, levels=n_levels, cmap=cmap)

        if self._iso_show_lines.get():
            cs = ax.contour(XX, YY, ZZ, levels=n_levels,
                            colors="white", linewidths=0.5, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=7,
                      fmt=lambda v: f"{v:.2e}", colors="white")

        if self._iso_show_pts.get():
            r_all_pos = r_hits[r_hits >= 0]
            p_all_pos = powers[r_hits >= 0]
            ax.scatter(r_all_pos, np.zeros_like(r_all_pos),
                       c=p_all_pos, cmap=cmap, s=8, alpha=0.6,
                       zorder=5, label="hit points (section)")

        # Remove any existing colorbars before adding a new one
        fig = ax.get_figure()
        for existing_ax in fig.axes:
            if existing_ax is not ax and hasattr(existing_ax, '_colorbar'):
                existing_ax.remove()
        # Also remove via the figure's internal list
        if hasattr(fig, '_axstack'):
            pass  # handled above
        # Safest approach: track the colorbar and remove it explicitly
        if hasattr(self, '_iso_colorbar') and self._iso_colorbar is not None:
            try:
                self._iso_colorbar.remove()
            except Exception:
                pass
        self._iso_colorbar = fig.colorbar(cf, ax=ax, pad=0.02)
        cb = self._iso_colorbar
        cb.ax.yaxis.set_tick_params(color="white")
        cb.outline.set_edgecolor("white")
        for lbl in cb.ax.yaxis.get_ticklabels():
            lbl.set_color("white")
        cb.set_label("Illuminamento relativo [a.u.]", color="white")

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Isofote – distribuzione di illuminamento sul target")
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.get_figure().tight_layout()
        canvas.draw_idle()

    @staticmethod
    def _save_isophote_png(fig):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")],
            title="Salva grafico isofote")
        if not path:
            return
        fig.savefig(path, dpi=150, facecolor="#1e1e1e")
        messagebox.showinfo("Salvato", f"Grafico salvato in:\n{path}")

    # ── LEE window ────────────────────────────────────────────────────────────
    def _show_lee(self):
        if not self._last_traces or self._last_lee is None:
            messagebox.showwarning("Attenzione", "Esegui prima una simulazione.")
            return

        lee = self._last_lee
        win = ctk.CTkToplevel(self)
        win.title("LEE – Light Extraction Efficiency Breakdown")
        win.geometry("900x620")

        # ── summary text card ─────────────────────────────────────────────────
        card = ctk.CTkFrame(win, fg_color=BG_CARD, corner_radius=8)
        card.pack(fill="x", padx=10, pady=(10, 4))

        rows = [
            ("✅ Sul target",         lee.target_power,  lee.eta_target,  "#2ecc71"),
            ("🔁 Fresnel riflessa",   lee.fresnel_power, lee.eta_fresnel, "#e74c3c"),
            ("🔒 Intrappolata TIR",   lee.tir_power,     lee.eta_tir,     "#e67e22"),
            ("■  Assorbita (block)",  lee.blocked_power, lee.eta_blocked, "#95a5a6"),
            ("💨 Dispersa laterale",  lee.escaped_power, lee.eta_escaped, "#3498db"),
        ]
        _lbl(card, f"Potenza emessa: {lee.total_emitted:.4f}  |  "
                   f"η target: {lee.eta_target*100:.2f}%",
             font=FONT_BOLD, text_color="#58a6ff").grid(
            row=0, column=0, columnspan=6, padx=12, pady=(8, 4), sticky="w")

        for col, (label, power, eta, color) in enumerate(rows):
            f = ctk.CTkFrame(card, fg_color="#1e1e1e", corner_radius=6)
            f.grid(row=1, column=col, padx=6, pady=(4, 8), sticky="ew")
            card.columnconfigure(col, weight=1)
            _lbl(f, label, font=FONT_SMALL).pack(padx=8, pady=(6, 2))
            _lbl(f, f"{eta*100:.2f}%", font=FONT_BOLD,
                 text_color=color).pack(padx=8)
            _lbl(f, f"{power:.4f}", font=FONT_SMALL,
                 text_color="gray").pack(padx=8, pady=(0, 6))

        # ── charts ────────────────────────────────────────────────────────────
        fig_lee = Figure(figsize=(8.5, 4.0), facecolor="#1e1e1e", dpi=100)
        ax_pie = fig_lee.add_subplot(1, 2, 1)
        ax_bar = fig_lee.add_subplot(1, 2, 2)
        ax_pie.set_facecolor("#252526"); ax_bar.set_facecolor("#252526")

        plot_lee_pie(lee, ax=ax_pie, dark=True)
        plot_lee_bar(lee, ax=ax_bar, dark=True)
        fig_lee.tight_layout(pad=2.0)

        canvas_lee = FigureCanvasTkAgg(fig_lee, master=win)
        canvas_lee.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 4))
        NavigationToolbar2Tk(canvas_lee, win, pack_toolbar=False).pack(
            side="bottom", fill="x", padx=10, pady=(0, 4))

        # save button
        _btn(win, "💾 Salva PNG",
             command=lambda: self._save_isophote_png(fig_lee),
             width=130).pack(side="right", padx=12, pady=4)

        canvas_lee.draw()

    def _draw_lens_fills(self, ax):
        """Draw the coloured interior fill for each lens entity."""
        if not hasattr(self, "_lens_fill_data"):
            return
        for f_pts, r_pts, color in self._lens_fill_data:
            # upper half: join fronte (r increasing) + retro reversed
            upper_z = np.concatenate([f_pts[:,0], r_pts[::-1,0]])
            upper_r = np.concatenate([f_pts[:,1], r_pts[::-1,1]])
            ax.fill(upper_z, upper_r, color=color, alpha=LENS_FILL_ALPHA,
                    zorder=1, linewidth=0)
            # lower half mirror
            ax.fill(upper_z, -upper_r, color=color, alpha=LENS_FILL_ALPHA,
                    zorder=1, linewidth=0)

    def _plot_chromatic(self, system, traces, reflected_length=20.0):
        from .visualize import _surf_color
        name_to_color, idx = {}, 0
        for surf in system.surfaces:
            if surf.name not in name_to_color:
                name_to_color[surf.name] = _surf_color(idx)
                idx += 1

        seen = set()
        for surf in system.surfaces:
            pts   = surf.geometry.sample_points()
            color = name_to_color[surf.name]
            label = surf.name if surf.name not in seen else None
            seen.add(surf.name)
            self.ax_system.plot(pts[:,0], pts[:,1],
                                color=color, linewidth=2.0, label=label)
            if surf.kind != "target":
                self.ax_system.plot(pts[:,0], -pts[:,1],
                                    color=color, linewidth=2.0,
                                    linestyle="--", alpha=0.45)

        for trace in traces:
            wl      = trace.get("wavelength_nm", 589.3)
            is_refl = trace.get("reflected", False)
            color   = wavelength_to_rgb(wl)
            path    = np.array(trace["path"])
            if is_refl:
                if reflected_length <= 0:
                    continue
                clipped, total = [path[0]], 0.0
                for i in range(1, len(path)):
                    sv  = path[i] - path[i-1]
                    sl  = np.linalg.norm(sv)
                    if total + sl >= reflected_length:
                        frac = (reflected_length - total) / sl
                        clipped.append(path[i-1] + frac*sv)
                        break
                    clipped.append(path[i]); total += sl
                path = np.array(clipped)
                self.ax_system.plot(path[:,0], path[:,1],
                                    color=color, linewidth=0.3,
                                    alpha=0.25, linestyle="--")
            else:
                self.ax_system.plot(path[:,0], path[:,1],
                                    color=color, linewidth=0.7, alpha=0.65)

        self.ax_system.axhline(0, color="gray", linestyle="--", linewidth=0.7)

        # auto-zoom on optical elements
        all_z, all_r = [], []
        for surf in system.surfaces:
            pts = surf.geometry.sample_points()
            all_z.extend(pts[:,0].tolist())
            all_r.extend(pts[:,1].tolist())
            if surf.kind != "target":
                all_r.extend((-pts[:,1]).tolist())
        if all_z:
            z_min, z_max = min(all_z), max(all_z)
            r_min_s, r_max_s = min(all_r), max(all_r)
            pad_z = max((z_max-z_min)*0.12, 5.0)
            pad_r = max((r_max_s-r_min_s)*0.15, 5.0)
            self.ax_system.set_xlim(z_min-pad_z, z_max+pad_z)
            self.ax_system.set_ylim(r_min_s-pad_r, r_max_s+pad_r)

        self.ax_system.set_xlabel("z [mm]"); self.ax_system.set_ylabel("r [mm]")
        self.ax_system.set_title("Ray trace — aberrazione cromatica")
        self.ax_system.set_aspect("equal", adjustable="datalim")
        self.ax_system.grid(True, linewidth=0.3, color="#444")
        if seen:
            self.ax_system.legend(loc="best", fontsize=7)


def main():
    app = OpenTIRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
