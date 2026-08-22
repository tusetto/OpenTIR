from .geometry import Segment, Arc, Profile
from .materials import (
    Material, AIR, PMMA, POLYCARBONATE, BK7_GLASS, SODA_LIME_GLASS,
)
from .optics import (
    Ray, Surface, OpticalSystem, reflect,
    snell_refract, fresnel_reflectance, critical_angle,
)
from .source import LEDSource
from .visualize import plot_system, plot_illuminance
from .sms import (
    design_sms_collimator, build_sms_surfaces, SMSChainResult,
    design_cartesian_oval_collimator, build_cartesian_oval_surface,
)
from .chromatic import wavelength_samples, wavelength_to_rgb, chromatic_rays
from .profiles import conic_sag, build_conic_profile, build_freeform_profile, profile_to_surfaces
from .export_dxf import export_dxf, get_hit_points
from .lee import compute_lee, LEEResult, plot_lee_pie, plot_lee_bar

__all__ = [
    "Segment", "Arc", "Profile",
    "Material", "AIR", "PMMA", "POLYCARBONATE", "BK7_GLASS", "SODA_LIME_GLASS",
    "Ray", "Surface", "OpticalSystem", "reflect",
    "snell_refract", "fresnel_reflectance", "critical_angle",
    "LEDSource",
    "plot_system", "plot_illuminance",
    "design_sms_collimator", "build_sms_surfaces", "SMSChainResult",
    "design_cartesian_oval_collimator", "build_cartesian_oval_surface",
    "conic_sag", "build_conic_profile", "build_freeform_profile", "profile_to_surfaces",
    "wavelength_samples", "wavelength_to_rgb", "chromatic_rays",
    "export_dxf", "get_hit_points",
    "compute_lee", "LEEResult", "plot_lee_pie", "plot_lee_bar",
]

__version__ = "0.6.2"
