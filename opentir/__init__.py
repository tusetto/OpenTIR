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
from .profiles import conic_sag, build_conic_profile, build_freeform_profile, profile_to_surfaces

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
]

__version__ = "0.3.1"
