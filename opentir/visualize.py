"""
opentir.visualize
~~~~~~~~~~~~~~~~~~
Matplotlib-based plotting for release 0.1 (no GUI yet).
"""

import matplotlib.pyplot as plt
import numpy as np

from .chromatic import wavelength_to_rgb


def plot_system(system, traces=None, ax=None, show_axis=True,
                 title="OpenTIR - 2D geometric ray trace",
                 chromatic=False, use_power_thickness=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    seen_labels = set()
    colors = {"mirror": "tab:blue", "target": "tab:green",
              "block": "black", "refract": "tab:red"}
    for surf in system.surfaces:
        pts = surf.geometry.sample_points()
        color = colors.get(surf.kind, "gray")
        label = surf.name if surf.name not in seen_labels else None
        seen_labels.add(surf.name)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, label=label)

    if traces:
        for trace in traces:
            path = np.array(trace["path"])
            
            # Check if this trace has wavelength information (chromatic mode)
            wl = trace.get("wavelength_nm", None)
            
            if chromatic and wl is not None:
                # Use wavelength-based color for chromatic mode
                ray_color = wavelength_to_rgb(wl)
            else:
                # Default orange color for non-chromatic mode
                ray_color = "orange"
            
            # Calculate line width based on power if requested
            if use_power_thickness:
                power = trace.get("power", 1.0)
                # Scale linewidth: min 0.3, max 3.0, proportional to sqrt of power
                # Normalize assuming max power ~1.0
                linewidth = 0.3 + 2.7 * min(1.0, np.sqrt(power))
            else:
                linewidth = 0.6
            
            ax.plot(path[:, 0], path[:, 1], color=ray_color, linewidth=linewidth, alpha=0.7)

    if show_axis:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)

    ax.set_xlabel("z (optical axis)")
    ax.set_ylabel("r (radial)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3)
    if seen_labels:
        ax.legend(loc="best", fontsize=8)
    return ax


def plot_illuminance(traces, target_name=None, n_bins=50, ax=None,
                      title="Illuminance distribution on target"):
    """Build a simple 1D illuminance histogram from ray hits on target surfaces."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    r_hits, powers = [], []
    for trace in traces:
        for surface, point in trace["hits"]:
            if target_name and surface.name != target_name:
                continue
            r_hits.append(point[1])
            powers.append(trace["power"])

    if not r_hits:
        ax.set_title(title + " (no hits recorded)")
        return ax

    ax.hist(r_hits, bins=n_bins, weights=powers, color="tab:orange", edgecolor="k", alpha=0.8)
    ax.set_xlabel("r (radial position on target)")
    ax.set_ylabel("relative power")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)
    return ax
