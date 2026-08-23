"""
opentir.visualize
~~~~~~~~~~~~~~~~~~
Matplotlib-based plotting for release 0.1 (no GUI yet).
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_system(system, traces=None, ax=None, show_axis=True,
                 title="OpenTIR - 2D geometric ray trace"):
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
            ax.plot(path[:, 0], path[:, 1], color="orange", linewidth=0.6, alpha=0.7)

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
