"""
opentir.visualize
~~~~~~~~~~~~~~~~~~
Matplotlib-based plotting for OpenTIR 0.5.

Surface rendering
-----------------
Each surface is drawn with two overlapping curves:
  - Outer edge (solid, full alpha, linewidth 2): the actual optical
    boundary – used for ray intersection calculations.
  - Inner face (thicker, alpha 0.30, slightly offset inward): a purely
    decorative rendering hint that gives the surface a "glass slab" look.
    It has no optical meaning.
Each surface gets a distinct colour automatically assigned from a
qualitative palette; the colour is reused for both curves so the legend
stays clean.

Ray rendering
-------------
  - Transmitted / mirror / TIR rays: solid, linewidth 0.8, alpha 0.80.
  - Fresnel-reflected rays: dashed, linewidth 0.35, alpha 0.30.
    `reflected_length` controls how far they extend from the bounce
    point (mm). Set to 0 to hide them entirely.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Qualitative colour cycle – 10 distinct colours
_SURF_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]


def _surf_color(index):
    return _SURF_COLORS[index % len(_SURF_COLORS)]


def _offset_points(pts, offset):
    """Offset a polyline inward by `offset` mm (perpendicular to each segment)."""
    pts = np.array(pts, dtype=float)
    if len(pts) < 2:
        return pts
    offsets = np.zeros_like(pts)
    for i in range(len(pts)):
        i0 = max(0, i - 1)
        i1 = min(len(pts) - 1, i + 1)
        d = pts[i1] - pts[i0]
        n = np.linalg.norm(d)
        if n > 1e-12:
            perp = np.array([-d[1], d[0]]) / n
            offsets[i] = perp * offset
    return pts + offsets


def plot_system(system, traces=None, ax=None, show_axis=True,
                symmetric=True, reflected_length=20.0,
                title="OpenTIR – ray trace"):
    """
    Draw the optical system and ray paths.

    Parameters
    ----------
    symmetric : bool
        Mirror each surface below r=0 (dashed, alpha 0.4) so the full
        cross-section of the axisymmetric optic is visible.
    reflected_length : float
        Length (mm) of Fresnel-reflected ray segments from the bounce
        point. Set to 0 to hide all reflected rays.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Build a map: surface name → colour (first occurrence wins so that
    # multi-segment profiles that share a name get the same colour)
    name_to_color = {}
    color_idx = 0
    for surf in system.surfaces:
        if surf.name not in name_to_color:
            name_to_color[surf.name] = _surf_color(color_idx)
            color_idx += 1

    seen_labels = set()

    for surf in system.surfaces:
        pts   = surf.geometry.sample_points()
        color = name_to_color[surf.name]
        label = surf.name if surf.name not in seen_labels else None
        seen_labels.add(surf.name)

        # — outer edge (solid) —
        ax.plot(pts[:, 0], pts[:, 1],
                color=color, linewidth=2.0, alpha=1.0,
                label=label, solid_capstyle="round")

        # — symmetric mirror below axis —
        if symmetric and surf.kind != "target":
            ax.plot(pts[:, 0], -pts[:, 1],
                    color=color, linewidth=2.0, linestyle="--", alpha=0.45)

    if traces:
        for trace in traces:
            is_refl = trace.get("reflected", False)
            color   = trace.get("_color", "orange")
            path    = np.array(trace["path"])

            if is_refl:
                if reflected_length <= 0:
                    continue
                # Truncate path: walk segments until accumulated length >= limit
                clipped = [path[0]]
                total = 0.0
                for i in range(1, len(path)):
                    seg_vec = path[i] - path[i - 1]
                    seg_len = np.linalg.norm(seg_vec)
                    if total + seg_len >= reflected_length:
                        frac = (reflected_length - total) / seg_len
                        clipped.append(path[i - 1] + frac * seg_vec)
                        break
                    clipped.append(path[i])
                    total += seg_len
                clipped = np.array(clipped)
                ax.plot(clipped[:, 0], clipped[:, 1],
                        color=color, linewidth=0.35, alpha=0.30, linestyle="--")
            else:
                ax.plot(path[:, 0], path[:, 1],
                        color=color, linewidth=0.8, alpha=0.80)

    if show_axis:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)

    # ── auto-zoom on optical elements (not on ray endpoints) ──────────────────
    # Collect bounding box of all surface profiles only
    all_z, all_r = [], []
    for surf in system.surfaces:
        pts = surf.geometry.sample_points()
        all_z.extend(pts[:, 0].tolist())
        all_r.extend(pts[:, 1].tolist())
        if symmetric and surf.kind != "target":
            all_r.extend((-pts[:, 1]).tolist())   # mirror below axis

    if all_z:
        z_min, z_max = min(all_z), max(all_z)
        r_min_s, r_max_s = min(all_r), max(all_r)
        pad_z = max((z_max - z_min) * 0.12, 5.0)
        pad_r = max((r_max_s - r_min_s) * 0.15, 5.0)
        ax.set_xlim(z_min - pad_z, z_max + pad_z)
        ax.set_ylim(r_min_s - pad_r, r_max_s + pad_r)

    ax.set_xlabel("z (asse ottico) [mm]")
    ax.set_ylabel("r (radiale) [mm]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3)
    if seen_labels:
        ax.legend(loc="best", fontsize=8)
    return ax


def plot_illuminance(traces, target_name=None, n_bins=50,
                      r_min=None, r_max=None,
                      ax=None, title="Illuminance distribution on target"):
    """
    1-D illuminance histogram on a target surface.

    r_min, r_max : optional radial limits (mm); hits outside are excluded
                   from the efficiency calculation but the full histogram
                   is still drawn with excluded bins greyed out.
    Returns (ax, efficiency) where efficiency is the fraction of total
    emitted power hitting the target within [r_min, r_max].
    """
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
        return ax, 0.0

    r_hits   = np.array(r_hits)
    powers   = np.array(powers)
    total_pw = powers.sum()

    # full histogram
    ax.hist(r_hits, bins=n_bins, weights=powers,
            color="tab:orange", edgecolor="k", alpha=0.8)

    # grey out bins outside the selected range
    if r_min is not None or r_max is not None:
        lo = r_min if r_min is not None else r_hits.min()
        hi = r_max if r_max is not None else r_hits.max()
        mask_out = (r_hits < lo) | (r_hits > hi)
        if mask_out.any():
            ax.hist(r_hits[mask_out], bins=n_bins, weights=powers[mask_out],
                    color="gray", edgecolor="k", alpha=0.4)

    ax.set_xlabel("r – radial position on target [mm]")
    ax.set_ylabel("relative power")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)

    # efficiency within window
    mask_in = np.ones(len(r_hits), dtype=bool)
    if r_min is not None:
        mask_in &= r_hits >= r_min
    if r_max is not None:
        mask_in &= r_hits <= r_max
    eff = powers[mask_in].sum() / total_pw if total_pw > 0 else 0.0
    return ax, eff
