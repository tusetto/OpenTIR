"""
opentir.lee
~~~~~~~~~~~~
Light Extraction Efficiency (LEE) analysis — release 0.6.2.

Classifies every ray branch produced by the ray tracer into one of
five mutually exclusive categories and computes the power fraction in
each.  This gives the designer a clear breakdown of where the emitted
power goes, rather than a single efficiency number.

Categories
----------
TARGET      Power reaching at least one 'target' surface.
            This is the useful output.
TIR_LOST    Power in branches whose last event was a TIR reflection
            (reflected=False but never hit a target and ended inside a
            refractive medium with r > 0 -- heuristic: last segment
            going "backward" after hitting a refract surface at TIR).
            More precisely: branches that were produced by a TIR event
            (no Fresnel split, all power reflected) and never hit a
            target.  Detected by examining the path: if the branch
            ended without a hit and its last point is inside the
            optical system bounding box, it is likely TIR-trapped.
FRESNEL     Power in Fresnel partial-reflection branches (reflected=True).
            These rays bounce off refractive surfaces and are lost
            unless they eventually hit a target.
BLOCKED     Power absorbed by 'block' surfaces (baffles).
ESCAPED     Everything else: rays that left the optical system laterally
            without hitting a target or a block.

Usage
-----
    result = compute_lee(traces, total_emitted_power, system=system)
    print(result.summary())
    plot_lee_pie(result, ax=ax)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LEEResult:
    total_emitted: float          # total power emitted by the source
    target_power:  float          # power reaching target(s)
    fresnel_power: float          # power lost to Fresnel partial reflections
    tir_power:     float          # power trapped by TIR (estimated)
    blocked_power: float          # power absorbed by block surfaces
    escaped_power: float          # power that left the system without hitting anything useful

    # Per-category breakdown as fraction of total_emitted
    @property
    def eta_target(self):
        return self.target_power  / self.total_emitted if self.total_emitted else 0.0

    @property
    def eta_fresnel(self):
        return self.fresnel_power / self.total_emitted if self.total_emitted else 0.0

    @property
    def eta_tir(self):
        return self.tir_power     / self.total_emitted if self.total_emitted else 0.0

    @property
    def eta_blocked(self):
        return self.blocked_power / self.total_emitted if self.total_emitted else 0.0

    @property
    def eta_escaped(self):
        return self.escaped_power / self.total_emitted if self.total_emitted else 0.0

    def summary(self):
        lines = [
            "═══════════════════════════════════════════════════════",
            f"  LEE – Light Extraction Efficiency",
            f"  Potenza emessa totale : {self.total_emitted:.4f}",
            "───────────────────────────────────────────────────────",
            f"  ✅ Sul target          : {self.target_power:.4f}  ({self.eta_target*100:6.2f} %)",
            f"  🔁 Riflessa Fresnel   : {self.fresnel_power:.4f}  ({self.eta_fresnel*100:6.2f} %)",
            f"  🔒 Intrappolata TIR   : {self.tir_power:.4f}  ({self.eta_tir*100:6.2f} %)",
            f"  ■  Assorbita (block)  : {self.blocked_power:.4f}  ({self.eta_blocked*100:6.2f} %)",
            f"  💨 Dispersa laterale  : {self.escaped_power:.4f}  ({self.eta_escaped*100:6.2f} %)",
            "───────────────────────────────────────────────────────",
            f"  Totale verificato    : {self.target_power+self.fresnel_power+self.tir_power+self.blocked_power+self.escaped_power:.4f}",
            "═══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


def _is_tir_branch(trace):
    """
    Heuristic: a branch is considered TIR-trapped when it has NOT been
    flagged as a Fresnel reflection (reflected=False) AND it hit no
    target AND its path is "short" relative to the system extent
    (it bounced back before escaping).
    Since we removed the axis fold, TIR rays bounce back inside the
    lens and typically have few path nodes with the last point still
    close to the optical surfaces.  We use a simple proxy: the ray
    path reverses z direction after the last bounce.
    """
    path = trace["path"]
    if len(path) < 3:
        return False
    # check if z-direction reverses anywhere (bounce back)
    z_vals = [p[0] for p in path]
    dz     = np.diff(z_vals)
    return bool(np.any(dz < 0) and np.any(dz > 0))


def compute_lee(traces, total_emitted_power, target_name=None):
    """
    Classify every branch in `traces` and return a LEEResult.

    traces              : list of trace dicts from OpticalSystem.trace_many()
    total_emitted_power : sum of powers of all base rays (before tracing)
    target_name         : optional; if given, only hits on that target count
    """
    target_pw  = 0.0
    fresnel_pw = 0.0
    tir_pw     = 0.0
    blocked_pw = 0.0
    escaped_pw = 0.0

    for trace in traces:
        p    = trace["power"]
        hits = trace["hits"]
        refl = trace.get("reflected", False)

        # filter by target name if requested
        valid_hits = hits
        if target_name:
            valid_hits = [(s, pt) for s, pt in hits if s.name == target_name]

        if valid_hits:
            target_pw += p
        elif refl:
            # Fresnel partial-reflection branch that didn't reach target
            fresnel_pw += p
        elif _is_tir_branch(trace):
            tir_pw += p
        else:
            # check if the last surface hit was a block (absorbed)
            # We detect this from the path length: a block terminates
            # immediately, a ray that escaped goes far.
            path   = trace["path"]
            r_last = abs(path[-1][1]) if path else 0.0
            z_last = path[-1][0] if path else 0.0
            # if the ray travelled a very long distance it escaped laterally
            path_len = sum(
                np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                for i in range(len(path)-1)
            ) if len(path) > 1 else 0.0

            if path_len < 5.0:
                # very short path → absorbed by a block
                blocked_pw += p
            else:
                escaped_pw += p

    # power balance check — any remainder goes to escaped
    accounted = target_pw + fresnel_pw + tir_pw + blocked_pw + escaped_pw
    remainder = total_emitted_power - accounted
    if remainder > 1e-9:
        escaped_pw += remainder

    return LEEResult(
        total_emitted = total_emitted_power,
        target_power  = target_pw,
        fresnel_power = fresnel_pw,
        tir_power     = tir_pw,
        blocked_power = blocked_pw,
        escaped_power = escaped_pw,
    )


def plot_lee_pie(result: LEEResult, ax=None, dark=True):
    """
    Draw a pie chart of the LEE breakdown.

    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    labels = [
        f"Target\n{result.eta_target*100:.1f}%",
        f"Fresnel refl.\n{result.eta_fresnel*100:.1f}%",
        f"TIR trapped\n{result.eta_tir*100:.1f}%",
        f"Block assorbita\n{result.eta_blocked*100:.1f}%",
        f"Dispersa\n{result.eta_escaped*100:.1f}%",
    ]
    values = [
        result.target_power,
        result.fresnel_power,
        result.tir_power,
        result.blocked_power,
        result.escaped_power,
    ]
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#7f8c8d", "#3498db"]

    # remove zero slices for cleaner chart
    labels, values, colors = zip(
        *[(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    ) if any(v > 0 for v in values) else (labels, values, colors)

    wedges, texts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "#1e1e1e" if dark else "white",
                    "linewidth": 1.5},
    )
    for t in texts:
        t.set_color("white" if dark else "black")
        t.set_fontsize(9)

    title_col = "white" if dark else "black"
    ax.set_title(
        f"LEE Breakdown  —  η_target = {result.eta_target*100:.1f}%",
        color=title_col, fontsize=11, pad=12)
    if dark:
        ax.set_facecolor("#252526")
    return ax


def plot_lee_bar(result: LEEResult, ax=None, dark=True):
    """
    Draw a horizontal bar chart of the LEE breakdown — easier to read
    when one category dominates the others.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    categories = ["Target", "Fresnel", "TIR", "Block", "Dispersa"]
    values_pct  = [
        result.eta_target  * 100,
        result.eta_fresnel * 100,
        result.eta_tir     * 100,
        result.eta_blocked * 100,
        result.eta_escaped * 100,
    ]
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#7f8c8d", "#3498db"]

    bars = ax.barh(categories, values_pct, color=colors,
                   edgecolor="#1e1e1e" if dark else "white")
    for bar, pct in zip(bars, values_pct):
        if pct > 1.0:
            ax.text(pct + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{pct:.1f}%",
                    va="center", ha="left",
                    color="white" if dark else "black", fontsize=9)

    ax.set_xlabel("Percentuale della potenza emessa (%)",
                  color="white" if dark else "black")
    ax.set_title(
        f"LEE Breakdown  —  η = {result.eta_target*100:.1f}%",
        color="white" if dark else "black", fontsize=11)
    ax.set_xlim(0, 105)
    if dark:
        ax.set_facecolor("#252526")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
    return ax
