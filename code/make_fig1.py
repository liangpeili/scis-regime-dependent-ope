"""Figure 1: calendar-regime timeline for P1.

Every regime label is computed with synthetic_env.calendar_regime, the same
function used by the generator; no regime span is hand-entered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from synthetic_env import (  # noqa: E402
    HALT_DAY,
    HALT_HOURS,
    REGIME_EXTENDED,
    REGIME_HALT,
    REGIME_HOLIDAY,
    REGIME_NAMES,
    REGIME_OPEN,
    REGIME_WEEKEND,
    WINDOW_DAYS,
    calendar_regime,
)

OUT = ROOT / "paper" / "figures"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {
    REGIME_OPEN: "#2a78d6",
    REGIME_EXTENDED: "#1baf7a",
    REGIME_WEEKEND: "#eb6834",
    REGIME_HOLIDAY: "#8c6bb1",
    REGIME_HALT: "#3d3d3d",
}

# Texture channel for grayscale: adjacent runs are 1--10 pt wide at print
# size, so use dots, horizontal lines, and crosses whose marks survive narrow
# clipping.  HALT is a 2 h sliver and relies on its callout plus darkest fill.
HATCH = {
    REGIME_OPEN: None,
    REGIME_EXTENDED: "ooo",
    REGIME_WEEKEND: "---",
    REGIME_HOLIDAY: "xxx",
    REGIME_HALT: None,
}

KERNEL = {
    REGIME_OPEN: r"$P=S\,e^{\varepsilon}$ (anchored)",
    REGIME_EXTENDED: r"$P=S\,e^{a}$, $a$ persistent (lagged)",
    REGIME_WEEKEND: r"$P=S_{\mathrm{last\ open}}\,e^{a}$ (stale)",
    REGIME_HOLIDAY: "stale anchor, longer freeze",
    REGIME_HALT: r"$P=P_{\mathrm{prev}}$ (frozen)",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "hatch.linewidth": 0.8,
        # Embed TrueType: matplotlib's default pdf.fonttype=3 emits Type 3
        # bitmaps, which IEEE PDF eXpress rejects.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def runs(labels):
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


hours = WINDOW_DAYS * 24
regime = [calendar_regime(h // 24, h % 24) for h in range(hours)]

# Canvas sized for the printed measure (\linewidth ~= 3.5 in): fonts
# aim for ~9-10 pt at final size.
fig, ax = plt.subplots(figsize=(3.4, 0.85))
y0, height = 0.0, 0.55
for a, b, r in runs(regime):
    ax.broken_barh([(a, b - a)], (y0, height), facecolors=COLORS[r],
                   hatch=HATCH[r], edgecolors="white", linewidth=0.35, zorder=2)

# Regime identity lives in the legend (the caption points there); in-bar
# labels at this canvas width collide with neighbouring segments.
# Day boundaries and labels.
for d in range(WINDOW_DAYS + 1):
    ax.axvline(d * 24, color=AXIS, lw=0.45, zorder=1)
# Explicit call-out for the two-hour halt (too narrow to label in place).
halt = HALT_DAY * 24 + HALT_HOURS[0]
ax.annotate("HALT", xy=(halt + 1, y0 + height), xytext=(halt + 18, y0 + height + 0.52),
            fontsize=9, color=COLORS[REGIME_HALT], ha="center",
            arrowprops=dict(arrowstyle="-", color=COLORS[REGIME_HALT], lw=0.6))

ax.set_xlim(0, hours)
ax.set_ylim(-0.05, y0 + height + 0.75)
ax.set_yticks([])
ax.set_xticks([d * 24 for d in range(WINDOW_DAYS + 1)])
ax.set_xticklabels([str(d) for d in range(WINDOW_DAYS + 1)])
ax.set_xlabel("hour of the window (day boundaries at 00:00)")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

handles = [Patch(facecolor=COLORS[r], hatch=HATCH[r],
                 edgecolor="white" if HATCH[r] else "none",
                 label=f"{REGIME_NAMES[r]}: {KERNEL[r]}")
           for r in (REGIME_OPEN, REGIME_EXTENDED, REGIME_WEEKEND, REGIME_HOLIDAY, REGIME_HALT)]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.68),
          ncol=1, frameon=False, handlelength=1.1, columnspacing=1.2,
          handletextpad=0.5)

fig.savefig(OUT / "fig1-regime-timeline.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT / "fig1-regime-timeline.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
print("WROTE", OUT / "fig1-regime-timeline.pdf")
