"""Create the SCIS observation-channel and evaluation-boundary schematic.

This is a conceptual figure with no numerical data. PDF and PNG outputs use the
SCIS-requested Times New Roman at 8 pt and embedded TrueType fonts.
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "scis"
FONT = ROOT / "results" / "fonts" / "Times New Roman.ttf"
if not FONT.exists():
    raise FileNotFoundError(f"SCIS font missing: {FONT}")
font_manager.fontManager.addfont(str(FONT))
font_manager.fontManager.addfont(str(ROOT / "results" / "fonts" / "Times New Roman Bold.ttf"))
font_manager.fontManager.addfont(str(ROOT / "results" / "fonts" / "Times New Roman Italic.ttf"))

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

BLUE = "#2a78d6"
GREEN = "#1b7f5c"
GRAY = "#666666"
INK = "#111111"
LIGHT_BLUE = "#eaf2fc"
LIGHT_GREEN = "#e9f5ef"
LIGHT_GRAY = "#f0f0f0"


def box(ax, x, y, w, h, text, face, edge=INK, fontsize=11, weight="normal", hatch=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
        hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, weight=weight, linespacing=1.25)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=0.9, color=INK, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.015, 0.73, 0.29, 0.17,
        "Latent trajectory\n$X_{0:H}$, rewards, value", LIGHT_BLUE, edge=BLUE)
    box(ax, 0.355, 0.73, 0.29, 0.17,
        "Observation kernels $h_r$\nnoise, bias, stale, freeze", LIGHT_BLUE, edge=BLUE)
    box(ax, 0.695, 0.73, 0.29, 0.17,
        "Logged observations\n$P_{0:H}$, actions, costs", LIGHT_BLUE, edge=BLUE)
    arrow(ax, (0.305, 0.815), (0.355, 0.815))
    arrow(ax, (0.645, 0.815), (0.695, 0.815))

    # The bold 11 pt first line is ~241 pt wide; a 0.60-wide frame (~219 pt)
    # let it overhang the border by ~11 pt on each side. Widen to 0.74 so the
    # text stays inside with ~12 pt padding on both sides.
    box(ax, 0.13, 0.44, 0.74, 0.17,
        "Off-policy evaluator: identify what the log supports\nbefore assigning coverage",
        "#ffffff", edge=INK, weight="bold")
    arrow(ax, (0.5, 0.73), (0.5, 0.61))
    # The caption states the evaluator reads the logged observations through the
    # observation model; give the logged-data box its own edge into the
    # evaluator so it is not a dead end and both inputs (model + data) arrive.
    arrow(ax, (0.84, 0.73), (0.70, 0.61))

    box(ax, 0.025, 0.12, 0.29, 0.23,
        "Identified\nKernels and dynamics\nBelief and policy value",
        LIGHT_GREEN, edge=GREEN, fontsize=10)
    box(ax, 0.355, 0.12, 0.29, 0.23,
        "Recoverable\nRegime calibration\nBranch labels + width cost",
        LIGHT_GREEN, edge=GREEN, fontsize=10)
    box(ax, 0.685, 0.12, 0.29, 0.23,
        "Not identified\nPost-decision coverage\nNo cross-event variation",
        LIGHT_GRAY, edge=GRAY, fontsize=10)

    for start_x, end_x in ((0.28, 0.17), (0.5, 0.5), (0.72, 0.83)):
        arrow(ax, (start_x, 0.44), (end_x, 0.35))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig0-observation-boundary.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "fig0-observation-boundary.png", dpi=600,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    paths = {
        "generator": Path(__file__),
        "pdf": OUT / "fig0-observation-boundary.pdf",
        "png": OUT / "fig0-observation-boundary.png",
    }
    provenance = {
        "figure": "fig0-observation-boundary",
        "conceptual": True,
        "sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        },
    }
    (OUT / "fig0-observation-boundary.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
