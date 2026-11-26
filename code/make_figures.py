"""M3 figures for P1, generated from results/*.json.

Static, print-first matplotlib output. No value is hand-entered: every series is
read from the JSON file named below. Writes PDF and 300-dpi PNG into
projects/p1-c1-ope/paper/figures/.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"

# Palette: first three categorical slots, validated (all-pairs, light).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

NOMINAL = 0.90
REGIMES = ["OPEN", "EXTENDED", "WEEKEND", "HOLIDAY", "HALT"]
REGIME_IDS = list(range(5))
DPI = 300

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "mathtext.fontset": "dejavusans",
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
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def tidy(ax: plt.Axes, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: L0 held-out-regime coverage and strategy heterogeneity
# ---------------------------------------------------------------------------
def figure3() -> None:
    naive = load("coverage_gap_naive_global.json")
    mond = load("coverage_gap_exp1_mondrian.json")

    naive_cov = [naive["heldout_regime_coverage"][str(r)] for r in REGIME_IDS]
    mond_cov = [mond["heldout_regime_coverage"][str(r)] for r in REGIME_IDS]
    n_test = [naive["n_test_per_regime"][str(r)] for r in REGIME_IDS]

    # Canvas sized for the printed measure (\linewidth ~= 3.5 in): fonts
    # aim for ~9-10 pt at final size.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 1.26), gridspec_kw={"wspace": 0.42})

    x = np.arange(len(REGIMES))
    # Dot plot, not a truncated bar: coverage lives in a narrow band around 0.9.
    # Distinct markers per series so the pair stays readable in grayscale.
    for xi, cov, n, color, label, dx, mk in [
        (x, naive_cov, n_test, BLUE, "Naive global", -0.08, "o"),
        (x, mond_cov, n_test, ORANGE, "Mondrian", 0.08, "s"),
    ]:
        se = np.sqrt(np.array(cov) * (1 - np.array(cov)) / np.array(n))
        ax1.errorbar(xi + dx, cov, yerr=1.96 * se, fmt=mk, ms=5.0, color=color,
                     markeredgecolor="white", markeredgewidth=0.6,
                     elinewidth=0.9, capsize=2.5, label=label, zorder=3)
    ax1.axhline(NOMINAL, color=MUTED, lw=0.9, zorder=2)
    ax1.set_xticks(x, REGIMES, rotation=90)
    ax1.set_xlim(-0.5, len(REGIMES) - 0.5)
    ax1.set_ylim(0.84, 0.955)
    ax1.set_ylabel("Coverage (95% CI)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               ncol=2, handlelength=1.0, columnspacing=0.8, borderaxespad=0.2,
               fontsize=9)
    tidy(ax1)
    # No direct value labels at the WEEKEND cluster: both series sit on the
    # nominal line there and their error bars span most of the panel height,
    # so any label lands on line/point/whisker ink.

    strat = naive["heldout_strategy_coverage"]["2"]
    names = list(strat.keys())
    vals = [strat[k] for k in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    labels = [n.replace("_", " ") for n in names]

    ax2.plot(np.arange(len(vals)), vals, "o", ms=5.0, color=BLUE,
             markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax2.axhline(NOMINAL, color=MUTED, lw=0.9, zorder=2)
    ax2.set_xticks(np.arange(len(vals)), labels, rotation=90)
    ax2.set_xlim(-0.5, len(vals) - 0.5)
    ax2.set_ylim(0.64, 1.02)
    ax2.set_ylabel("Coverage")
    tidy(ax2)
    ax2.annotate(f"{vals[0]:.3f}", xy=(0, vals[0]), xytext=(5, -3),
                 textcoords="offset points", ha="left", fontsize=9, color=INK)
    ax2.annotate(f"{vals[-1]:.3f}", xy=(len(vals) - 1, vals[-1]), xytext=(-5, 3),
                 textcoords="offset points", ha="right", fontsize=9, color=INK)

    save(fig, "fig3-l0-coverage")


# ---------------------------------------------------------------------------
# Figure 4: L1 LORO boundary and paired identification effect
# ---------------------------------------------------------------------------
def figure4() -> None:
    traj = load("trajectory_value_check.json")
    pol = traj["policies"]["Wide_Rebalance_2pct"]
    scores = [("raw", "Raw", BLUE, "o", "-"),
              ("studentized", "Studentized", ORANGE, "s", "--"),
              ("cqr", "CQR", AQUA, "^", ":")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.41),
                                   gridspec_kw={"wspace": 0.95, "width_ratios": [1.0, 1.1]})

    x = np.arange(len(REGIMES))
    # Marker shape + line style per series: the trio must separate in grayscale.
    for key, label, color, mk, ls in scores:
        pg = pol["segment_wise"]["ident"]["regime_level"][key]["leave_one_regime_out"]["per_group"]
        ys = [pg[r]["coverage"] for r in REGIMES]
        ses = [pg[r]["coverage_se"] for r in REGIMES]
        ax1.errorbar(x, ys, yerr=1.96 * np.array(ses), color=color, lw=1.4,
                     marker=mk, linestyle=ls, ms=4.5, capsize=2, elinewidth=0.8,
                     markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
    ax1.axhline(NOMINAL, color=MUTED, lw=0.9, zorder=2)
    ax1.set_xticks(x, REGIMES, rotation=90)
    ax1.set_ylim(0.58, 1.02)
    ax1.set_ylabel("LORO coverage")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               handlelength=1.2, borderaxespad=0.2, fontsize=9, ncol=2,
               columnspacing=0.8)
    tidy(ax1)

    # Paired bootstrap, regime level.
    rows = []
    for pkey, pname in [("Wide_Rebalance_2pct", "Wide"), ("Narrow_Rebalance_1pct", "Narrow")]:
        for score, sname in [("studentized", "studentized"), ("cqr", "CQR"),
                             ("plugin_normal", "plug-in")]:
            d = traj["policies"][pkey]["paired"]["regime"][score]
            rows.append((pname, sname, d["diff"], d["ci95"]))

    ypos = np.arange(len(rows))[::-1]
    for y, (pname, sname, diff, ci) in zip(ypos, rows):
        color = BLUE if pname == "Wide" else ORANGE
        ax2.plot([ci[0] * 100, ci[1] * 100], [y, y], color=color, lw=1.4, zorder=3)
        ax2.plot(diff * 100, y, "o", color=color, ms=5.0,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax2.axvline(0.0, color=MUTED, lw=0.9, zorder=2)
    ax2.set_yticks(ypos, [f"{p[:1]}\u00b7{s if s != 'studentized' else 'stud'}"
                          .replace('plug-in', 'plug') for p, s, _, _ in rows])
    ax2.tick_params(axis="y", pad=3.5)
    # Two single-line texts instead of a two-line \label: matplotlib's tight
    # bbox under-measures multiline text that overhangs its axes, which clipped
    # the label at the page edge.
    ax2.text(0.5, -0.30, "Coverage difference (pp)", transform=ax2.transAxes,
             ha="center", va="top", fontsize=9.5)
    ax2.text(0.5, -0.52, "identified \u2212 naive", transform=ax2.transAxes,
             ha="center", va="top", fontsize=9.5)
    ax2.set_xlim(-2.5, 22.0)
    tidy(ax2, ygrid=False)
    ax2.grid(axis="x", zorder=0)
    # Legend by policy, kept above the panel so it cannot hit the CI rows.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=BLUE, lw=1.4, marker="o", ms=4, label="Wide"),
               Line2D([0], [0], color=ORANGE, lw=1.4, marker="o", ms=4, label="Narrow")]
    ax2.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.02),
               frameon=False, handlelength=1.2, borderaxespad=0.2, ncol=2,
               columnspacing=0.8, fontsize=9)

    save(fig, "fig4-l1-boundary")


# ---------------------------------------------------------------------------
# Figure 5: L2 branch coverage and union dominance
# ---------------------------------------------------------------------------
def figure5() -> None:
    pol = load("policy_value_check.json")
    methods = [("identified_raw", "Raw"), ("identified_studentized", "Studentized"),
               ("identified_cqr", "CQR")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.6, 1.42),
                                   gridspec_kw={"wspace": 0.5, "width_ratios": [1.0, 1.1]})

    x = np.arange(len(methods))
    hold, rebal, union, floors = [], [], [], []
    for meth, _ in methods:
        bj = pol["event_conformal"]["policies"]["Wide_Rebalance_2pct"]["methods"][meth]["within_tau2"]["branch_union_joint"]
        hold.append(bj["branches"]["hold"]["marginal_coverage"])
        rebal.append(bj["branches"]["rebalance"]["marginal_coverage"])
        union.append(bj["branches"]["rebalance"]["union_coverage"])
        # Empirical trivial bound from the measured overall coverage.
        q_b = bj["branches"]["rebalance"]["n_test"] / (
            bj["branches"]["rebalance"]["n_test"] + bj["branches"]["hold"]["n_test"])
        floors.append(max(0.0, 1.0 - (1.0 - bj["overall_marginal_coverage"]) / q_b))

    # Dot plot; a bar chart would need a zero baseline that hides the differences.
    ax1.scatter(x - 0.12, hold, s=28, color=BLUE, edgecolors="white", marker="o",
                linewidths=0.6, zorder=3, label="Hold")
    ax1.scatter(x, rebal, s=28, color=ORANGE, edgecolors="white", marker="s",
                linewidths=0.6, zorder=3, label="Rebal.")
    ax1.scatter(x + 0.12, union, s=30, marker="^", color=AQUA,
                edgecolors="white", linewidths=0.6, zorder=3,
                label="Union")
    ax1.scatter(x, floors, marker="_", s=90, color=INK, linewidths=1.2, zorder=4,
                label="Floor")
    ax1.axhline(NOMINAL, color=MUTED, lw=0.9, zorder=2)
    ax1.set_xticks(x, [m[1] for m in methods], rotation=90)
    ax1.set_xlim(-0.45, len(methods) - 0.45)
    ax1.set_ylim(0.45, 1.03)
    ax1.set_ylabel("Coverage")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               fontsize=9, ncol=2, handlelength=1.0, columnspacing=0.7,
               borderaxespad=0.2)
    tidy(ax1)
    # No direct value labels on the scatter: at this canvas the hold/rebalance/
    # union cluster plus whiskers leaves no clear pocket (the values are in
    # Table I Panel A).

    # 112 branch cells: marginal vs union.
    xs, ys = [], []
    for pname, pd in pol["event_conformal"]["policies"].items():
        for meth, rec in pd["methods"].items():
            for direction in ["within_tau2", "within_tau81",
                              "cal_tau2_test_tau81", "cal_tau81_test_tau2"]:
                bj = rec.get(direction, {}).get("branch_union_joint")
                if not bj:
                    continue
                for branch in bj["branches"].values():
                    xs.append(branch["marginal_coverage"])
                    ys.append(branch["union_coverage"])

    ax2.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=2)
    ax2.axhline(NOMINAL, color=GRID, lw=0.8, zorder=1)
    ax2.axvline(NOMINAL, color=GRID, lw=0.8, zorder=1)
    ax2.scatter(xs, ys, s=14, color=BLUE, alpha=0.65, edgecolors="white",
                linewidths=0.4, zorder=3, label=f"Branch cells (n={len(xs)})")
    ax2.set_xlim(0.35, 1.02)
    ax2.set_ylim(0.45, 1.03)
    ax2.set_xlabel("Marginal coverage")
    ax2.set_ylabel("Union coverage")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               handlelength=1.0, borderaxespad=0.2, fontsize=9)
    tidy(ax2)
    ax2.grid(axis="x", zorder=0)

    save(fig, "fig5-l2-branches")


# ---------------------------------------------------------------------------
# Figure 6: L3 specification curve for h and coverage sensitivity
# ---------------------------------------------------------------------------
def figure6() -> None:
    obs = load("observable_h_check.json")
    specs = [
        ("observable_h_check.json", obs["h_estimate"], "z=3"),
        ("trajectory_value_check_obs_h_z25.json", None, "z=2.5"),
        ("trajectory_value_check_obs_h_z40.json", None, "z=4"),
        ("trajectory_value_check_obs_h_sigma_open.json", None, "sd OPEN"),
        ("observable_h_check.json", obs["h_unmasked_mle"]["h"], "unmasked"),
    ]
    names, ext_phi, open_sd = [], [], []
    for fname, h, label in specs:
        if h is None:
            h = load(fname)["inputs"]["h"]
        names.append(label)
        ext_phi.append(h["ext_phi"])
        open_sd.append(h["open_sd"])

    truth_ext = obs["h_truth"]["ext_phi"]
    truth_open = obs["h_truth"]["open_sd"]
    bs = obs["uncertainty"]["bootstrap"]

    # ax2's full-precision y-ticks (0.0005 ...) are wide enough to push its
    # ylabel onto ax1's truth line; compact scientific y-ticks below free the
    # inter-panel gap without widening the canvas (keeps the display scale and
    # the raised 8-9.5 pt fonts).
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.3, 1.19),
                                        gridspec_kw={"wspace": 1.25})
    # Five rotated spec labels per panel need ~11 pt of pitch; this dense
    # three-panel canvas allows ~7.5 pt.  Keep primary text (titles, axis
    # labels, legend, annotations) at the raised 9-9.5 pt but hold these short
    # tick codes at 8 pt so the rotated boxes do not collide.
    for _ax in (ax1, ax2, ax3):
        _ax.tick_params(labelsize=8)
    x = np.arange(len(names))

    ax1.axhline(truth_ext, color=MUTED, lw=0.9, zorder=2)
    ax1.plot(x, ext_phi, "o", ms=5, color=BLUE,
             markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    # Bootstrap 90% interval on the main estimate only.
    ax1.errorbar([x[0]], [ext_phi[0]],
                 yerr=[[ext_phi[0] - bs["ext_phi"]["q05"]], [bs["ext_phi"]["q95"] - ext_phi[0]]],
                 fmt="none", ecolor=BLUE, elinewidth=1.0, capsize=2.5, zorder=4)
    ax1.set_xticks(x, names, rotation=90)
    ax1.set_ylim(0.72, 0.87)
    ax1.set_ylabel("$\\hat h_{\\mathrm{ext},\\phi}$")
    tidy(ax1)

    ax2.axhline(truth_open, color=MUTED, lw=0.9, zorder=2)
    ax2.plot(x, open_sd, "o", ms=5, color=BLUE,
             markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax2.errorbar([x[0]], [open_sd[0]],
                 yerr=[[open_sd[0] - bs["open_sd"]["q05"]], [bs["open_sd"]["q95"] - open_sd[0]]],
                 fmt="none", ecolor=BLUE, elinewidth=1.0, capsize=2.5, zorder=4)
    ax2.set_xticks(x, names, rotation=90)
    ax2.set_ylim(0.00022, 0.00056)
    # Compact ticks (display-only; the underlying 0.0003/0.0004/0.0005 values
    # are unchanged): wide full-precision labels would push the rotated ylabel
    # left onto ax1's horizontal truth reference line.
    ax2.set_yticks([0.0003, 0.0004, 0.0005], ["3", "4", "5"])
    ax2.set_ylabel("$\\hat h_{\\mathrm{open},\\mathrm{sd}}\\ (\\times 10^{-4})$")
    tidy(ax2)

    # Coverage difference vs the true-S G0 row, in percentage points.
    traj = load("trajectory_value_check.json")
    base = {}
    for pname, key in [("Wide", "Wide_Rebalance_2pct"), ("Narrow", "Narrow_Rebalance_1pct")]:
        rec = traj["summary"][key]["identified_raw"]
        base[pname] = (rec["full_window_within_coverage"], rec["regime_within_coverage"])
    files = [
        ("trajectory_value_check_obs_h.json", "z=3"),
        ("trajectory_value_check_obs_h_z25.json", "z=2.5"),
        ("trajectory_value_check_obs_h_z40.json", "z=4"),
        ("trajectory_value_check_obs_h_sigma_open.json", "sd OPEN"),
    ]
    diffs = {"Wide": ([], []), "Narrow": ([], [])}
    for fname, _ in files:
        s = load(fname)["summary"]
        for pname, key in [("Wide", "Wide_Rebalance_2pct"), ("Narrow", "Narrow_Rebalance_1pct")]:
            rec = s[key]["identified_raw"]
            diffs[pname][0].append(100 * (rec["full_window_within_coverage"] - base[pname][0]))
            diffs[pname][1].append(100 * (rec["regime_within_coverage"] - base[pname][1]))

    x3 = np.arange(len(files))
    w = 0.34
    ax3.bar(x3 - w / 2 - 0.01, diffs["Wide"][0], width=w, color=BLUE,
            label="Wide, $G$", zorder=3)
    # Hatch the second series so the pair separates in grayscale.
    ax3.bar(x3 + w / 2 + 0.01, diffs["Narrow"][0], width=w, color=ORANGE,
            hatch="///", edgecolor="white", linewidth=0.6,
            label="Narrow, $G$", zorder=3)
    ax3.axhline(0.0, color=MUTED, lw=0.9, zorder=2)
    ax3.set_xticks(x3, [n for _, n in files], rotation=90)
    ax3.set_ylim(-0.85, 0.65)
    ax3.set_ylabel("$\\Delta G$ (pp)")
    ax3.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               fontsize=9, ncol=1, handlelength=1.0, columnspacing=0.6,
               borderaxespad=0.2)
    tidy(ax3)

    save(fig, "fig6-l3-h")


def figure7() -> None:
    general = load("general_pomdp_check.json")
    models = general["models"]
    branch = general["branch"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.3, 1.19),
                                        gridspec_kw={"wspace": 0.85})
    # Same dense rotated-tick constraint as figure6: keep primary text raised,
    # hold short tick labels at 8 pt so the boxes do not collide.
    for _ax in (ax1, ax2, ax3):
        _ax.tick_params(labelsize=8)

    names = ["Known", "Fitted"]
    values = [models[k]["mean_absolute_return_error"] for k in ("identified", "naive")]
    bars = ax1.bar(names, values, color=[BLUE, ORANGE], width=0.58, zorder=3)
    ax1.tick_params(axis="x", rotation=90)
    for bar, value in zip(bars, values):
        # Lift the value label a couple of points off the bar top so the raised
        # 9 pt glyphs do not sit on the dark bar edge (TEXT-ON-INK).
        ax1.annotate(f"{value:.2f}",
                     (bar.get_x() + bar.get_width() / 2, value),
                     textcoords="offset points", xytext=(0, 2),
                     ha="center", va="bottom", fontsize=9,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax1.set_ylabel("Return MAE")
    ax1.set_ylim(0, max(values) * 1.20)
    tidy(ax1)

    regimes = ["normal", "degraded", "outage"]
    x = np.arange(len(regimes))
    width = 0.34
    for offset, model, color, label, hatch in [
        (-width / 2, "identified", BLUE, "Known", None),
        (width / 2, "naive", ORANGE, "Fitted", "///"),
    ]:
        rec = [models[model]["scores"]["raw"]["leave_one_regime_out"][r] for r in regimes]
        cov = [r["coverage"] for r in rec]
        err = [1.96 * r["coverage_se"] for r in rec]
        kw = {"hatch": hatch, "edgecolor": "white", "linewidth": 0.6} if hatch else {}
        ax2.bar(x + offset, cov, width=width, color=color, label=label,
                yerr=err, capsize=2.5, zorder=3, **kw)
    ax2.axhline(NOMINAL, color=MUTED, lw=0.9, zorder=2)
    ax2.set_xticks(x, regimes, rotation=90)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("LORO coverage")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
               ncol=1, fontsize=9, borderaxespad=0.2)
    tidy(ax2)

    rules = ["marginal", "predecision", "branch_union"]
    labels = ["Marginal", "Predecision", "Branch-union"]
    rec = [branch["rules"][r]["per_branch"]["1"] for r in rules]
    cov = [r["coverage"] for r in rec]
    err = [1.96 * r["coverage_se"] for r in rec]
    ax3.bar(labels, cov, color=BLUE, width=0.58, yerr=err, capsize=2.5, zorder=3)
    ax3.axhline(NOMINAL, color=MUTED, lw=0.9, zorder=2)
    ratio = branch["rules"]["branch_union"]["mean_width"] / branch["rules"]["marginal"]["mean_width"]
    ax3.text(0.03, 1.03, f"union width = {ratio:.2f}×\nmarginal",
             transform=ax3.transAxes, ha="left", va="bottom", fontsize=9,
             color=MUTED, linespacing=1.0)
    ax3.set_ylim(0.55, 1.09)
    ax3.set_ylabel("Branch coverage")
    ax3.tick_params(axis="x", rotation=90)
    tidy(ax3)

    save(fig, "fig7-general-pomdp")
    paths = {
        "generator": Path(__file__),
        "input": RES / "general_pomdp_check.json",
        "pdf": OUT / "fig7-general-pomdp.pdf",
        "png": OUT / "fig7-general-pomdp.png",
    }
    provenance = {
        "figure": "fig7-general-pomdp",
        "comparison_scope": general["config"]["comparison_scope"],
        "sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in paths.items()
        },
    }
    (OUT / "fig7-general-pomdp.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    print("wrote", OUT)
    for p in sorted(OUT.iterdir()):
        print(" ", p.name, p.stat().st_size)
