"""Generate the P1 synthetic dataset and diagnostics.

Usage (from projects/p1-c1-ope/code):
    ../../../.venv-c1/bin/python generate_dataset.py
    ../../../.venv-c1/bin/python generate_dataset.py --n-seeds 8 --windows W1

The script writes:
    results/dataset.csv.gz          evaluation dataset (observable features only)
    results/true_paths.csv.gz       diagnostics: latent S, pool P, regime, sigma
    results/generation_summary.json counts, freeze-event summary, power notes

IMPORTANT: `true_paths.csv.gz` contains the latent price and must never be
joined into the evaluation features.  It exists only to audit the observation
mapping and the freeze-event design.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_env import (
    REGIME_NAMES,
    STRATEGIES,
    simulate_market,
    simulate_strategy,
    true_path_rows,
)

C1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = C1_DIR.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--windows", nargs="+", default=["W1", "W2", "W3", "W4"])
    p.add_argument("--steps", type=int, default=336, help="hourly steps per window")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "dataset.csv.gz")
    p.add_argument(
        "--diagnostics", type=Path, default=RESULTS_DIR / "true_paths.csv.gz"
    )
    p.add_argument("--summary", type=Path, default=RESULTS_DIR / "generation_summary.json")
    return p.parse_args()


def freeze_events(regime: np.ndarray) -> list[int]:
    """Lengths (hours) of contiguous freeze runs: weekend, holiday, halt."""
    events: list[int] = []
    run = 0
    for r in regime:
        if r in (2, 3, 4):
            run += 1
        elif run:
            events.append(run)
            run = 0
    if run:
        events.append(run)
    return events


def observation_diagnostics(market) -> dict:
    """How informative is the pool price about the true price, by regime?"""
    s_ret = np.diff(np.log(market.true_price))[: len(market.regime)]
    p_ret = np.diff(np.log(market.pool_price))[: len(market.regime)]
    out = {}
    for r, name in REGIME_NAMES.items():
        mask = market.regime == r
        if mask.sum() < 2:
            continue
        sr, pr = s_ret[mask], p_ret[mask]
        if np.std(sr) > 0 and np.std(pr) > 0:
            corr = float(np.corrcoef(sr, pr)[0, 1])
        else:
            corr = None
        out[name] = {
            "n_steps": int(mask.sum()),
            "corr_true_return_pool_return": corr,
            "mean_abs_true_return": float(np.mean(np.abs(sr))),
            "mean_abs_pool_return": float(np.mean(np.abs(pr))),
        }
    return out


def main() -> None:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    windows = list(args.windows)
    n_steps = int(args.steps)

    frames: list[pd.DataFrame] = []
    diag_frames: list[pd.DataFrame] = []
    tau_hours: list[int] = []
    sigmas: list[float] = []
    obs_diag_accum: dict[str, list[dict]] = {}

    for w_idx, window in enumerate(windows):
        for seed in seeds:
            market = simulate_market([seed, w_idx], n_steps)
            sigmas.append(float(market.sigma))
            tau_hours.extend(freeze_events(market.regime))

            diag_frames.append(pd.DataFrame(true_path_rows(market, window, seed)))
            d = observation_diagnostics(market)
            for name, vals in d.items():
                obs_diag_accum.setdefault(name, []).append(vals)

            for s_idx, strategy in enumerate(STRATEGIES):
                rows = simulate_strategy(
                    market, strategy, [seed, w_idx, s_idx], window, seed, n_steps
                )
                frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["strategy", "window", "seed", "episode_step"], ignore_index=True
    )

    diag = pd.concat(diag_frames, ignore_index=True)
    diag = diag.sort_values(["window", "seed", "episode_step"], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    diag.to_csv(args.diagnostics, index=False, compression="gzip")

    regime_counts = {
        REGIME_NAMES[int(r)]: int(n)
        for r, n in df["regime"].value_counts().sort_index().items()
    }
    group_counts = df.groupby(["strategy", "window", "seed"]).ngroups

    # Binomial standard error of a coverage estimate at the nominal 0.9 level.
    power = {
        name: {
            "n": n,
            "binomial_se_at_0.9": float(math.sqrt(0.9 * 0.1 / n)) if n else None,
        }
        for name, n in regime_counts.items()
    }

    obs_summary = {}
    for name, vals in obs_diag_accum.items():
        corrs = [
            v["corr_true_return_pool_return"]
            for v in vals
            if v["corr_true_return_pool_return"] is not None
        ]
        obs_summary[name] = {
            "mean_corr_true_return_pool_return": float(np.mean(corrs)) if corrs else None,
            "mean_abs_true_return": float(
                np.mean([v["mean_abs_true_return"] for v in vals])
            ),
            "mean_abs_pool_return": float(
                np.mean([v["mean_abs_pool_return"] for v in vals])
            ),
        }

    tau = np.array(tau_hours, dtype=int)
    summary = {
        "generated": "2026-10-20",
        "design": (
            "compact synthetic environment; latent S dynamics identical across "
            "regimes (I1'); regimes change only the observation mapping h_r"
        ),
        "policies": "heuristic logging policies (no RL training in v1)",
        "seeds": seeds,
        "windows": windows,
        "steps_per_window": n_steps,
        "strategies": list(STRATEGIES),
        "n_rows": int(len(df)),
        "n_groups_strategy_window_seed": int(group_counts),
        "regime_counts": regime_counts,
        "power": power,
        "freeze_events": {
            "n_events": int(len(tau)),
            "tau_hours_min": int(tau.min()) if len(tau) else None,
            "tau_hours_median": float(np.median(tau)) if len(tau) else None,
            "tau_hours_max": int(tau.max()) if len(tau) else None,
            "tau_hours_counts": {
                str(k): int(v) for k, v in zip(*np.unique(tau, return_counts=True))
            },
        },
        "sigma": {
            "min": float(np.min(sigmas)),
            "median": float(np.median(sigmas)),
            "max": float(np.max(sigmas)),
        },
        "observation_diagnostics": obs_summary,
        "notes": [
            "regime labels are calendar-derived and known in advance; no HMM smoothing",
            "true_paths.csv.gz is diagnostics only and must not enter features",
            "reward_env = (fees + change in fundamental IL vs HODL - costs) / initial notional",
            "portfolio_value is the observable pool-marked value (P), not the reward mark",
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"rows: {len(df)}  groups: {group_counts}")
    print("regime counts:", regime_counts)
    print(
        "freeze events:", summary["freeze_events"]["n_events"],
        "tau hours:", summary["freeze_events"]["tau_hours_counts"],
    )
    print("sigma min/median/max:",
          f"{summary['sigma']['min']:.4f}",
          f"{summary['sigma']['median']:.4f}",
          f"{summary['sigma']['max']:.4f}")
    print("written:", args.out)
    print("written:", args.diagnostics)
    print("written:", args.summary)


if __name__ == "__main__":
    main()
