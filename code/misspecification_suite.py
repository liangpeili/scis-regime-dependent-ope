#!/usr/bin/env python3
"""S2 misspecification suite: does the L1 ladder survive model misspecification?

The suite reruns the trajectory-level identified-vs-naive evaluation of
``trajectory_value_check.py`` on independent paths whose data-generating
process is perturbed in one of four ways:

* ``ar2_obs``      observation noise is AR(2) instead of the assumed AR(1)
                   (the latent path is unchanged);
* ``t_jumps``      latent jumps are Student-t(3) instead of Gaussian, scaled to
                   the same standard deviation;
* ``regime_noise`` 15% of OPEN hours are declared EXTENDED and 15% of EXTENDED
                   hours are declared OPEN (the latent path and P are unchanged);
* ``latent_regime`` a hidden two-state stale/regular observation state switches
                   independently of the calendar during OPEN/EXTENDED hours.

``correct`` reruns the published simulator on the same reduced seed set, so the
suite has an internal baseline.  The model side is held fixed: G0's
``kappa/mu/jump`` and the G0 ``h`` estimate are used for every variant, as in
``trajectory_value_check.py``.  This is a stress test of the estimator, not a
re-estimation exercise.

Output: ``results/stress/misspecification_suite.json``.  The script does not
modify the frozen simulator, the dataset, or ``coverage_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from synthetic_env import (
    KAPPA,
    MU_LOG,
    REGIME_EXTENDED,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    SIGMA_BASE,
    MarketPath,
    simulate_market,
)
import trajectory_value_check as tvc

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "stress" / "misspecification_suite.json"
PVC_JSON = ROOT / "results" / "policy_value_check.json"

VARIANTS = (
    "correct",
    "ar2_obs",
    "t_jumps",
    "regime_noise",
    "latent_regime",
)

# Baseline AR(1) observation maps, mirroring synthetic_env.simulate_market.
BASE_EXT_PHI, BASE_EXT_SD = 0.80, 0.002
BASE_STALE_PHI, BASE_STALE_SD = 0.98, 0.0008
BASE_EXT_STATIONARY_SD = BASE_EXT_SD / math.sqrt(1.0 - BASE_EXT_PHI**2)
BASE_STALE_STATIONARY_SD = BASE_STALE_SD / math.sqrt(1.0 - BASE_STALE_PHI**2)

# AR(2) observation-noise parameters.  The target is the baseline AR(1)
# *stationary* sd, not its innovation sd, so the marginal noise scale is held
# fixed and only the memory structure changes.
AR2_EXT = (0.70, 0.20, BASE_EXT_STATIONARY_SD)
AR2_STALE = (0.90, 0.08, BASE_STALE_STATIONARY_SD)

# Regime-label noise and hidden-regime transition probabilities.
REGIME_FLIP_PROB = 0.15
HIDDEN_ENTER = 0.01
HIDDEN_EXIT = 0.125
HIDDEN_BURN_IN = 24


def _ar2_innovation_sd(phi1: float, phi2: float, target_sd: float) -> float:
    """Innovation sd giving an AR(2) with the requested stationary sd."""
    factor = (1.0 - phi2) / (
        (1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1**2)
    )
    return target_sd / math.sqrt(factor)


def _observe_ar1(true_price: np.ndarray, regime: np.ndarray, rng) -> np.ndarray:
    """The simulator's AR(1) observation map, with a caller-owned RNG."""
    n_steps = len(regime)
    pool_price = np.empty(n_steps + 1, dtype=float)
    eta = 0.0
    last_open_price = true_price[0]
    for t in range(n_steps + 1):
        r = regime[t] if t < n_steps else regime[-1]
        if r == REGIME_OPEN:
            last_open_price = true_price[t]
            eta = 0.0
            pool_price[t] = true_price[t] * math.exp(rng.normal(0.0, 0.0005))
        elif r == REGIME_EXTENDED:
            eta = BASE_EXT_PHI * eta + rng.normal(0.0, BASE_EXT_SD)
            pool_price[t] = true_price[t] * math.exp(eta)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = BASE_STALE_PHI * eta + rng.normal(0.0, BASE_STALE_SD)
            pool_price[t] = last_open_price * math.exp(eta)
        else:
            pool_price[t] = pool_price[t - 1] if t > 0 else true_price[t]
    return pool_price


def _observe_ar2(true_price: np.ndarray, regime: np.ndarray, rng) -> np.ndarray:
    """AR(2) observation noise in EXTENDED and stale regimes."""
    n_steps = len(regime)
    pool_price = np.empty(n_steps + 1, dtype=float)
    eta1 = eta2 = 0.0
    last_open_price = true_price[0]
    ext_phi1, ext_phi2, ext_sd = AR2_EXT
    stale_phi1, stale_phi2, stale_sd = AR2_STALE
    ext_eps_sd = _ar2_innovation_sd(ext_phi1, ext_phi2, ext_sd)
    stale_eps_sd = _ar2_innovation_sd(stale_phi1, stale_phi2, stale_sd)
    for t in range(n_steps + 1):
        r = regime[t] if t < n_steps else regime[-1]
        if r == REGIME_OPEN:
            last_open_price = true_price[t]
            eta1 = eta2 = 0.0
            pool_price[t] = true_price[t] * math.exp(rng.normal(0.0, 0.0005))
        elif r == REGIME_EXTENDED:
            eta_new = ext_phi1 * eta1 + ext_phi2 * eta2 + rng.normal(0.0, ext_eps_sd)
            eta2, eta1 = eta1, eta_new
            pool_price[t] = true_price[t] * math.exp(eta_new)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta_new = (
                stale_phi1 * eta1 + stale_phi2 * eta2 + rng.normal(0.0, stale_eps_sd)
            )
            eta2, eta1 = eta1, eta_new
            pool_price[t] = last_open_price * math.exp(eta_new)
        else:
            pool_price[t] = pool_price[t - 1] if t > 0 else true_price[t]
    return pool_price


def _observe_hidden(true_price: np.ndarray, regime: np.ndarray, rng) -> np.ndarray:
    """AR(1) observation map with a hidden stale state during dense hours.

    The declared calendar regime remains ``regime``; only the observation law
    switches.  The hidden state is a two-state Markov chain on OPEN/EXTENDED
    hours, reset outside that window and during a 24-hour burn-in.
    """
    n_steps = len(regime)
    pool_price = np.empty(n_steps + 1, dtype=float)
    eta = 0.0
    last_open_price = true_price[0]
    hidden = False
    for t in range(n_steps + 1):
        r = regime[t] if t < n_steps else regime[-1]
        if r == REGIME_OPEN:
            last_open_price = true_price[t]
        if t >= HIDDEN_BURN_IN and r in (REGIME_OPEN, REGIME_EXTENDED):
            if hidden:
                if rng.random() < HIDDEN_EXIT:
                    hidden = False
            elif rng.random() < HIDDEN_ENTER:
                hidden = True
        else:
            hidden = False
        effective = REGIME_WEEKEND if hidden else r
        if effective == REGIME_OPEN:
            eta = 0.0
            pool_price[t] = true_price[t] * math.exp(rng.normal(0.0, 0.0005))
        elif effective == REGIME_EXTENDED:
            eta = BASE_EXT_PHI * eta + rng.normal(0.0, BASE_EXT_SD)
            pool_price[t] = true_price[t] * math.exp(eta)
        elif effective in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = BASE_STALE_PHI * eta + rng.normal(0.0, BASE_STALE_SD)
            pool_price[t] = last_open_price * math.exp(eta)
        else:
            pool_price[t] = pool_price[t - 1] if t > 0 else true_price[t]
    return pool_price


def _t_jump_path(base: MarketPath, seed_seq, n_steps: int) -> MarketPath:
    """Latent path with Student-t(3) jumps of the same sd as the Gaussian jump.

    The innovations follow the same draw order as ``simulate_market`` so the
    non-jump shocks stay paired with the correctly specified path; the regime
    and volume arrays are reused from ``base``.
    """
    rng = np.random.default_rng(seed_seq)
    sigma = SIGMA_BASE * math.exp(rng.normal(0.0, 0.35))
    true_price = np.empty(n_steps + 1, dtype=float)
    true_price[0] = math.exp(MU_LOG)
    for t in range(n_steps):
        log_s = math.log(true_price[t])
        eps = rng.normal()
        if rng.random() < 0.005:
            # t_3 has sd sqrt(3); scale to sd 3*sigma.
            jump = float(rng.standard_t(3)) * math.sqrt(3.0) * sigma
        else:
            jump = 0.0
        true_price[t + 1] = math.exp(
            log_s + KAPPA * (MU_LOG - log_s) + sigma * eps + jump
        )
    obs_rng = np.random.default_rng([int(seed_seq[0]), int(seed_seq[1]), 31415])
    pool_price = _observe_ar1(true_price, base.regime, obs_rng)
    return MarketPath(true_price, pool_price, base.regime, base.volume, sigma)


def _corrupt_regime(regime: np.ndarray, seed_seq) -> np.ndarray:
    """Flip 15% of OPEN<->EXTENDED labels; other regimes are untouched."""
    rng = np.random.default_rng([int(seed_seq[0]), int(seed_seq[1]), 27182])
    out = regime.copy()
    for t, r in enumerate(regime):
        if r == REGIME_OPEN and rng.random() < REGIME_FLIP_PROB:
            out[t] = REGIME_EXTENDED
        elif r == REGIME_EXTENDED and rng.random() < REGIME_FLIP_PROB:
            out[t] = REGIME_OPEN
    return out


def simulate_variant(seed_seq, variant: str, n_steps: int = 336) -> MarketPath:
    """One path under the requested misspecification."""
    if variant == "correct":
        return simulate_market(seed_seq, n_steps)
    base = simulate_market(seed_seq, n_steps)
    if variant == "ar2_obs":
        rng = np.random.default_rng([int(seed_seq[0]), int(seed_seq[1]), 161803])
        pool = _observe_ar2(base.true_price, base.regime, rng)
        return MarketPath(base.true_price, pool, base.regime, base.volume, base.sigma)
    if variant == "t_jumps":
        return _t_jump_path(base, seed_seq, n_steps)
    if variant == "regime_noise":
        regime = _corrupt_regime(base.regime, seed_seq)
        return MarketPath(base.true_price, base.pool_price, regime, base.volume, base.sigma)
    if variant == "latent_regime":
        rng = np.random.default_rng([int(seed_seq[0]), int(seed_seq[1]), 141421])
        pool = _observe_hidden(base.true_price, base.regime, rng)
        return MarketPath(base.true_price, pool, base.regime, base.volume, base.sigma)
    raise ValueError(f"unknown variant: {variant}")


def load_inputs() -> tuple[dict, dict]:
    raw = json.loads(PVC_JSON.read_text(encoding="utf-8"))
    params = raw["parameter_estimates"]
    h_raw = raw["observation_mapping_estimate"]
    h = {
        "open_sd": float(h_raw["open_sd"]),
        "ext_phi": float(h_raw["ext_phi"]),
        "ext_sd": float(h_raw["ext_sd"]),
        "stale_phi": float(h_raw["stale_phi"]),
        "stale_sd": float(h_raw["stale_sd"]),
    }
    inputs = {
        "kappa": float(params["kappa_plug_in"]),
        "mu": float(params["mu_plug_in"]),
        "jump_prob": float(params["jump_prob_plug_in"]),
        "h": h,
    }
    return inputs, raw


def _extract(policy_out: dict, paired: dict) -> dict:
    """Compact per-policy summary: bias plus the four published protocols."""
    out: dict = {"paired": paired, "models": {}}
    for prefix in ("identified", "naive"):
        cont = policy_out["continuous"][prefix]
        seg = policy_out["segment_wise"][prefix]
        out["models"][prefix] = {
            "bias": cont["bias"],
            # full-window G from the continuous rollout; G_r from the
            # segment-wise estimator (the paper's primary protocol).
            "full_window": cont["full_window"],
            "regime_level": seg["regime_level"],
        }
    return out


def _run_variant_policy(
    variant: str,
    policy_name: str,
    inputs: dict,
    cal_seeds,
    eval_seeds,
    reps: int,
) -> dict:
    policy = tvc.POLICIES[policy_name]
    cal_records = [
        tvc.build_path_record(
            simulate_variant([seed, wi], variant),
            tvc.WINDOWS[wi],
            seed,
            policy,
            inputs,
            reps,
        )
        for wi in range(len(tvc.WINDOWS))
        for seed in cal_seeds
    ]
    ev_records = [
        tvc.build_path_record(
            simulate_variant([seed, wi], variant),
            tvc.WINDOWS[wi],
            seed,
            policy,
            inputs,
            reps,
        )
        for wi in range(len(tvc.WINDOWS))
        for seed in eval_seeds
    ]

    stacks: dict = {}
    blocks: dict = {}
    for model_key in ("cont", "seg"):
        for method in ("ident", "naive"):
            cal = tvc._stack(cal_records, model_key, method)
            ev = tvc._stack(ev_records, model_key, method)
            stacks[(model_key, method)] = (cal, ev)
            blocks[(model_key, method)] = tvc._evaluate_model(cal, ev, tvc.ALPHA)

    paired_full = tvc._paired_full_window(
        stacks[("cont", "ident")][0],
        stacks[("cont", "ident")][1],
        stacks[("cont", "naive")][0],
        stacks[("cont", "naive")][1],
        tvc.ALPHA,
    )
    paired_regime = tvc._paired_regime(
        stacks[("seg", "ident")][0],
        stacks[("seg", "ident")][1],
        stacks[("seg", "naive")][0],
        stacks[("seg", "naive")][1],
        tvc.ALPHA,
    )

    policy_out = {
        "continuous": {
            "identified": blocks[("cont", "ident")],
            "naive": blocks[("cont", "naive")],
        },
        "segment_wise": {
            "identified": blocks[("seg", "ident")],
            "naive": blocks[("seg", "naive")],
        },
    }
    summary = _extract(policy_out, {"full_window": paired_full, "regime": paired_regime})
    summary["n_cal_paths"] = len(cal_records)
    summary["n_eval_paths"] = len(ev_records)
    summary["truth"] = {
        "full": tvc._truth_summary(
            np.array([r["truth_total"] for r in ev_records])
        ),
        "by_regime": tvc._regime_truth_summary(
            np.stack([r["truth_regime"] for r in ev_records])
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-seeds", type=int, default=64)
    parser.add_argument("--eval-seeds", type=int, default=64)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help="comma-separated subset"
    )
    parser.add_argument(
        "--policies",
        default="Wide_Rebalance_2pct,Narrow_Rebalance_1pct",
        help="comma-separated policy names",
    )
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    for p in policies:
        if p not in tvc.POLICIES:
            raise SystemExit(f"unknown policy: {p}")

    inputs, _raw = load_inputs()
    cal_seeds = tvc.CAL_SEEDS[: args.cal_seeds]
    eval_seeds = tvc.EVAL_SEEDS[: args.eval_seeds]

    design = {
        "correct": "published simulator; internal baseline on the reduced seed set",
        "ar2_obs": {
            "description": "observation noise AR(2) in EXTENDED and stale regimes; "
            "the latent path is the published simulator's path",
            "extended": {"phi1": AR2_EXT[0], "phi2": AR2_EXT[1], "stationary_sd": AR2_EXT[2]},
            "stale": {"phi1": AR2_STALE[0], "phi2": AR2_STALE[1], "stationary_sd": AR2_STALE[2]},
        },
        "t_jumps": {
            "description": "latent jumps Student-t(3), scaled to sd 3*sigma; "
            "the jump probability stays 0.005",
        },
        "regime_noise": {
            "description": "declared OPEN/EXTENDED labels flipped with probability "
            f"{REGIME_FLIP_PROB}; the latent path and P are unchanged",
        },
        "latent_regime": {
            "description": "hidden two-state stale/regular observation state during "
            "OPEN/EXTENDED hours; the model and policy see the calendar",
            "enter": HIDDEN_ENTER,
            "exit": HIDDEN_EXIT,
            "burn_in_hours": HIDDEN_BURN_IN,
        },
        "note": "the model side is fixed at the G0 kappa/mu/jump and G0 h for "
        "every variant; this is an estimator stress test, not re-estimation",
    }

    results: dict = {
        "config": {
            "cal_seeds": [int(cal_seeds[0]), int(cal_seeds[-1])],
            "eval_seeds": [int(eval_seeds[0]), int(eval_seeds[-1])],
            "n_cal_seeds": len(cal_seeds),
            "n_eval_seeds": len(eval_seeds),
            "windows": list(tvc.WINDOWS),
            "reps": int(args.reps),
            "alpha": tvc.ALPHA,
            "policies": policies,
            "variants": variants,
            "model_inputs": inputs,
        },
        "design": design,
        "results": {},
        "limits": [
            "The model side is not re-estimated inside the suite: kappa/mu/jump "
            "and h stay at their G0 values, so the suite measures estimator "
            "sensitivity to the DGP perturbation, not a full refit.",
            "Each variant changes one ingredient. Interactions (e.g., AR(2) "
            "observation noise plus regime noise) are not covered.",
            "Each variant is run at one perturbation magnitude (15% label "
            "flips, 7.4% hidden occupancy, the stated AR(2) coefficients, "
            "t(3) jumps); dose-response is not mapped.",
            "The reduced seed set is smaller than the main L1 run; coverage "
            "SEs are correspondingly larger and are reported per cell.",
            "regime_noise changes the labels used by both the policy and the "
            "model; it is a calendar-classification error, not a change in the "
            "latent observation law.",
            "latent_regime keeps the calendar in the model and policy; only "
            "the observation law has a hidden state. The hidden state is "
            "simulation-only and is never passed to an estimator.",
            "The estimand remains the same model-based target policy value; "
            "this suite does not add IPS/DR or a real-data check.",
        ],
    }

    for variant in variants:
        print(f"[misspec] variant={variant}", flush=True)
        results["results"][variant] = {}
        for policy_name in policies:
            print(f"[misspec]   policy={policy_name}", flush=True)
            summary = _run_variant_policy(
                variant,
                policy_name,
                inputs,
                cal_seeds,
                eval_seeds,
                args.reps,
            )
            results["results"][variant][policy_name] = summary
            ident = summary["models"]["identified"]
            naive = summary["models"]["naive"]
            print(
                "[misspec]     "
                f"bias id={ident['bias']['full']:.4f} "
                f"naive={naive['bias']['full']:.4f}; "
                f"within id raw="
                f"{ident['full_window']['raw']['within_window']['coverage']:.3f} "
                f"naive raw="
                f"{naive['full_window']['raw']['within_window']['coverage']:.3f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
