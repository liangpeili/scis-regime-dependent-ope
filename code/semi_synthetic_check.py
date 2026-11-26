#!/usr/bin/env python3
"""Pilot semi-synthetic coverage check (S2, 2026-10-22).

Non-C1 task.  The DGP keeps P1's calendar and observation map but replaces the
latent dynamics with per-symbol parameters calibrated from real hourly bars:

* ``sigma`` = 24/7-equivalent realised variance rate from
  ``results/semi_synthetic_calibration.json``;
* ``kappa`` = AR(1) mean reversion on regular-session bars;
* jump probability and absolute jump sd from the 4-sigma tail;
* ``h`` = P1's observable-only estimate; this pilot does not independently
  calibrate the full observation map to real pool prices;
* freeze durations remain P1's 14-day calendar (65 h weekend, 81 h holiday).

The identified and naive model-based OPE pipelines are P1's
``trajectory_value_check`` functions.  The forward simulators are replaced by
local copies so that the calibrated absolute jump sd, rather than the frozen
``3 * sigma`` literal, is used on both the data and model side.  This is a
pilot: small seed counts and model draws, reported with binomial/path-level
uncertainty only where cheap.

Writes ``results/semi_synthetic_check.json``.  It does not modify the frozen
P1 dataset, ``coverage_eval.py``, or any main-text result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from synthetic_env import (
    BASE_VOLUME,
    REGIME_EXTENDED,
    REGIME_HALT,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    WINDOW_DAYS,
    MarketPath,
    calendar_regime,
)
import policy_value_check as pvc
import trajectory_value_check as tvc

ROOT = Path(__file__).resolve().parents[1]
CAL_JSON = ROOT / "results" / "semi_synthetic_calibration.json"
OBS_H_JSON = ROOT / "results" / "observable_h_check.json"
OUT_JSON = ROOT / "results" / "semi_synthetic_check.json"

JUMP_SD_ABS = 0.0  # set per symbol before building records


def calibrated_params(symbol: str) -> dict:
    raw = json.loads(CAL_JSON.read_text())
    s = raw["symbols"][symbol]
    h_raw = json.loads(OBS_H_JSON.read_text())["h_estimate"]
    return {
        "sigma": float(s["sigma_247"]),
        "kappa": float(s["kappa_ar1_regular"]),
        "mu": math.log(100.0),
        "jump_prob": float(s["jump"]["threshold_4sd"]["prob"]),
        "jump_sd": float(s["jump"]["threshold_4sd"]["cond_sd"]),
        "h": {
            "open_sd": float(h_raw["open_sd"]),
            "ext_phi": float(h_raw["ext_phi"]),
            "ext_sd": float(h_raw["ext_sd"]),
            "stale_phi": float(h_raw["stale_phi"]),
            "stale_sd": float(h_raw["stale_sd"]),
        },
    }


def simulate_calibrated_market(params: dict, seed_seq, n_steps: int = 336) -> MarketPath:
    """P1 calendar + observation map, calibrated latent dynamics."""
    rng = np.random.default_rng(seed_seq)
    sigma = params["sigma"]
    kappa = params["kappa"]
    mu = params["mu"]
    jump_prob = params["jump_prob"]
    jump_sd = params["jump_sd"]
    h = params["h"]

    true_price = np.empty(n_steps + 1, dtype=float)
    true_price[0] = math.exp(mu)
    for t in range(n_steps):
        log_s = math.log(true_price[t])
        eps = rng.normal()
        jump = rng.normal(0.0, jump_sd) if rng.random() < jump_prob else 0.0
        true_price[t + 1] = math.exp(
            log_s + kappa * (mu - log_s) + sigma * eps + jump
        )

    regime = np.array(
        [calendar_regime((t // 24) % WINDOW_DAYS, t % 24) for t in range(n_steps)],
        dtype=int,
    )
    volume = rng.lognormal(math.log(BASE_VOLUME), 0.5, size=n_steps)
    volume[regime == REGIME_HALT] = 0.0

    pool_price = np.empty(n_steps + 1, dtype=float)
    eta = 0.0
    last_open_price = true_price[0]
    for t in range(n_steps + 1):
        r = regime[t] if t < n_steps else regime[-1]
        if r == REGIME_OPEN:
            last_open_price = true_price[t]
            eta = 0.0
            pool_price[t] = true_price[t] * math.exp(rng.normal(0.0, h["open_sd"]))
        elif r == REGIME_EXTENDED:
            eta = h["ext_phi"] * eta + rng.normal(0.0, h["ext_sd"])
            pool_price[t] = true_price[t] * math.exp(eta)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = h["stale_phi"] * eta + rng.normal(0.0, h["stale_sd"])
            pool_price[t] = last_open_price * math.exp(eta)
        else:  # halt
            pool_price[t] = pool_price[t - 1] if t > 0 else true_price[t]

    return MarketPath(true_price, pool_price, regime, volume, sigma)


def _sim_identified(
    rng,
    market,
    start,
    end,
    belief_mean,
    belief_cov,
    kappa,
    mu,
    jump_prob,
    sigma,
    h,
):
    ls, eta, ls_last = pvc.sample_pre_state(rng, belief_mean, belief_cov)
    s_list = [math.exp(ls)]
    p_list = [float(market.pool_price[start])]
    for t in range(start + 1, end + 2):
        eps = rng.normal()
        jump = (
            rng.normal(0.0, JUMP_SD_ABS) if rng.random() < jump_prob else 0.0
        )
        ls_new = ls + kappa * (mu - ls) + sigma * eps + jump
        r = int(market.regime[t]) if t < len(market.regime) else int(market.regime[-1])
        if r == REGIME_OPEN:
            eta = 0.0
            ls_last = ls_new
            p_new = math.exp(ls_new + rng.normal(0.0, h["open_sd"]))
        elif r == REGIME_EXTENDED:
            eta = h["ext_phi"] * eta + rng.normal(0.0, h["ext_sd"])
            p_new = math.exp(ls_new + eta)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = h["stale_phi"] * eta + rng.normal(0.0, h["stale_sd"])
            p_new = math.exp(ls_last + eta)
        elif r == REGIME_HALT:
            p_new = p_list[-1]
        else:
            raise ValueError(f"unknown regime: {r}")
        s_list.append(math.exp(ls_new))
        p_list.append(p_new)
        ls = ls_new
    return np.asarray(s_list), np.asarray(p_list)


def _sim_naive(rng, market, start, end, kappa_p, mu_p, jump_prob, sigma):
    lp = math.log(float(market.pool_price[start]))
    p_list = [math.exp(lp)]
    for _t in range(start + 1, end + 2):
        eps = rng.normal()
        jump = (
            rng.normal(0.0, JUMP_SD_ABS) if rng.random() < jump_prob else 0.0
        )
        lp = lp + kappa_p * (mu_p - lp) + sigma * eps + jump
        p_list.append(math.exp(lp))
    return np.asarray(p_list)


# Replace the frozen forward simulators at module level.  build_path_record
# and the segment helpers resolve these names at call time.
tvc._simulate_identified_forward = _sim_identified
tvc._simulate_naive_forward = _sim_naive


def build_records(
    params: dict,
    seeds: list[int],
    windows: list[str],
    reps: int,
    policy_name: str,
) -> list[dict]:
    records = []
    policy = pvc.POLICIES[policy_name]
    for w_idx, window in enumerate(windows):
        for seed in seeds:
            market = simulate_calibrated_market(
                params, [int(seed), w_idx, 20261022]
            )
            records.append(
                tvc.build_path_record(
                    market,
                    window,
                    int(seed),
                    policy,
                    {
                        "kappa": params["kappa"],
                        "mu": params["mu"],
                        "jump_prob": params["jump_prob"],
                        "h": params["h"],
                    },
                    reps,
                )
            )
    return records


def _extract(ev: dict) -> dict:
    out: dict = {"bias": ev["bias"]}
    fw = ev["full_window"]
    rl = ev["regime_level"]
    out["full_window"] = {
        "raw": {
            "within": fw["raw"]["within_window"]["coverage"],
            "loro": fw["raw"]["leave_one_window_out"]["coverage"],
        },
        "studentized": {
            "within": fw["studentized"]["within_window"]["coverage"],
            "loro": fw["studentized"]["leave_one_window_out"]["coverage"],
        },
        "cqr": {
            "within": fw["cqr"]["within_window"]["coverage"],
            "loro": fw["cqr"]["leave_one_window_out"]["coverage"],
        },
        "plugin_normal": {
            "coverage": fw["plugin_normal"]["coverage"],
            "width": fw["plugin_normal"]["mean_width"],
        },
    }
    out["regime"] = {
        "raw": {
            "within": rl["raw"]["within_regime"]["coverage"],
            "loro": rl["raw"]["leave_one_regime_out"]["coverage"],
        },
        "studentized": {
            "within": rl["studentized"]["within_regime"]["coverage"],
            "loro": rl["studentized"]["leave_one_regime_out"]["coverage"],
        },
        "cqr": {
            "within": rl["cqr"]["within_regime"]["coverage"],
            "loro": rl["cqr"]["leave_one_regime_out"]["coverage"],
        },
        "plugin_normal": {
            "coverage": rl["plugin_normal"]["all"]["coverage"],
            "width": rl["plugin_normal"]["all"]["mean_width"],
        },
    }
    return out


def run_symbol(symbol: str, cal_seeds: int, eval_seeds: int, reps: int) -> dict:
    global JUMP_SD_ABS
    params = calibrated_params(symbol)
    JUMP_SD_ABS = params["jump_sd"]
    windows = list(pvc.WINDOWS)
    out = {
        "params": {
            "sigma_247": params["sigma"],
            "kappa": params["kappa"],
            "jump_prob": params["jump_prob"],
            "jump_sd": params["jump_sd"],
            "h": params["h"],
        },
        "n_cal_paths_per_policy": cal_seeds * len(windows),
        "n_eval_paths_per_policy": eval_seeds * len(windows),
        "policies": {},
    }
    for policy_name in pvc.POLICIES:
        cal = build_records(
            params, list(range(5000, 5000 + cal_seeds)), windows, reps, policy_name
        )
        ev = build_records(
            params, list(range(6000, 6000 + eval_seeds)), windows, reps, policy_name
        )
        cal_id = tvc._stack(cal, "cont", "ident")
        ev_id = tvc._stack(ev, "cont", "ident")
        cal_naive = tvc._stack(cal, "cont", "naive")
        ev_naive = tvc._stack(ev, "cont", "naive")
        ident = tvc._evaluate_model(cal_id, ev_id, pvc.ALPHA)
        naive = tvc._evaluate_model(cal_naive, ev_naive, pvc.ALPHA)
        rec = {
            "identified": _extract(ident),
            "naive": _extract(naive),
        }
        rec["paired"] = {}
        for level in ("full_window", "regime"):
            for score in ("raw", "studentized", "cqr"):
                a = rec["identified"][level][score]["within"]
                b = rec["naive"][level][score]["within"]
                rec["paired"][f"{level}.{score}.within"] = float(a - b)
            a = rec["identified"][level]["plugin_normal"]["coverage"]
            b = rec["naive"][level]["plugin_normal"]["coverage"]
            rec["paired"][f"{level}.plugin_normal"] = float(a - b)
        out["policies"][policy_name] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "TSLA", "NVDA"])
    parser.add_argument("--cal-seeds", type=int, default=16)
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    result: dict = {
        "config": {
            "symbols": args.symbols,
            "cal_seeds": [5000, 5000 + args.cal_seeds - 1],
            "eval_seeds": [6000, 6000 + args.eval_seeds - 1],
            "windows": list(pvc.WINDOWS),
            "reps": args.reps,
            "alpha": pvc.ALPHA,
            "calendar": "P1 14-day window; weekend freeze 65 h, holiday 81 h",
            "sigma": "calibrated sigma_247 (per elapsed hour)",
            "jump": "calibrated 4-sigma tail probability and absolute sd",
            "h": "P1 observable-only h",
            "estimator": "P1 trajectory_value_check cont/ident and cont/naive",
        },
        "symbols": {},
        "limits": [
            "Pilot seed counts and model draws; intervals are descriptive.",
            "The calendar and freeze durations remain P1's, not the measured "
            "P2 durations; the semi-synthetic check calibrates return dynamics, "
            "not the calendar.",
            "h is P1's observable-only estimate and is not independently "
            "calibrated to real pool prices in this pilot.",
            "The jump scale is a 4-sigma tail descriptor, not a fitted "
            "jump-diffusion mixture; the forward model uses it as a fixed "
            "absolute jump sd.",
            "The model filter ignores jumps in its process noise, as in P1; "
            "only the forward draws include them.",
        ],
    }
    for symbol in args.symbols:
        print(f"running {symbol} ...", flush=True)
        result["symbols"][symbol] = run_symbol(
            symbol, args.cal_seeds, args.eval_seeds, args.reps
        )
        print(f"  done {symbol}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
