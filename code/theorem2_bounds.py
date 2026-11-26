"""Persist the per-fold strict kappa-hat row of Theorem 2A/2B/Corollary 2.2.

Paper/theory provenance tool, not a C1 experiment.  It rebuilds the exact
Wide/Narrow freeze-episode conformity scores used by
``policy_value_check.py`` (same seed, same RNG order, same fitted parameters)
and recomputes, for each of the five folds in the tau=2 -> tau=81 direction:

* the strict two-sample KS distance ``kappa-hat`` between the fold's
  calibration scores (tau=2, other four folds) and its test scores (tau=81);
* the 2A exact marginal plug-in, 2B conditional PAC bound, and Corollary 2.2
  marginal confidence bound, both with the strict per-fold kappa-hat and with
  the pooled kappa-hat used in the earlier hand-computed row.

Only reads ``results/policy_value_check.json``; writes
``results/theorem2_bounds.json``. It also runs a three-sample out-of-sample
check: independent calibration paths determine the radius, independent target
paths estimate kappa-hat, and a final disjoint target sample evaluates coverage
and width. Each sampled seed contributes one path, with window types cycling
by seed, so no within-seed windows enter as extra observations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import policy_value_check as pv  # noqa: E402

IN_JSON = ROOT / "results" / "policy_value_check.json"
OUT_JSON = ROOT / "results" / "theorem2_bounds.json"

DELTA_CAL = 0.025
DELTA_KAPPA_2B = 0.025
DELTA_KAPPA_COR = 0.05
DIRECTION = "cal_tau2_test_tau81"


def _eps_cal(n_cal: int) -> float:
    return math.sqrt(math.log(2.0 / DELTA_CAL) / (2.0 * n_cal))


def _eps_kappa(n_cal: int, m: int, delta: float) -> float:
    return math.sqrt(math.log(4.0 / delta) / (2.0 * n_cal)) + math.sqrt(
        math.log(4.0 / delta) / (2.0 * m)
    )


def _build_scores(
    rng: np.random.Generator, offline: list, policy, theta: float, params: dict
) -> dict:
    """Replicate the scoring loop of ``episode_conformal_study``."""
    kappa = params["kappa"]
    mu = params["mu"]
    jump_prob = params["jump_prob"]
    h = params["h"]
    rows = {tau: [] for tau in pv.EVENT_TAUS}
    for m in offline:
        sigma_path = pv.estimate_sigma_open(m)
        sigma_all_path = pv.estimate_sigma_all(m)
        kappa_p_path, mu_p_path = pv.estimate_naive_kappa([m])
        for episode in pv.extract_freeze_episodes(m):
            tau = episode["tau"]
            if tau not in pv.EVENT_TAUS:
                continue
            pre = episode["pre"]
            pos0 = pv.position_before(m, policy, pre)
            p_pre = float(m.pool_price[pre])
            if pos0.in_pos:
                dev = p_pre / pos0.anchor - 1.0
            else:
                dev = 0.0
            pos_post = pv.position_before(m, policy, episode["post"])
            post_rebalance = bool(
                pos_post.in_pos
                and int(m.regime[episode["post"]])
                in (pv.REGIME_OPEN, pv.REGIME_EXTENDED)
                and abs(float(m.pool_price[episode["post"]]) / pos_post.anchor - 1.0)
                > theta
            )
            truth = pv.true_episode_value(m, episode, policy)
            belief_mean, belief_cov = pv.kalman_pre_belief(
                m, pre, kappa, mu, sigma_path, h
            )
            ident = np.array(
                [
                    pv.simulate_identified_episode(
                        rng,
                        m,
                        episode,
                        policy,
                        pos0,
                        kappa,
                        mu,
                        jump_prob,
                        sigma_path,
                        h,
                        belief_mean,
                        belief_cov,
                    )
                    for _ in range(pv.R_EVENT)
                ]
            )
            naive = np.array(
                [
                    pv.simulate_naive_episode(
                        rng,
                        m,
                        episode,
                        policy,
                        pos0,
                        kappa_p_path,
                        mu_p_path,
                        jump_prob,
                        sigma_all_path,
                    )
                    for _ in range(pv.R_EVENT)
                ]
            )
            mark = pv.rollout_segment(
                m.pool_price[pre : episode["post"] + 1],
                m.pool_price[pre : episode["post"] + 1],
                m.regime[pre : episode["post"]],
                m.volume[pre : episode["post"]],
                policy,
                copy.deepcopy(pos0),
                pos0.il(float(m.pool_price[pre])) if pos0.in_pos else 0.0,
                t_offset=pre,
            )
            if not pos0.in_pos:
                onset_action = "open"
            elif abs(dev) > theta:
                onset_action = "rebalance"
            else:
                onset_action = "hold"
            onset_near = bool(pos0.in_pos and abs(dev) / theta > 0.5)
            onset_side = "above" if dev > 0.0 else ("below" if dev < 0.0 else "flat")
            rows[tau].append(
                {
                    "truth": float(truth),
                    "ident_mean": float(ident.mean()),
                    "ident_sd": float(ident.std(ddof=1)),
                    "ident_q05": float(np.quantile(ident, 0.05)),
                    "ident_q95": float(np.quantile(ident, 0.95)),
                    "naive_mean": float(naive.mean()),
                    "naive_sd": float(naive.std(ddof=1)),
                    "naive_q05": float(np.quantile(naive, 0.05)),
                    "naive_q95": float(np.quantile(naive, 0.95)),
                    "mark": float(mark),
                    "in_position": bool(pos0.in_pos),
                    "onset_action": onset_action,
                    "onset_near": onset_near,
                    "onset_side": onset_side,
                    "post_rebalance": post_rebalance,
                }
            )

    out = {}
    folds_by_tau = {}
    for tau in pv.EVENT_TAUS:
        arr = rows[tau]
        folds_by_tau[tau] = (np.arange(len(arr)) % len(pv.OFFLINE_SEEDS)) % pv.N_FOLDS
        truth = np.array([r["truth"] for r in arr])
        ident_mean = np.array([r["ident_mean"] for r in arr])
        ident_sd = np.array([r["ident_sd"] for r in arr])
        ident_q05 = np.array([r["ident_q05"] for r in arr])
        ident_q95 = np.array([r["ident_q95"] for r in arr])
        naive_mean = np.array([r["naive_mean"] for r in arr])
        naive_sd = np.array([r["naive_sd"] for r in arr])
        naive_q05 = np.array([r["naive_q05"] for r in arr])
        naive_q95 = np.array([r["naive_q95"] for r in arr])
        mark = np.array([r["mark"] for r in arr])
        out[tau] = {
            "scores": {
                "identified_raw": np.abs(truth - ident_mean),
                "identified_studentized": np.abs(truth - ident_mean)
                / np.maximum(ident_sd, 1e-12),
                "identified_cqr": np.maximum(ident_q05 - truth, truth - ident_q95),
                "naive_raw": np.abs(truth - naive_mean),
                "naive_studentized": np.abs(truth - naive_mean)
                / np.maximum(naive_sd, 1e-12),
                "naive_cqr": np.maximum(naive_q05 - truth, truth - naive_q95),
                "naive_mark_raw": np.abs(truth - mark),
            },
            "scale": {
                "identified_studentized": ident_sd,
                "naive_studentized": naive_sd,
            },
            "base_width": {
                "identified_cqr": ident_q95 - ident_q05,
                "naive_cqr": naive_q95 - naive_q05,
            },
        }
    return out, folds_by_tau


def _fold_row(
    s2: np.ndarray,
    s81: np.ndarray,
    f2: np.ndarray,
    f81: np.ndarray,
    fold: int,
    kappa_pooled: float,
) -> dict:
    cal = s2[f2 != fold]
    test = s81[f81 == fold]
    n_cal = int(len(cal))
    m = int(len(test))
    kappa_strict = pv.ks_distance(cal, test)
    k = math.ceil((n_cal + 1) * (1.0 - pv.ALPHA))
    eps_cal = _eps_cal(n_cal)
    eps_kappa_2b = _eps_kappa(n_cal, m, DELTA_KAPPA_2B)
    eps_kappa_cor = _eps_kappa(n_cal, m, DELTA_KAPPA_COR)
    return {
        "fold": fold,
        "n_cal": n_cal,
        "m": m,
        "k": k,
        "kappa_hat_strict": kappa_strict,
        "kappa_hat_pooled": kappa_pooled,
        "eps_cal": eps_cal,
        "eps_kappa_2b": eps_kappa_2b,
        "eps_kappa_cor22": eps_kappa_cor,
        "two_a_strict": max(0.0, k / (n_cal + 1) - kappa_strict),
        "two_a_pooled_kappa": max(0.0, k / (n_cal + 1) - kappa_pooled),
        "two_b_strict": max(
            0.0, k / n_cal - eps_cal - kappa_strict - eps_kappa_2b
        ),
        "two_b_pooled_kappa": max(
            0.0, k / n_cal - eps_cal - kappa_pooled - eps_kappa_2b
        ),
        "cor22_strict": max(
            0.0, k / (n_cal + 1) - kappa_strict - eps_kappa_cor
        ),
        "cor22_pooled_kappa": max(
            0.0, k / (n_cal + 1) - kappa_pooled - eps_kappa_cor
        ),
    }


def _oos_group_scores(
    seeds: list[int],
    rng: np.random.Generator,
    policy,
    theta: float,
    params: dict,
) -> dict:
    rows = {tau: [] for tau in pv.EVENT_TAUS}
    for seed in seeds:
        wi = seed % len(pv.WINDOWS)
        m = pv.simulate_market([seed, wi])
        sigma_path = pv.estimate_sigma_open(m)
        for episode in pv.extract_freeze_episodes(m):
            tau = episode["tau"]
            if tau not in pv.EVENT_TAUS:
                continue
            pre = episode["pre"]
            pos0 = pv.position_before(m, policy, pre)
            truth = pv.true_episode_value(m, episode, policy)
            belief_mean, belief_cov = pv.kalman_pre_belief(
                m,
                pre,
                params["kappa"],
                params["mu"],
                sigma_path,
                params["h"],
            )
            draws = np.array(
                [
                    pv.simulate_identified_episode(
                        rng,
                        m,
                        episode,
                        policy,
                        pos0,
                        params["kappa"],
                        params["mu"],
                        params["jump_prob"],
                        sigma_path,
                        params["h"],
                        belief_mean,
                        belief_cov,
                    )
                    for _ in range(pv.R_EVENT)
                ]
            )
            rows[tau].append(
                {
                    "truth": float(truth),
                    "mean": float(draws.mean()),
                    "sd": float(draws.std(ddof=1)),
                    "q05": float(np.quantile(draws, 0.05)),
                    "q95": float(np.quantile(draws, 0.95)),
                }
            )
    out = {}
    for tau, arr in rows.items():
        truth = np.array([r["truth"] for r in arr])
        mean = np.array([r["mean"] for r in arr])
        sd = np.array([r["sd"] for r in arr])
        q05 = np.array([r["q05"] for r in arr])
        q95 = np.array([r["q95"] for r in arr])
        out[tau] = {
            "scores": {
                "identified_raw": np.abs(truth - mean),
                "identified_studentized": np.abs(truth - mean)
                / np.maximum(sd, 1e-12),
                "identified_cqr": np.maximum(q05 - truth, truth - q95),
            },
            "scale": sd,
            "base_width": q95 - q05,
        }
    return out


def _wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    z = 1.96
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def _oos_row(
    cal: dict,
    drift: dict,
    final: dict,
    cal_tau: int,
    test_tau: int,
    method: str,
) -> dict:
    cal_scores = cal[cal_tau]["scores"][method]
    drift_scores = drift[test_tau]["scores"][method]
    test_scores = final[test_tau]["scores"][method]
    n_cal = len(cal_scores)
    m = len(drift_scores)
    k = math.ceil((n_cal + 1) * (1.0 - pv.ALPHA))
    q = pv.conformal_quantile(cal_scores, pv.ALPHA)
    kappa = pv.ks_distance(cal_scores, drift_scores)
    two_a = max(0.0, k / (n_cal + 1) - kappa)
    two_b = max(
        0.0,
        k / n_cal
        - _eps_cal(n_cal)
        - kappa
        - _eps_kappa(n_cal, m, DELTA_KAPPA_2B),
    )
    cor22 = max(
        0.0,
        k / (n_cal + 1)
        - kappa
        - _eps_kappa(n_cal, m, DELTA_KAPPA_COR),
    )
    covered = test_scores <= q
    if method == "identified_raw":
        width = np.full(len(test_scores), 2.0 * q)
    elif method == "identified_studentized":
        width = 2.0 * q * final[test_tau]["scale"]
    else:
        width = final[test_tau]["base_width"] + 2.0 * q
    return {
        "n_cal": n_cal,
        "m_drift": m,
        "n_final": len(test_scores),
        "k": k,
        "q": q,
        "kappa_hat": kappa,
        "two_a": two_a,
        "two_b": two_b,
        "cor22": cor22,
        "coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
        "below_two_b": bool(covered.mean() < two_b),
        "below_cor22": bool(covered.mean() < cor22),
    }


def _oos_summary(rows: list[dict]) -> dict:
    coverage = np.array([r["coverage"] for r in rows])
    se = float(coverage.std(ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else math.nan
    summary = {
        "n_replicates": len(rows),
        "mean_coverage": float(coverage.mean()),
        "coverage_se_across_replicates": se,
        "coverage_mean_ci95": [
            max(0.0, float(coverage.mean()) - 1.96 * se),
            min(1.0, float(coverage.mean()) + 1.96 * se),
        ],
        "coverage_q05": float(np.quantile(coverage, 0.05)),
        "coverage_q95": float(np.quantile(coverage, 0.95)),
    }
    for key in ("kappa_hat", "two_a", "two_b", "cor22", "mean_width"):
        vals = np.array([r[key] for r in rows])
        name = "width" if key == "mean_width" else key
        summary[f"mean_{name}"] = float(vals.mean())
        summary[f"min_{name}"] = float(vals.min())
        summary[f"max_{name}"] = float(vals.max())
    summary["mean_tightness_two_b"] = float(
        np.mean([r["coverage"] - r["two_b"] for r in rows])
    )
    summary["mean_tightness_cor22"] = float(
        np.mean([r["coverage"] - r["cor22"] for r in rows])
    )
    for key in ("two_b", "cor22"):
        count = sum(r[f"below_{key}"] for r in rows)
        summary[f"below_{key}_count"] = count
        summary[f"below_{key}_fraction"] = count / len(rows)
        summary[f"below_{key}_fraction_ci95"] = _wilson(count, len(rows))
    return summary


def _run_oos(params: dict, replicates: int, group_seeds: int, final_seeds: int) -> dict:
    methods = ("identified_raw", "identified_studentized", "identified_cqr")
    directions = ((2, 81), (81, 2))
    rows = {(cal_tau, test_tau, method): [] for cal_tau, test_tau in directions for method in methods}
    policy_name = "Wide_Rebalance_2pct"
    policy = pv.POLICIES[policy_name]
    theta = pv.THRESHOLDS[policy_name]
    for rep in range(replicates):
        base = 8_000_000 + 1_000 * rep
        rng = np.random.default_rng(90_000_000 + rep)
        groups = []
        offset = 0
        for size in (group_seeds, group_seeds, final_seeds):
            seeds = list(range(base + offset, base + offset + size))
            groups.append(_oos_group_scores(seeds, rng, policy, theta, params))
            offset += size
        cal, drift, final = groups
        for cal_tau, test_tau in directions:
            for method in methods:
                row = _oos_row(cal, drift, final, cal_tau, test_tau, method)
                row["replicate"] = rep
                rows[(cal_tau, test_tau, method)].append(row)
        print(f"oos {rep + 1}/{replicates}", flush=True)
    output = {}
    for (cal_tau, test_tau, method), recs in rows.items():
        direction = f"cal_tau{cal_tau}_test_tau{test_tau}"
        output.setdefault(direction, {})[method] = {
            "summary": _oos_summary(recs),
            "replicates": recs,
        }
    return output


def _self_test() -> None:
    cal = {
        2: {
            "scores": {
                method: np.arange(1.0, 11.0)
                for method in ("identified_raw", "identified_studentized", "identified_cqr")
            }
        }
    }
    drift = {81: {"scores": cal[2]["scores"]}}
    final = {
        81: {
            "scores": {
                method: np.array([1.0, 11.0])
                for method in ("identified_raw", "identified_studentized", "identified_cqr")
            },
            "scale": np.array([2.0, 4.0]),
            "base_width": np.array([1.0, 3.0]),
        }
    }
    raw = _oos_row(cal, drift, final, 2, 81, "identified_raw")
    assert raw["q"] == 10.0 and raw["kappa_hat"] == 0.0
    assert raw["coverage"] == 0.5 and raw["mean_width"] == 20.0
    studentized = _oos_row(cal, drift, final, 2, 81, "identified_studentized")
    cqr = _oos_row(cal, drift, final, 2, 81, "identified_cqr")
    assert studentized["mean_width"] == 60.0
    assert cqr["mean_width"] == 22.0
    final[81]["scores"] = {
        method: np.array([100.0, 100.0])
        for method in ("identified_raw", "identified_studentized", "identified_cqr")
    }
    changed = _oos_row(cal, drift, final, 2, 81, "identified_raw")
    for key in ("q", "kappa_hat", "two_a", "two_b", "cor22"):
        assert changed[key] == raw[key]
    assert changed["coverage"] == 0.0
    summary = _oos_summary([raw, changed])
    assert summary["mean_width"] == 20.0 and "mean_mean_width" not in summary
    print("self-test 8/8 PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--oos-replicates", type=int, default=100)
    parser.add_argument("--oos-group-seeds", type=int, default=64)
    parser.add_argument("--oos-final-seeds", type=int, default=256)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.oos_replicates < 1 or args.oos_group_seeds < 1 or args.oos_final_seeds < 1:
        raise SystemExit("OOS replicate and seed counts must be positive")
    source = json.loads(IN_JSON.read_text(encoding="utf-8"))
    params = {
        "kappa": float(source["parameter_estimates"]["kappa_plug_in"]),
        "mu": float(source["parameter_estimates"]["mu_plug_in"]),
        "jump_prob": float(source["parameter_estimates"]["jump_prob_plug_in"]),
        "h": {
            key: float(source["observation_mapping_estimate"][key])
            for key in ("open_sd", "ext_phi", "ext_sd", "stale_phi", "stale_sd")
        },
    }

    offline = [
        pv.simulate_market([seed, wi])
        for wi in range(len(pv.WINDOWS))
        for seed in pv.OFFLINE_SEEDS
    ]

    result = {
        "config": {
            "source": "results/policy_value_check.json",
            "seed": pv.EVENT_SEEDS[0],
            "direction": DIRECTION,
            "alpha": pv.ALPHA,
            "n_folds": pv.N_FOLDS,
            "delta_cal": DELTA_CAL,
            "delta_kappa_2b": DELTA_KAPPA_2B,
            "delta_kappa_cor22": DELTA_KAPPA_COR,
            "kappa_plug_in": params["kappa"],
            "mu_plug_in": params["mu"],
            "jump_prob_plug_in": params["jump_prob"],
            "h": params["h"],
            "out_of_sample": {
                "policy": "Wide_Rebalance_2pct",
                "replicates": args.oos_replicates,
                "calibration_and_drift_seeds_each": args.oos_group_seeds,
                "final_test_seeds": args.oos_final_seeds,
                "windows_per_seed": 1,
                "window_rule": "window index = seed mod 4",
                "r_event": pv.R_EVENT,
                "seed_rule": "replicate r uses consecutive disjoint seeds from 8000000+1000*r; calibration, drift, and final groups are consecutive and non-overlapping",
                "score_rng_rule": "default_rng(90000000+r)",
                "note": "The radius uses calibration only; kappa-hat uses calibration and the independent drift sample; final-test scores are untouched by both. Below-bound frequencies compare noisy final-sample coverage with the bound and are not exact PAC violation frequencies.",
            },
            "note": "Strict kappa-hat uses each fold's own calibration and test "
            "scores (disjoint from the point being evaluated). The pooled row "
            "uses the single pooled KS distance 0.1171875; the earlier "
            "hand-computed 'strict' row in theorem2.md actually used this "
            "pooled value and only replaced m by the fold size.",
        },
        "policies": {},
    }

    rng = np.random.default_rng(pv.EVENT_SEEDS[0])
    for name, policy in pv.POLICIES.items():
        theta = pv.THRESHOLDS[name]
        scores, folds = _build_scores(rng, offline, policy, theta, params)
        s2_all = scores[2]["scores"]
        s81_all = scores[81]["scores"]
        f2, f81 = folds[2], folds[81]
        result["policies"][name] = {}
        for method in pv.METHODS:
            s2 = s2_all[method]
            s81 = s81_all[method]
            kappa_pooled = pv.ks_distance(s2, s81)
            covered, _, _ = pv._cross_tau_detail(
                s2,
                s81,
                f2,
                f81,
                pv.ALPHA,
                scores[81]["scale"].get(method),
                scores[81]["base_width"].get(method),
            )
            stored = source["event_conformal"]["policies"][name]["methods"][method]
            rows = [
                _fold_row(s2, s81, f2, f81, fold, kappa_pooled)
                for fold in range(pv.N_FOLDS)
            ]
            summary = {}
            for key in (
                "kappa_hat_strict",
                "two_a_strict",
                "two_a_pooled_kappa",
                "two_b_strict",
                "two_b_pooled_kappa",
                "cor22_strict",
                "cor22_pooled_kappa",
            ):
                vals = [row[key] for row in rows]
                summary[key] = {"min": min(vals), "max": max(vals)}
            result["policies"][name][method] = {
                "kappa_hat_pooled": kappa_pooled,
                "coverage_recomputed": float(covered.mean()),
                "coverage_stored": float(
                    stored[DIRECTION]["coverage"]
                ),
                "coverage_match": bool(
                    abs(float(covered.mean()) - stored[DIRECTION]["coverage"]) < 1e-12
                ),
                "kappa_pooled_match": bool(
                    abs(kappa_pooled - stored["kappa_conf_tau2_vs_tau81"]) < 1e-12
                ),
                "folds": rows,
                "summary": summary,
            }
        print(f"done {name}", flush=True)

    result["out_of_sample"] = _run_oos(
        params,
        args.oos_replicates,
        args.oos_group_seeds,
        args.oos_final_seeds,
    )
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
