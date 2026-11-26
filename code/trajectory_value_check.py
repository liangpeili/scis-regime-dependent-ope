"""G0 第 11 步：轨迹级 ``V^pi(R_eval)`` 覆盖（非 C1 循环）。

主 estimand
-----------
目标策略 ``pi`` 在一条评估路径上的**累计回报**：

* ``G``：整个 336 步窗口的累计 reward（全窗口 ``V^pi``）；
* ``G_r``：按制度 ``r`` 切段后的累计 reward（评估制度上的轨迹回报）。

两条目标策略 ``Wide_Rebalance_2pct`` / ``Narrow_Rebalance_1pct`` 不属于 6 条
logging policy，因此这是 model-based off-policy 评估，不是“自己评自己”。

估计量
------
* ``identified``：用 G0 估出的 ``h_r``、``kappa/mu/jump``，在每条路径自己的
  开盘收益上估 ``sigma``；从 ``P_0`` 的边缘后验（制度相关初始化）出发，
  按制度生成 336 步的 ``(S,P)``，目标策略在生成的 ``P`` 上行动。
* ``naive``：把观测到的 ``P`` 当作 ``S``，用该路径自己的 ``AR(1)``
  （``kappa_P, mu_P, sigma_all``）生成路径；冻结期不知道 ``P`` 被冻结。
* ``segment_wise``（``G_r`` 的主估计量）：对每个制度段 ``[s,e]``，用
  ``filter_path`` 在 ``P_{0:s}`` 上的后验采样 ``(log S_s, eta_s, anchor_s)``，
  从**观测到的** ``P_s`` 和**实际**仓位出发，只模拟该段内部；段与段之间用
  观测历史重新锚定。它是“给定观测状态的条件值”，不是一条连续 rollout。
* ``continuous_regime``：从连续全窗口 rollout 中按制度分解 reward，作为
  ``segment_wise`` 的诊断对照（仓位在段间不重新锚定）。

覆盖协议
--------
* 校准集与测试集是**独立**路径：校准 seeds 2000–2127、测试 seeds 1000–1127，
  各 128 seeds × 4 windows = 512 条；真实 ``S`` 只用于打分。
* ``G``：within-window 阳性对照（同窗口校准/测试）与 leave-one-window-out
  （用其余三个窗口校准，测试留出窗口）。
* ``G_r``：within-regime 阳性对照与 leave-one-regime-out（用其余四个制度
  校准，测试留出制度）。
* score 与第 6 步同口径：raw、studentized（除以模型 MC sd）、CQR
  （``max(q05-y, y-q95)``）；另报 plug-in normal 区间
  ``mean ± z_{0.95} sd`` 作为“标准 model-based OPE”基线。
* 区间宽度：raw ``2q``；studentized ``2q*sd``；CQR ``base+2q``；
  plug-in normal ``2*1.645*sd``。
* 配对比较：同一测试路径上的 identified vs naive 覆盖差，按路径做 2000 次
  bootstrap，条件在已拟合的标定分位数上；full-window 用 ``continuous``，
  regime-level 用 ``segment_wise``。它只量化测试路径抽样，不含标定集不确定性。

只读 ``results/policy_value_check.json`` 里的 G0 参数估计（``kappa/mu/jump``
与 ``h``），只写 ``results/trajectory_value_check.json``；不修改
``synthetic_env.py`` / ``generate_dataset.py`` / ``coverage_eval.py``。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from synthetic_env import (
    FEE_RATE,
    LIQUIDITY_SHARE,
    REGIME_EXTENDED,
    REGIME_HALT,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    V0,
    Position,
    simulate_market,
)
import policy_value_check as pvc

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "trajectory_value_check.json"
PVC_JSON = ROOT / "results" / "policy_value_check.json"

N_STEPS = pvc.N_STEPS
WINDOWS = pvc.WINDOWS
ALPHA = pvc.ALPHA
POLICIES = pvc.POLICIES
Z90 = 1.6448536269514722

REGIME_IDS = (
    REGIME_OPEN,
    REGIME_EXTENDED,
    REGIME_WEEKEND,
    REGIME_HOLIDAY,
    REGIME_HALT,
)
REGIME_LABELS = {
    REGIME_OPEN: "OPEN",
    REGIME_EXTENDED: "EXTENDED",
    REGIME_WEEKEND: "WEEKEND",
    REGIME_HOLIDAY: "HOLIDAY",
    REGIME_HALT: "HALT",
}
REGIME_ORDER = tuple(REGIME_LABELS[r] for r in REGIME_IDS)

CAL_SEEDS = tuple(range(2000, 2128))
EVAL_SEEDS = tuple(range(1000, 1128))
R_REPS = 32

CONFORMAL_METHODS = (
    "identified_raw",
    "identified_studentized",
    "identified_cqr",
    "naive_raw",
    "naive_studentized",
    "naive_cqr",
)
PLUGIN_METHODS = (
    "identified_plugin_normal",
    "naive_plugin_normal",
)
ALL_METHODS = CONFORMAL_METHODS + PLUGIN_METHODS


# ---------------------------------------------------------------------------
# Truth and observable state
# ---------------------------------------------------------------------------


def rollout_steps(
    s: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    volume: np.ndarray,
    policy,
) -> np.ndarray:
    """Per-step reward, mirroring ``synthetic_env.simulate_strategy``.

    The returned array has length ``len(regime)``; entry ``t`` is the reward
    accrued between ``t`` and ``t+1`` and belongs to regime ``regime[t]``.
    """
    pos = Position()
    prev_il_true = 0.0
    rewards = np.empty(len(regime), dtype=float)
    for t in range(len(regime)):
        price_from = float(p[t])
        price_to = float(p[t + 1])
        true_to = float(s[t + 1])
        r = int(regime[t])
        next_r = int(regime[t + 1]) if t + 1 < len(regime) else r
        action = policy(t, r, next_r, price_from, pos)
        cost, changed = pvc.apply_action(pos, action, price_from)

        if changed:
            prev_il_true = 0.0

        in_range = bool(pos.in_pos and pos.lower <= price_from <= pos.upper)
        fee = FEE_RATE * float(volume[t]) * LIQUIDITY_SHARE * float(in_range)

        if pos.in_pos:
            il_true_now = pos.il(true_to)
            il_true_step = il_true_now - prev_il_true
            prev_il_true = il_true_now
        else:
            il_true_step = 0.0
            prev_il_true = 0.0

        rewards[t] = (fee + il_true_step - cost) / V0
    return rewards


def regime_segments(regime: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous ``(regime, start, end)`` segments, inclusive."""
    if len(regime) == 0:
        return []
    out: list[tuple[int, int, int]] = []
    start = 0
    for t in range(1, len(regime)):
        if int(regime[t]) != int(regime[start]):
            out.append((int(regime[start]), start, t - 1))
            start = t
    out.append((int(regime[start]), start, len(regime) - 1))
    return out


def positions_at_segments(
    market, policy, segments: list[tuple[int, int, int]]
) -> list[Position]:
    """Observable policy position *before* the action at each segment start."""
    starts = {s: i for i, (_r, s, _e) in enumerate(segments)}
    out: list[Position | None] = [None] * len(segments)
    pos = Position()
    regime = market.regime
    for t in range(len(regime)):
        if t in starts:
            out[starts[t]] = copy.deepcopy(pos)
        r = int(regime[t])
        next_r = int(regime[t + 1]) if t + 1 < len(regime) else r
        action = policy(t, r, next_r, float(market.pool_price[t]), pos)
        pvc.apply_action(pos, action, float(market.pool_price[t]))
    if any(x is None for x in out):
        raise RuntimeError("a segment start was not visited")
    return [x for x in out if x is not None]


def truth_totals(market, policy) -> dict:
    rewards = rollout_steps(
        market.true_price,
        market.pool_price,
        market.regime,
        market.volume,
        policy,
    )
    total = float(rewards.sum())
    by_regime = {
        REGIME_LABELS[r]: float(rewards[market.regime == r].sum())
        for r in REGIME_IDS
    }
    segments = regime_segments(market.regime)
    by_segment = [float(rewards[s : e + 1].sum()) for _r, s, e in segments]
    return {
        "total": total,
        "by_regime": by_regime,
        "by_segment": by_segment,
        "segments": segments,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Full-path Kalman filter
# ---------------------------------------------------------------------------


def _initial_belief(market, h: dict) -> tuple[np.ndarray, np.ndarray]:
    """Posterior over ``(log S_0, eta_0, log S_last_open)`` given ``P_0``.

    ``simulate_market`` initialises ``last_open_price = true_price[0]`` and
    ``eta = 0`` *before* the t=0 regime transition, so at t=0 the anchor equals
    the latent price for every regime.  Conditioning on the observed ``P_0``
    then gives a rank-one (or zero) Gaussian:
    """
    r = int(market.regime[0])
    log_p0 = float(np.log(market.pool_price[0]))
    mean = np.array([log_p0, 0.0, log_p0], dtype=float)
    cov = np.zeros((3, 3), dtype=float)
    if r == REGIME_OPEN:
        v = h["open_sd"] ** 2
        cov[0, 0] = v
        cov[2, 2] = v
        cov[0, 2] = cov[2, 0] = v
    elif r == REGIME_HALT:
        pass
    else:
        v = h["ext_sd"] ** 2 if r == REGIME_EXTENDED else h["stale_sd"] ** 2
        cov[0, 0] = v
        cov[1, 1] = v
        cov[2, 2] = v
        cov[0, 1] = cov[1, 0] = -v
        cov[0, 2] = cov[2, 0] = v
        cov[1, 2] = cov[2, 1] = -v
    return mean, cov


def filter_path(
    market,
    kappa: float,
    mu: float,
    sigma: float,
    h: dict,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Filter the identified model on the full observed ``P`` history.

    Returns the posterior at every ``t = 0..N_STEPS``; ``posterior[s]`` is
    conditioned on ``P_{0:s}`` and is the belief used at the start of a
    segment-wise rollout.
    """
    regime = market.regime
    logp = np.log(market.pool_price)
    mean, cov = _initial_belief(market, h)
    posterior: dict[int, tuple[np.ndarray, np.ndarray]] = {
        0: (mean.copy(), cov.copy())
    }
    for t in range(1, N_STEPS + 1):
        r = int(regime[t]) if t < len(regime) else int(regime[-1])
        mean, cov = pvc._predict_state(mean, cov, kappa, mu, sigma, h, r)
        obs_vec, obs_var = pvc._obs_model(r, h)
        if obs_vec is not None:
            mean, cov, _gain = pvc._update_state(
                mean, cov, float(logp[t]), obs_vec, obs_var
            )
        posterior[t] = (mean.copy(), cov.copy())
    return posterior


# ---------------------------------------------------------------------------
# Model rollouts
# ---------------------------------------------------------------------------


def _simulate_identified_forward(
    rng: np.random.Generator,
    market,
    start: int,
    end: int,
    belief_mean: np.ndarray,
    belief_cov: np.ndarray,
    kappa: float,
    mu: float,
    jump_prob: float,
    sigma: float,
    h: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``(S, P)`` for steps ``start..end+1``.

    ``p[0]`` is the observed ``P_start`` (the policy's first decision uses it);
    ``p[1:]`` and all ``s`` are model draws.  The latent state is sampled from
    the supplied posterior, which is conditioned on ``P_{0:start}``.
    """
    ls, eta, ls_last = pvc.sample_pre_state(rng, belief_mean, belief_cov)
    s_list = [math.exp(ls)]
    p_list = [float(market.pool_price[start])]
    for t in range(start + 1, end + 2):
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < jump_prob else 0.0
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


def _simulate_naive_forward(
    rng: np.random.Generator,
    market,
    start: int,
    end: int,
    kappa_p: float,
    mu_p: float,
    jump_prob: float,
    sigma: float,
) -> np.ndarray:
    """Naive model: observed ``P`` is the state, no observation mapping."""
    lp = math.log(float(market.pool_price[start]))
    p_list = [math.exp(lp)]
    for _t in range(start + 1, end + 2):
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < jump_prob else 0.0
        lp = lp + kappa_p * (mu_p - lp) + sigma * eps + jump
        p_list.append(math.exp(lp))
    return np.asarray(p_list)


def _identified_segment_reward(
    rng: np.random.Generator,
    market,
    start: int,
    end: int,
    policy,
    pos0: Position,
    belief_mean: np.ndarray,
    belief_cov: np.ndarray,
    kappa: float,
    mu: float,
    jump_prob: float,
    sigma: float,
    h: dict,
) -> float:
    s, p = _simulate_identified_forward(
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
    )
    il0 = pos0.il(float(s[0])) if pos0.in_pos else 0.0
    return pvc.rollout_segment(
        s,
        p,
        market.regime[start : end + 1],
        market.volume[start : end + 1],
        policy,
        copy.deepcopy(pos0),
        il0,
        t_offset=start,
    )


def _naive_segment_reward(
    rng: np.random.Generator,
    market,
    start: int,
    end: int,
    policy,
    pos0: Position,
    kappa_p: float,
    mu_p: float,
    jump_prob: float,
    sigma: float,
) -> float:
    p = _simulate_naive_forward(
        rng, market, start, end, kappa_p, mu_p, jump_prob, sigma
    )
    il0 = pos0.il(float(p[0])) if pos0.in_pos else 0.0
    return pvc.rollout_segment(
        p,
        p,
        market.regime[start : end + 1],
        market.volume[start : end + 1],
        policy,
        copy.deepcopy(pos0),
        il0,
        t_offset=start,
    )


def build_path_record(
    market,
    window: str,
    seed: int,
    policy,
    inputs: dict,
    reps: int,
) -> dict:
    """Truth and model draws for one ``(window, seed, policy)`` path."""
    kappa = inputs["kappa"]
    mu = inputs["mu"]
    jump_prob = inputs["jump_prob"]
    h = inputs["h"]

    sigma = pvc.estimate_sigma_open(market)
    sigma_all = pvc.estimate_sigma_all(market)
    kappa_p, mu_p = pvc.estimate_naive_kappa([market])

    truth = truth_totals(market, policy)
    segments = truth["segments"]
    pos_at = positions_at_segments(market, policy, segments)
    posterior = filter_path(market, kappa, mu, sigma, h)

    cont_ident_total = np.empty(reps, dtype=float)
    cont_ident_regime = np.zeros((reps, len(REGIME_IDS)), dtype=float)
    cont_naive_total = np.empty(reps, dtype=float)
    cont_naive_regime = np.zeros((reps, len(REGIME_IDS)), dtype=float)
    seg_ident_total = np.zeros(reps, dtype=float)
    seg_ident_regime = np.zeros((reps, len(REGIME_IDS)), dtype=float)
    seg_naive_total = np.zeros(reps, dtype=float)
    seg_naive_regime = np.zeros((reps, len(REGIME_IDS)), dtype=float)

    # A policy-independent RNG stream: the two policies see the same model
    # paths, which makes the identified/naive and Wide/Narrow comparisons paired.
    rng = np.random.default_rng([int(seed), WINDOWS.index(window), 11011])

    for r in range(reps):
        s, p = _simulate_identified_forward(
            rng,
            market,
            0,
            N_STEPS - 1,
            posterior[0][0],
            posterior[0][1],
            kappa,
            mu,
            jump_prob,
            sigma,
            h,
        )
        rew = rollout_steps(s, p, market.regime, market.volume, policy)
        cont_ident_total[r] = float(rew.sum())
        for k, rid in enumerate(REGIME_IDS):
            cont_ident_regime[r, k] = float(rew[market.regime == rid].sum())

        p_naive = _simulate_naive_forward(
            rng, market, 0, N_STEPS - 1, kappa_p, mu_p, jump_prob, sigma_all
        )
        rew = rollout_steps(
            p_naive, p_naive, market.regime, market.volume, policy
        )
        cont_naive_total[r] = float(rew.sum())
        for k, rid in enumerate(REGIME_IDS):
            cont_naive_regime[r, k] = float(rew[market.regime == rid].sum())

        for seg_idx, (rid, start, end) in enumerate(segments):
            k = REGIME_IDS.index(rid)
            belief_mean, belief_cov = posterior[start]
            val = _identified_segment_reward(
                rng,
                market,
                start,
                end,
                policy,
                pos_at[seg_idx],
                belief_mean,
                belief_cov,
                kappa,
                mu,
                jump_prob,
                sigma,
                h,
            )
            seg_ident_total[r] += val
            seg_ident_regime[r, k] += val

            val = _naive_segment_reward(
                rng,
                market,
                start,
                end,
                policy,
                pos_at[seg_idx],
                kappa_p,
                mu_p,
                jump_prob,
                sigma_all,
            )
            seg_naive_total[r] += val
            seg_naive_regime[r, k] += val

    return {
        "window": window,
        "seed": int(seed),
        "sigma_open": float(sigma),
        "sigma_all": float(sigma_all),
        "kappa_p": float(kappa_p),
        "mu_p": float(mu_p),
        "truth_total": truth["total"],
        "truth_regime": np.array(
            [truth["by_regime"][label] for label in REGIME_ORDER], dtype=float
        ),
        "truth_segment": np.asarray(truth["by_segment"], dtype=float),
        "segments": segments,
        "cont": {
            "ident": {"total": cont_ident_total, "regime": cont_ident_regime},
            "naive": {"total": cont_naive_total, "regime": cont_naive_regime},
        },
        "seg": {
            "ident": {"total": seg_ident_total, "regime": seg_ident_regime},
            "naive": {"total": seg_naive_total, "regime": seg_naive_regime},
        },
    }


# ---------------------------------------------------------------------------
# Conformal coverage
# ---------------------------------------------------------------------------


def _summary(
    covered: np.ndarray, widths: np.ndarray, quantiles: list[float]
) -> dict:
    n = int(len(covered))
    cov = float(np.mean(covered))
    out = {
        "n_test": n,
        "coverage": cov,
        "coverage_se": float(math.sqrt(cov * (1.0 - cov) / n)) if n else float("nan"),
        "mean_width": float(np.mean(widths)) if n else float("nan"),
        "median_width": float(np.median(widths)) if n else float("nan"),
    }
    if quantiles:
        out.update(
            {
                "mean_q": float(np.mean(quantiles)),
                "min_q": float(np.min(quantiles)),
                "max_q": float(np.max(quantiles)),
            }
        )
    return out


def _conformal(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    test_scale: np.ndarray | None = None,
    test_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    q = pvc.conformal_quantile(cal_scores, alpha)
    covered = test_scores <= q
    if test_base is not None:
        widths = test_base + 2.0 * q
    elif test_scale is not None:
        widths = 2.0 * q * test_scale
    else:
        widths = np.full(len(test_scores), 2.0 * q)
    return covered, widths, float(q)


def _pool(parts: list[tuple[np.ndarray, np.ndarray, list[float]]]) -> dict:
    covered = np.concatenate([p[0] for p in parts])
    widths = np.concatenate([p[1] for p in parts])
    quantiles = [q for p in parts for q in p[2]]
    return _summary(covered, widths, quantiles)


def _group_block(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_groups: np.ndarray,
    test_groups: np.ndarray,
    alpha: float,
    test_scale: np.ndarray | None = None,
    test_base: np.ndarray | None = None,
    leave_one_out: bool = False,
) -> dict:
    parts = []
    per_group: dict[str, dict] = {}
    for group in sorted(set(test_groups.tolist())):
        test_mask = test_groups == group
        if leave_one_out:
            cal = cal_scores[cal_groups != group]
        else:
            cal = cal_scores[cal_groups == group]
        scale = test_scale[test_mask] if test_scale is not None else None
        base = test_base[test_mask] if test_base is not None else None
        covered, widths, q = _conformal(
            cal, test_scores[test_mask], alpha, scale, base
        )
        parts.append((covered, widths, [q]))
        per_group[str(group)] = _summary(covered, widths, [q])
    out = _pool(parts)
    out["per_group"] = per_group
    return out


def _plugin_normal(
    truth: np.ndarray, mean: np.ndarray, sd: np.ndarray
) -> tuple[dict, np.ndarray]:
    diff = np.abs(truth - mean)
    covered = diff <= Z90 * np.maximum(sd, 0.0)
    widths = 2.0 * Z90 * sd
    return _summary(covered, widths, []), covered


def _score_arrays(
    truth_total: np.ndarray,
    mean_total: np.ndarray,
    sd_total: np.ndarray,
    q05_total: np.ndarray,
    q95_total: np.ndarray,
    truth_regime: np.ndarray,
    mean_regime: np.ndarray,
    sd_regime: np.ndarray,
    q05_regime: np.ndarray,
    q95_regime: np.ndarray,
) -> dict:
    """Raw/studentized/CQR scores for the total and per-regime units."""
    eps = 1e-12
    return {
        "total": {
            "raw": np.abs(truth_total - mean_total),
            "studentized": np.abs(truth_total - mean_total)
            / np.maximum(sd_total, eps),
            "cqr": np.maximum(q05_total - truth_total, truth_total - q95_total),
            "scale": sd_total,
            "base": q95_total - q05_total,
        },
        "regime": {
            "raw": np.abs(truth_regime - mean_regime),
            "studentized": np.abs(truth_regime - mean_regime)
            / np.maximum(sd_regime, eps),
            "cqr": np.maximum(q05_regime - truth_regime, truth_regime - q95_regime),
            "scale": sd_regime,
            "base": q95_regime - q05_regime,
        },
    }


def _stack(records: list[dict], model: str, method: str) -> dict:
    total = np.stack([r[model][method]["total"] for r in records])
    regime = np.stack([r[model][method]["regime"] for r in records])
    return {
        "truth_total": np.array([r["truth_total"] for r in records]),
        "truth_regime": np.stack([r["truth_regime"] for r in records]),
        "mean_total": total.mean(axis=1),
        "sd_total": total.std(axis=1, ddof=1),
        "q05_total": np.quantile(total, 0.05, axis=1),
        "q95_total": np.quantile(total, 0.95, axis=1),
        "mean_regime": regime.mean(axis=1),
        "sd_regime": regime.std(axis=1, ddof=1),
        "q05_regime": np.quantile(regime, 0.05, axis=1),
        "q95_regime": np.quantile(regime, 0.95, axis=1),
        "window": np.array([r["window"] for r in records]),
    }


def _truth_summary(values: np.ndarray) -> dict:
    n = int(len(values))
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "se": float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "sd": float(np.std(values, ddof=1)) if n > 1 else float("nan"),
    }


def _regime_truth_summary(values: np.ndarray) -> dict:
    return {
        label: _truth_summary(values[:, k]) for k, label in enumerate(REGIME_ORDER)
    }


def _bias_block(
    truth_total: np.ndarray,
    mean_total: np.ndarray,
    truth_regime: np.ndarray,
    mean_regime: np.ndarray,
) -> dict:
    return {
        "full": float(np.mean(mean_total - truth_total)),
        "by_regime": {
            label: float(np.mean(mean_regime[:, k] - truth_regime[:, k]))
            for k, label in enumerate(REGIME_ORDER)
        },
    }


def _path_bootstrap(
    diff: np.ndarray, n_boot: int = 2000, seed: int = 20261031
) -> dict:
    """Path-level bootstrap for a paired coverage difference.

    ``diff`` is ``(n_paths,)`` or ``(n_paths, n_regimes)``; resampling whole
    paths preserves the within-path dependence across regimes.
    """
    rng = np.random.default_rng(seed)
    n = int(diff.shape[0])
    obs = float(diff.mean())
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = float(diff[idx].mean())
    return {
        "diff": obs,
        "se": float(vals.std(ddof=1)),
        "ci95": [
            float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975)),
        ],
    }


def _coverage_pair(covered_id: np.ndarray, covered_naive: np.ndarray) -> dict:
    out = {
        "coverage_identified": float(covered_id.mean()),
        "coverage_naive": float(covered_naive.mean()),
    }
    out.update(_path_bootstrap(covered_id.astype(float) - covered_naive.astype(float)))
    return out


def _paired_full_window(
    cal_id: dict,
    ev_id: dict,
    cal_naive: dict,
    ev_naive: dict,
    alpha: float,
) -> dict:
    sid_cal = _score_arrays(
        cal_id["truth_total"], cal_id["mean_total"], cal_id["sd_total"],
        cal_id["q05_total"], cal_id["q95_total"], cal_id["truth_regime"],
        cal_id["mean_regime"], cal_id["sd_regime"], cal_id["q05_regime"],
        cal_id["q95_regime"],
    )
    sid_ev = _score_arrays(
        ev_id["truth_total"], ev_id["mean_total"], ev_id["sd_total"],
        ev_id["q05_total"], ev_id["q95_total"], ev_id["truth_regime"],
        ev_id["mean_regime"], ev_id["sd_regime"], ev_id["q05_regime"],
        ev_id["q95_regime"],
    )
    sna_cal = _score_arrays(
        cal_naive["truth_total"], cal_naive["mean_total"], cal_naive["sd_total"],
        cal_naive["q05_total"], cal_naive["q95_total"], cal_naive["truth_regime"],
        cal_naive["mean_regime"], cal_naive["sd_regime"],
        cal_naive["q05_regime"], cal_naive["q95_regime"],
    )
    sna_ev = _score_arrays(
        ev_naive["truth_total"], ev_naive["mean_total"], ev_naive["sd_total"],
        ev_naive["q05_total"], ev_naive["q95_total"], ev_naive["truth_regime"],
        ev_naive["mean_regime"], ev_naive["sd_regime"],
        ev_naive["q05_regime"], ev_naive["q95_regime"],
    )

    out: dict = {}
    for score in ("studentized", "cqr"):
        covered_id = np.zeros(len(ev_id["truth_total"]), dtype=bool)
        covered_naive = np.zeros(len(ev_naive["truth_total"]), dtype=bool)
        for w in sorted(set(ev_id["window"].tolist())):
            cal_mask_id = cal_id["window"] != w
            cal_mask_naive = cal_naive["window"] != w
            test_mask = ev_id["window"] == w
            q_id = pvc.conformal_quantile(
                sid_cal["total"][score][cal_mask_id], alpha
            )
            q_naive = pvc.conformal_quantile(
                sna_cal["total"][score][cal_mask_naive], alpha
            )
            covered_id[test_mask] = sid_ev["total"][score][test_mask] <= q_id
            covered_naive[test_mask] = (
                sna_ev["total"][score][test_mask] <= q_naive
            )
        out[score] = _coverage_pair(covered_id, covered_naive)

    diff_id = np.abs(ev_id["truth_total"] - ev_id["mean_total"])
    diff_naive = np.abs(ev_naive["truth_total"] - ev_naive["mean_total"])
    covered_id = diff_id <= Z90 * np.maximum(ev_id["sd_total"], 0.0)
    covered_naive = diff_naive <= Z90 * np.maximum(ev_naive["sd_total"], 0.0)
    out["plugin_normal"] = _coverage_pair(covered_id, covered_naive)
    return out


def _paired_regime(
    cal_id: dict,
    ev_id: dict,
    cal_naive: dict,
    ev_naive: dict,
    alpha: float,
) -> dict:
    sid_cal = _score_arrays(
        cal_id["truth_total"], cal_id["mean_total"], cal_id["sd_total"],
        cal_id["q05_total"], cal_id["q95_total"], cal_id["truth_regime"],
        cal_id["mean_regime"], cal_id["sd_regime"], cal_id["q05_regime"],
        cal_id["q95_regime"],
    )
    sid_ev = _score_arrays(
        ev_id["truth_total"], ev_id["mean_total"], ev_id["sd_total"],
        ev_id["q05_total"], ev_id["q95_total"], ev_id["truth_regime"],
        ev_id["mean_regime"], ev_id["sd_regime"], ev_id["q05_regime"],
        ev_id["q95_regime"],
    )
    sna_cal = _score_arrays(
        cal_naive["truth_total"], cal_naive["mean_total"], cal_naive["sd_total"],
        cal_naive["q05_total"], cal_naive["q95_total"], cal_naive["truth_regime"],
        cal_naive["mean_regime"], cal_naive["sd_regime"],
        cal_naive["q05_regime"], cal_naive["q95_regime"],
    )
    sna_ev = _score_arrays(
        ev_naive["truth_total"], ev_naive["mean_total"], ev_naive["sd_total"],
        ev_naive["q05_total"], ev_naive["q95_total"], ev_naive["truth_regime"],
        ev_naive["mean_regime"], ev_naive["sd_regime"],
        ev_naive["q05_regime"], ev_naive["q95_regime"],
    )

    out: dict = {}
    for score in ("studentized", "cqr"):
        covered_id = np.zeros_like(ev_id["truth_regime"], dtype=bool)
        covered_naive = np.zeros_like(ev_naive["truth_regime"], dtype=bool)
        for k, _label in enumerate(REGIME_ORDER):
            others = [j for j in range(len(REGIME_ORDER)) if j != k]
            q_id = pvc.conformal_quantile(
                sid_cal["regime"][score][:, others].ravel(), alpha
            )
            q_naive = pvc.conformal_quantile(
                sna_cal["regime"][score][:, others].ravel(), alpha
            )
            covered_id[:, k] = sid_ev["regime"][score][:, k] <= q_id
            covered_naive[:, k] = sna_ev["regime"][score][:, k] <= q_naive
        diff = covered_id.astype(float) - covered_naive.astype(float)
        block = _path_bootstrap(diff)
        block["coverage_identified"] = float(covered_id.mean())
        block["coverage_naive"] = float(covered_naive.mean())
        block["per_regime"] = {
            label: {
                "coverage_identified": float(covered_id[:, k].mean()),
                "coverage_naive": float(covered_naive[:, k].mean()),
                "diff": float(diff[:, k].mean()),
                "se": float(diff[:, k].std(ddof=1) / math.sqrt(len(diff))),
            }
            for k, label in enumerate(REGIME_ORDER)
        }
        out[score] = block

    diff_id = np.abs(ev_id["truth_regime"] - ev_id["mean_regime"])
    diff_naive = np.abs(ev_naive["truth_regime"] - ev_naive["mean_regime"])
    covered_id = diff_id <= Z90 * np.maximum(ev_id["sd_regime"], 0.0)
    covered_naive = diff_naive <= Z90 * np.maximum(ev_naive["sd_regime"], 0.0)
    diff = covered_id.astype(float) - covered_naive.astype(float)
    block = _path_bootstrap(diff)
    block["coverage_identified"] = float(covered_id.mean())
    block["coverage_naive"] = float(covered_naive.mean())
    block["per_regime"] = {
        label: {
            "coverage_identified": float(covered_id[:, k].mean()),
            "coverage_naive": float(covered_naive[:, k].mean()),
            "diff": float(diff[:, k].mean()),
            "se": float(diff[:, k].std(ddof=1) / math.sqrt(len(diff))),
        }
        for k, label in enumerate(REGIME_ORDER)
    }
    out["plugin_normal"] = block
    return out


def _evaluate_model(
    cal: dict,
    ev: dict,
    alpha: float,
) -> dict:
    """All protocols for one (model kind, estimator) pair."""
    scores_cal = _score_arrays(
        cal["truth_total"],
        cal["mean_total"],
        cal["sd_total"],
        cal["q05_total"],
        cal["q95_total"],
        cal["truth_regime"],
        cal["mean_regime"],
        cal["sd_regime"],
        cal["q05_regime"],
        cal["q95_regime"],
    )
    scores_ev = _score_arrays(
        ev["truth_total"],
        ev["mean_total"],
        ev["sd_total"],
        ev["q05_total"],
        ev["q95_total"],
        ev["truth_regime"],
        ev["mean_regime"],
        ev["sd_regime"],
        ev["q05_regime"],
        ev["q95_regime"],
    )
    out: dict = {"bias": _bias_block(
        ev["truth_total"], ev["mean_total"], ev["truth_regime"], ev["mean_regime"]
    )}

    # Full-window G.
    full: dict = {}
    for score in ("raw", "studentized", "cqr"):
        scale = scores_ev["total"]["scale"] if score == "studentized" else None
        base = scores_ev["total"]["base"] if score == "cqr" else None
        full[score] = {
            "within_window": _group_block(
                scores_cal["total"][score],
                scores_ev["total"][score],
                cal["window"],
                ev["window"],
                alpha,
                scale,
                base,
                leave_one_out=False,
            ),
            "leave_one_window_out": _group_block(
                scores_cal["total"][score],
                scores_ev["total"][score],
                cal["window"],
                ev["window"],
                alpha,
                scale,
                base,
                leave_one_out=True,
            ),
        }
    plugin, plugin_cov = _plugin_normal(
        ev["truth_total"], ev["mean_total"], ev["sd_total"]
    )
    plugin["per_group"] = {
        str(w): _summary(
            plugin_cov[ev["window"] == w],
            2.0 * Z90 * ev["sd_total"][ev["window"] == w],
            [],
        )
        for w in sorted(set(ev["window"].tolist()))
    }
    full["plugin_normal"] = plugin
    out["full_window"] = full

    # Regime-level G_r.
    regime_out: dict = {}
    for score in ("raw", "studentized", "cqr"):
        scale = scores_ev["regime"]["scale"] if score == "studentized" else None
        base = scores_ev["regime"]["base"] if score == "cqr" else None
        within = {"per_group": {}}
        loro = {"per_group": {}}
        parts_within = []
        parts_loro = []
        for k, label in enumerate(REGIME_ORDER):
            cal_s = scores_cal["regime"][score][:, k]
            test_s = scores_ev["regime"][score][:, k]
            sc = scale[:, k] if scale is not None else None
            bs = base[:, k] if base is not None else None
            c, w, q = _conformal(cal_s, test_s, alpha, sc, bs)
            parts_within.append((c, w, [q]))
            within["per_group"][label] = _summary(c, w, [q])
            cal_other = scores_cal["regime"][score][
                :, [j for j in range(len(REGIME_ORDER)) if j != k]
            ].ravel()
            c, w, q = _conformal(cal_other, test_s, alpha, sc, bs)
            parts_loro.append((c, w, [q]))
            loro["per_group"][label] = _summary(c, w, [q])
        within.update(_pool(parts_within))
        loro.update(_pool(parts_loro))
        regime_out[score] = {
            "within_regime": within,
            "leave_one_regime_out": loro,
        }

    plugin_regime: dict = {"per_group": {}}
    plugin_cov_regime = np.zeros_like(ev["truth_regime"], dtype=bool)
    for k, label in enumerate(REGIME_ORDER):
        c, cov = _plugin_normal(
            ev["truth_regime"][:, k],
            ev["mean_regime"][:, k],
            ev["sd_regime"][:, k],
        )
        plugin_regime["per_group"][label] = c
        plugin_cov_regime[:, k] = cov
    plugin_regime["all"] = _summary(
        plugin_cov_regime.ravel(),
        (2.0 * Z90 * ev["sd_regime"]).ravel(),
        [],
    )
    regime_out["plugin_normal"] = plugin_regime
    out["regime_level"] = regime_out
    return out


def _method_prefix(method: str) -> tuple[str, str]:
    if method.startswith("identified_"):
        return "ident", method[len("identified_") :]
    if method.startswith("naive_"):
        return "naive", method[len("naive_") :]
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-seeds", type=int, default=len(CAL_SEEDS))
    parser.add_argument("--eval-seeds", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--reps", type=int, default=R_REPS)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument(
        "--h-json",
        type=Path,
        default=None,
        help="optional JSON with an h_estimate block (observable-only h); "
        "defaults to the G0 true-S diagnostic h",
    )
    args = parser.parse_args()

    raw = json.loads(PVC_JSON.read_text(encoding="utf-8"))
    params = raw["parameter_estimates"]
    if args.h_json is None:
        h_raw = raw["observation_mapping_estimate"]
        h_source = "results/policy_value_check.json (G0 true-S diagnostic)"
    else:
        h_doc = json.loads(args.h_json.read_text(encoding="utf-8"))
        h_raw = h_doc["h_estimate"]
        h_source = str(args.h_json)
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

    cal_seeds = CAL_SEEDS[: args.cal_seeds]
    eval_seeds = EVAL_SEEDS[: args.eval_seeds]

    truth_note = "open_sd/ext_phi are weakly identified from P alone"
    if args.h_json is not None and h_doc.get("h_truth") is not None:
        truth_note = (
            f"its ext_phi point estimate is {h['ext_phi']:.4f} vs simulator "
            f"truth {float(h_doc['h_truth']['ext_phi']):.4f}; open_sd/ext_phi "
            "are weakly identified from P alone"
        )

    if args.h_json is None:
        h_label = "G0 true-S diagnostic h"
        h_limit = (
            "The identified model's h (open_sd/ext_phi/ext_sd/stale_phi/"
            "stale_sd) is a G0 estimate obtained with true-S diagnostics, so "
            "the pipeline as run is not observable-only; reconstructing h "
            "from observables alone is not implemented here (Props 1-2)."
        )
        fixed_limit = (
            "kappa/mu/jump and h are fixed G0 estimates shared by calibration "
            "and test; they are not re-estimated inside folds."
        )
    else:
        h_label = "observable-only h"
        h_limit = (
            f"h is the observable-only estimate from {h_source} "
            "(code/observable_h_check.py). It uses only P and the calendar; "
            f"{truth_note} (see results/observable_h_check.json for bootstrap/"
            "Hessian SEs). The pipeline is observable-only for h, not for "
            "kappa/mu/jump."
        )
        fixed_limit = (
            "kappa/mu/jump are fixed G0 true-S estimates and h is fixed at "
            "the observable-only estimate; neither is re-estimated inside "
            "folds."
        )

    results: dict = {
        "config": {
            "estimand": "G = full 336-step cumulative reward; G_r = cumulative "
            "reward over all segments of regime r",
            "cal_seeds": [int(cal_seeds[0]), int(cal_seeds[-1])],
            "eval_seeds": [int(eval_seeds[0]), int(eval_seeds[-1])],
            "n_cal_paths": len(cal_seeds) * len(WINDOWS),
            "n_eval_paths": len(eval_seeds) * len(WINDOWS),
            "windows": list(WINDOWS),
            "reps": int(args.reps),
            "alpha": ALPHA,
            "z90": Z90,
            "policies": list(POLICIES),
            "regimes": list(REGIME_ORDER),
            "methods": list(ALL_METHODS),
            "score_definitions": {
                "raw": "|truth - model_mean|",
                "studentized": "|truth - model_mean| / model_sd",
                "cqr": "max(q05 - truth, truth - q95)",
                "plugin_normal": "|truth - model_mean| <= 1.645 * model_sd",
            },
            "protocols": {
                "within_window": "calibrate and test on the same window",
                "leave_one_window_out": "calibrate on the other three windows",
                "within_regime": "calibrate and test on the same regime",
                "leave_one_regime_out": "calibrate on the other four regimes",
            },
            "estimators": {
                "identified": f"{h_label} + G0 kappa/mu/jump; per-path "
                "sigma_open; filter_path belief at each segment start; "
                "regime-aware P",
                "naive": "P treated as S; per-path kappa_P/mu_P/sigma_all; "
                "no freeze knowledge",
                "continuous": "one model rollout from t=0, conditioned on P_0; "
                "regime decomposition of the same rollout",
                "segment_wise": "per-regime segment rollouts re-anchored at the "
                "observed position and filtered belief at each segment start",
            },
            "paired_comparison": "identified vs naive on the same test paths; "
            "path-level bootstrap with 2000 resamples, conditioning on the "
            "fitted calibration quantiles; full-window uses the continuous "
            "rollout, regime-level uses the segment-wise estimator",
        },
        "inputs": {
            "parameter_source": "results/policy_value_check.json",
            "h_source": h_source,
            "kappa": inputs["kappa"],
            "mu": inputs["mu"],
            "jump_prob": inputs["jump_prob"],
            "h": h,
            "note": (
                "true S is used in this script only for scoring and truth "
                "checks; kappa/mu/jump are the G0 estimates; h is the G0 "
                "true-S diagnostic estimate (not observable-only); per-path "
                "sigma and naive parameters use only observable P"
                if args.h_json is None
                else "true S is used in this script only for scoring and "
                "truth checks; kappa/mu/jump are the G0 estimates; h is the "
                f"observable-only estimate from {h_source}; {truth_note} "
                "(see results/observable_h_check.json for SEs); per-path "
                "sigma and naive parameters use only observable P"
            ),
        },
        "data_checks": {},
        "limits": [
            "True S is used only for scoring; in deployment the calibration "
            "target (the path's realized reward) requires reconstructing S or "
            "an observable-reward variant (D7).",
            h_limit,
            "The plug-in conditions on the realized future volume path, as "
            "steps 5-6 do. Volume is exogenous in the simulator, but future "
            "volume is not known at decision time; this is a conditional-on-"
            "volume evaluation, not a fully deployable OPE.",
            "Per-path sigma_open/sigma_all and naive kappa_P/mu_P are estimated "
            "from the same path's observable P history (transductive), as in "
            "step 6; the two estimators use the same information.",
            fixed_limit,
            "The regime calendar is deterministic and known at decision time; "
            "the model is allowed to know future regime labels.",
            "segment_wise re-anchors at each segment start using the observed "
            "position and the filtered belief given P_{0:s}; it is a sum of "
            "conditional segment values, not a single continuous rollout. The "
            "continuous estimator is the standard model-based rollout.",
            "The regime-level LORO calibration pool contains the other four "
            "regime units of each calibration path; units within a path are "
            "correlated, so the effective calibration sample is smaller than "
            "the nominal 2048 and the pooled quantile has more finite-sample "
            "variability than that nominal count suggests. Calibration and "
            "test paths are disjoint, so there is no test-path contamination.",
            "Target policies are fixed heuristics, not trained RL policies; "
            "the experiment has no IPS/DR or importance weights, only "
            "model-based plug-in.",
            "Coverage is marginal over the calibration/test split and pooled "
            "over the four held-out groups; per-group numbers are descriptive "
            "with binomial SE, not simultaneous. For the regime-level pool the "
            "binomial SE over 5x512 units ignores within-path correlation and "
            "is systematically understated; inferential claims use the "
            "path-level paired bootstrap.",
            "R model draws per path propagate model/MC noise; the 5%/95% "
            "quantiles are empirical and therefore noisy at R=32.",
            "The paired identified-vs-naive bootstrap resamples test paths and "
            "conditions on the fitted calibration quantiles; it quantifies "
            "test-path sampling, not calibration-set uncertainty, and is not "
            "simultaneous across scores or regimes.",
            "Per-regime paired differences report a normal-approximation SE; "
            "only the pooled paired difference gets a percentile bootstrap CI.",
            "Regime-level pooled coverage averages over the five regime units "
            "of each path; per-regime rows are descriptive and not a per-path "
            "coverage guarantee.",
            "Identification (Props 5/11) is a population statement; this step "
            "is finite-sample coverage, not an identification proof.",
        ],
        "policies": {},
        "summary": {},
    }

    # Cheap exactness checks on the truth / segment machinery.
    check_paths = [
        simulate_market([seed, wi])
        for wi in range(len(WINDOWS))
        for seed in list(cal_seeds)[:2]
    ]
    max_rollout_diff = 0.0
    max_seg_sum_diff = 0.0
    max_seg_rollout_diff = 0.0
    for m in check_paths:
        for policy in POLICIES.values():
            truth = truth_totals(m, policy)
            ref = pvc.rollout(
                m.true_price, m.pool_price, m.regime, m.volume, policy
            )
            max_rollout_diff = max(max_rollout_diff, abs(truth["total"] - ref))
            max_seg_sum_diff = max(
                max_seg_sum_diff,
                abs(sum(truth["by_segment"]) - truth["total"]),
            )
            pos_at = positions_at_segments(m, policy, truth["segments"])
            for seg_idx, (_r, start, end) in enumerate(truth["segments"]):
                pos = pos_at[seg_idx]
                il0 = pos.il(float(m.true_price[start])) if pos.in_pos else 0.0
                val = pvc.rollout_segment(
                    m.true_price[start : end + 2],
                    m.pool_price[start : end + 2],
                    m.regime[start : end + 1],
                    m.volume[start : end + 1],
                    policy,
                    copy.deepcopy(pos),
                    il0,
                    t_offset=start,
                )
                max_seg_rollout_diff = max(
                    max_seg_rollout_diff, abs(val - truth["by_segment"][seg_idx])
                )
    results["data_checks"]["truth_rollout"] = {
        "max_abs_diff_total_vs_policy_value_check_rollout": float(
            max_rollout_diff
        ),
        "max_abs_diff_sum_segments_vs_total": float(max_seg_sum_diff),
        "max_abs_diff_segment_rollout_vs_direct": float(max_seg_rollout_diff),
    }

    # Filter check against the pre-anchor Kalman used in steps 6-7.
    filt_mean = 0.0
    filt_cov = 0.0
    n_filt = 0
    for m in check_paths[:4]:
        policy = POLICIES["Wide_Rebalance_2pct"]
        sigma = pvc.estimate_sigma_open(m)
        post = filter_path(m, inputs["kappa"], inputs["mu"], sigma, h)
        for ep in pvc.extract_freeze_episodes(m):
            pre = ep["pre"]
            m1, c1 = pvc.kalman_pre_belief(
                m, pre, inputs["kappa"], inputs["mu"], sigma, h
            )
            m2, c2 = post[pre]
            filt_mean = max(filt_mean, float(np.max(np.abs(m1 - m2))))
            filt_cov = max(filt_cov, float(np.max(np.abs(c1 - c2))))
            n_filt += 1
    results["data_checks"]["filter_vs_kalman_pre_belief"] = {
        "n_checked": n_filt,
        "max_abs_mean_diff": filt_mean,
        "max_abs_cov_diff": filt_cov,
        "note": "filter_path starts from P_0 and filters the full path; "
        "kalman_pre_belief starts at the last OPEN before the episode. They "
        "should agree at the pre-anchor up to the early-window information.",
    }

    for name, policy in POLICIES.items():
        print(f"[trajectory] policy={name}: building calibration records", flush=True)
        cal_records = [
            build_path_record(
                simulate_market([seed, wi]),
                WINDOWS[wi],
                seed,
                policy,
                inputs,
                args.reps,
            )
            for wi in range(len(WINDOWS))
            for seed in cal_seeds
        ]
        print(f"[trajectory] policy={name}: building evaluation records", flush=True)
        ev_records = [
            build_path_record(
                simulate_market([seed, wi]),
                WINDOWS[wi],
                seed,
                policy,
                inputs,
                args.reps,
            )
            for wi in range(len(WINDOWS))
            for seed in eval_seeds
        ]

        truth_full = _truth_summary(
            np.array([r["truth_total"] for r in ev_records])
        )
        truth_regime = _regime_truth_summary(
            np.stack([r["truth_regime"] for r in ev_records])
        )
        policy_out: dict = {
            "truth": {"full": truth_full, "by_regime": truth_regime},
            "continuous": {},
            "segment_wise": {},
        }

        stacks: dict = {}
        for model_key, label in (("cont", "continuous"), ("seg", "segment_wise")):
            for method in ("ident", "naive"):
                cal = _stack(cal_records, model_key, method)
                ev = _stack(ev_records, model_key, method)
                stacks[(model_key, method)] = (cal, ev)
                block = _evaluate_model(cal, ev, ALPHA)
                policy_out[label][method] = block

        # Paired identified-vs-naive coverage comparisons on the same test
        # paths. Full-window uses the continuous rollout; regime-level uses
        # the segment-wise estimator (the primary G_r protocol).
        cal_id, ev_id = stacks[("cont", "ident")]
        cal_naive, ev_naive = stacks[("cont", "naive")]
        paired_full = _paired_full_window(cal_id, ev_id, cal_naive, ev_naive, ALPHA)
        cal_id, ev_id = stacks[("seg", "ident")]
        cal_naive, ev_naive = stacks[("seg", "naive")]
        paired_regime = _paired_regime(cal_id, ev_id, cal_naive, ev_naive, ALPHA)
        policy_out["paired"] = {
            "full_window": paired_full,
            "regime": paired_regime,
        }

        # Compact headline: combine the continuous total and the segment-level
        # regime protocol (the primary G_r estimator).
        combined: dict = {}
        for method in ALL_METHODS:
            prefix, score = _method_prefix(method)
            cont_block = policy_out["continuous"][prefix]
            seg_block = policy_out["segment_wise"][prefix]
            if score == "plugin_normal":
                combined[method] = {
                    "full_window_coverage": cont_block["full_window"][
                        "plugin_normal"
                    ]["coverage"],
                    "full_window_width": cont_block["full_window"][
                        "plugin_normal"
                    ]["mean_width"],
                    "regime_coverage": seg_block["regime_level"][
                        "plugin_normal"
                    ]["all"]["coverage"],
                    "regime_width": seg_block["regime_level"]["plugin_normal"][
                        "all"
                    ]["mean_width"],
                    "regime_per_group_coverage": {
                        label: seg_block["regime_level"]["plugin_normal"][
                            "per_group"
                        ][label]["coverage"]
                        for label in REGIME_ORDER
                    },
                    "bias_full": cont_block["bias"]["full"],
                    "bias_by_regime": seg_block["bias"]["by_regime"],
                }
            else:
                combined[method] = {
                    "full_window_within_coverage": cont_block["full_window"][
                        score
                    ]["within_window"]["coverage"],
                    "full_window_loro_coverage": cont_block["full_window"][
                        score
                    ]["leave_one_window_out"]["coverage"],
                    "full_window_loro_width": cont_block["full_window"][score][
                        "leave_one_window_out"
                    ]["mean_width"],
                    "regime_within_coverage": seg_block["regime_level"][score][
                        "within_regime"
                    ]["coverage"],
                    "regime_loro_coverage": seg_block["regime_level"][score][
                        "leave_one_regime_out"
                    ]["coverage"],
                    "regime_loro_width": seg_block["regime_level"][score][
                        "leave_one_regime_out"
                    ]["mean_width"],
                    "regime_loro_per_group_coverage": {
                        label: seg_block["regime_level"][score][
                            "leave_one_regime_out"
                        ]["per_group"][label]["coverage"]
                        for label in REGIME_ORDER
                    },
                    "bias_full": cont_block["bias"]["full"],
                    "bias_by_regime": seg_block["bias"]["by_regime"],
                }
        policy_out["headline"] = combined
        results["policies"][name] = policy_out
        results["summary"][name] = combined
        print(f"[trajectory] policy={name}: done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
