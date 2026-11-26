"""G0 识别检查第 5–7 步：目标策略 V^pi 的 plug-in 诊断、轨迹级覆盖与 D2′ 分解。

非 C1 循环（见 program.md §1 的 G0 例外）。只读冻结数据，只写
``results/policy_value_check.json``；真实 S 只作诊断/打分，不进入任何特征。

诊断对象
--------
两条**不属于 6 条 logging policy** 的目标策略：

* ``Wide_Rebalance_2pct``：空仓且在 OPEN/EXTENDED 时开宽区间；持仓且在
  OPEN/EXTENDED 时若 ``|P/anchor-1| > 2%`` 则重开宽区间；不主动在冻结前退出。
* ``Narrow_Rebalance_1pct``：同上，但用窄区间与 1% 阈值（压力策略）。

五个估值口径
------------
* ``true``：在 128 条冻结路径上，用真实 S 记基本面 IL 的精确 rollout；
* ``oracle``：真参数模型模拟（新息重抽），同一 volume/regime，检验模拟器；
* ``estimated``：用冻结数据估出的参数（σ 从 OPEN 收益估，κ/μ/跳频率取
  G0 第 2 步，h 从第 1/3 步估），σ 经验重采样；
* ``naive_mark``：同一批真实路径，但把池价 P 当作 S 记 IL（只改记账）；
* ``naive_model``：把 P 当 S 的完整错误模型（P 的 AR(1) + σ_all，无 h）。

``estimated_true_sigma`` 用真实 per-path σ 替换经验 σ，用于分解误差来源。

另做 ``block_study``：在 10 个独立的 128 路径离线块上重估 σ/κ_P，在固定的
512 条评估路径上比较 estimated/naive 的偏差，检验有限样本波动。

第 6 步 ``event_conformal``：以冻结事件前后（pre-anchor 到 post-anchor）的
目标策略收益贡献为轨迹单元，用估计/naive 模型做 plug-in，比较 raw、
studentized、CQR 三种 conformity score 在**同 duration（τ=2 vs τ=81）内**
校准的阳性对照，以及**跨 duration 迁移**（τ=2 校准→τ=81 测试，及反向）
的覆盖。κ_conf 用 KS 距离度量两个 τ 下 score 分布的偏移，给出
``max(0, 1-α-κ_conf)`` 的保守下界。``event_conformal_seed_robustness``
用三个 Monte Carlo 种子重复整段研究（主结果取第一个），报告覆盖的种子间波动。
``mean_width`` 一律为区间全宽：CQR 为 ``base+2Q``，raw/studentized 为
``2q`` 或 ``2q·ŝ``。

第 7 步 ``d2prime``（D2′ 值差分解诊断）：沿**观测到的**冻结事件历史，逐事件、
逐步计算三项敏感度——

* ``w1`` 估计项：识别模型对 ``log S_t`` 的后验与 naive 模型把池价当状态的
  点信念 ``δ_{log P_t}`` 之间的 1-D Wasserstein-1（= 后验下的 ``E|log S_t-log P_t|``）；
* ``tv`` 信息项：下一步观测 ``log P_{t+1}`` 的识别/naive 一步预测分布之间的 TV
  （高斯预测做数值积分；HALT 下识别模型预测为点质量，TV=1）；
* ``margin``/``flip`` 边界诊断：观测路径上目标策略距阈值的裕度，以及
  rebalance 决策在两种预测下的概率差 ``|p_id-p_naive|``（最小耦合翻转概率）。

识别侧用同一线性高斯状态 ``(log S, η, log S_last_open)`` 的 Kalman 滤波从最后
一个 OPEN 推进到 post-anchor；该滤波在 pre-anchor 处与 ``kalman_pre_belief`` 交叉校验。
这些是**沿观测历史的敏感度诊断**，不是对信念的均匀 sup；跳跃（0.3%）在滤波与
预测中省略。
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_env import (
    FEE_RATE,
    KAPPA,
    LIQUIDITY_SHARE,
    MU_LOG,
    NARROW,
    REGIME_EXTENDED,
    REGIME_HALT,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    V0,
    WIDE,
    WINDOW_DAYS,
    Position,
    calendar_regime,
    simulate_market,
)

ROOT = Path(__file__).resolve().parents[1]
TRUE_PATHS = ROOT / "results" / "true_paths.csv.gz"
OUT_JSON = ROOT / "results" / "policy_value_check.json"

N_STEPS = 336
WINDOWS = ("W1", "W2", "W3", "W4")
OFFLINE_SEEDS = range(42, 74)          # 冻结数据集的 128 条路径
EVAL_SEEDS = range(1000, 1512)         # 独立评估集：2048 条路径
BLOCK_SEEDS_START = 5000               # block study 的独立离线块
N_BLOCKS = 10
N_EVAL_BLOCK = 512                     # block study 的评估子集（128 seeds × 4）
R_MAIN = 12                            # 主表每个路径的模拟重复数
R_BLOCK = 3                            # block study 每个路径的模拟重复数
Z95 = 1.959963984540054

FREEZE = {REGIME_WEEKEND, REGIME_HOLIDAY, REGIME_HALT}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_model(
    rng: np.random.Generator,
    sigma: float,
    kappa: float,
    mu: float,
    jump_prob: float,
    open_sd: float,
    ext_phi: float,
    ext_sd: float,
    stale_phi: float,
    stale_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate (S, P, regime) with the same structure as synthetic_env."""
    s = np.empty(N_STEPS + 1)
    s[0] = math.exp(mu)
    for t in range(N_STEPS):
        log_s = math.log(s[t])
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < jump_prob else 0.0
        s[t + 1] = math.exp(log_s + kappa * (mu - log_s) + sigma * eps + jump)

    regime = np.array(
        [calendar_regime((t // 24) % WINDOW_DAYS, t % 24) for t in range(N_STEPS)],
        dtype=int,
    )
    p = np.empty(N_STEPS + 1)
    eta = 0.0
    last_open = s[0]
    for t in range(N_STEPS + 1):
        r = regime[t] if t < N_STEPS else regime[-1]
        if r == REGIME_OPEN:
            last_open = s[t]
            eta = 0.0
            p[t] = s[t] * math.exp(rng.normal(0.0, open_sd))
        elif r == REGIME_EXTENDED:
            eta = ext_phi * eta + rng.normal(0.0, ext_sd)
            p[t] = s[t] * math.exp(eta)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = stale_phi * eta + rng.normal(0.0, stale_sd)
            p[t] = last_open * math.exp(eta)
        else:
            p[t] = p[t - 1] if t > 0 else s[t]
    return s, p, regime


def simulate_naive(
    rng: np.random.Generator,
    sigma: float,
    kappa: float,
    mu: float,
    jump_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Naive model: observed P is treated as latent S (no observation mapping)."""
    s = np.empty(N_STEPS + 1)
    s[0] = math.exp(mu)
    for t in range(N_STEPS):
        log_s = math.log(s[t])
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < jump_prob else 0.0
        s[t + 1] = math.exp(log_s + kappa * (mu - log_s) + sigma * eps + jump)
    regime = np.array(
        [calendar_regime((t // 24) % WINDOW_DAYS, t % 24) for t in range(N_STEPS)],
        dtype=int,
    )
    return s, s.copy(), regime


def apply_action(pos: Position, action: int, price: float) -> tuple[float, bool]:
    """Apply an action at ``price``; return (cost, position_changed)."""
    cost = 0.0
    changed = False
    if action == 0:
        pass
    elif action in (1, 2):
        if pos.in_pos:
            cost += pos.close(price)
        cost += pos.open(price, NARROW if action == 1 else WIDE)
        changed = True
    elif action == 3:
        if pos.in_pos:
            cost += pos.close(price)
            changed = True
    elif action in (4, 5):
        if not pos.in_pos:
            cost += pos.open(price, NARROW if action == 4 else WIDE)
            changed = True
    else:
        raise ValueError(f"unknown action: {action}")
    return cost, changed


def rollout(
    s: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    volume: np.ndarray,
    policy,
) -> float:
    """Exact rollout; mirrors synthetic_env.simulate_strategy's reward."""
    pos = Position()
    prev_il_true = 0.0
    total = 0.0
    for t in range(len(regime)):
        price_from = float(p[t])
        price_to = float(p[t + 1])
        true_to = float(s[t + 1])
        r = int(regime[t])
        next_r = int(regime[t + 1]) if t + 1 < len(regime) else r
        action = policy(t, r, next_r, price_from, pos)
        cost, changed = apply_action(pos, action, price_from)

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

        total += (fee + il_true_step - cost) / V0
    return float(total)


def rollout_segment(
    s: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    volume: np.ndarray,
    policy,
    pos: Position,
    il_true0: float,
    t_offset: int = 0,
) -> float:
    """Reward of a policy on a contiguous episode with an inherited position.

    ``pos`` is the state *before* the action at the episode's first step;
    ``il_true0`` is that position's fundamental IL at the first price.
    The position is mutated, so pass a deep copy when reusing it.
    """
    prev_il_true = float(il_true0)
    total = 0.0
    for t in range(len(regime)):
        price_from = float(p[t])
        price_to = float(p[t + 1])
        true_to = float(s[t + 1])
        r = int(regime[t])
        next_r = int(regime[t + 1]) if t + 1 < len(regime) else r
        action = policy(t_offset + t, r, next_r, price_from, pos)
        cost, changed = apply_action(pos, action, price_from)

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

        total += (fee + il_true_step - cost) / V0
    return float(total)


# ---------------------------------------------------------------------------
# Target policies (not in STRATEGIES)
# ---------------------------------------------------------------------------


def policy_wide(t: int, r: int, next_r: int, p: float, pos: Position) -> int:
    """Wide range, 2% rebalance, no pre-freeze exit."""
    if not pos.in_pos and r in (REGIME_OPEN, REGIME_EXTENDED):
        return 5
    if (
        pos.in_pos
        and r in (REGIME_OPEN, REGIME_EXTENDED)
        and abs(p / pos.anchor - 1.0) > 0.02
    ):
        return 2
    return 0


def policy_narrow(t: int, r: int, next_r: int, p: float, pos: Position) -> int:
    """Narrow range, 1% rebalance, no pre-freeze exit (stress policy)."""
    if not pos.in_pos and r in (REGIME_OPEN, REGIME_EXTENDED):
        return 4
    if (
        pos.in_pos
        and r in (REGIME_OPEN, REGIME_EXTENDED)
        and abs(p / pos.anchor - 1.0) > 0.01
    ):
        return 1
    return 0


POLICIES = {
    "Wide_Rebalance_2pct": policy_wide,
    "Narrow_Rebalance_1pct": policy_narrow,
}


# ---------------------------------------------------------------------------
# Parameter estimation from observed (or diagnostic) data
# ---------------------------------------------------------------------------


def estimate_sigma_open(market) -> float:
    lp = np.log(market.pool_price)
    ret = (lp[1:] - lp[:-1])[market.regime == REGIME_OPEN]
    return float(ret.std(ddof=1))


def estimate_sigma_all(market) -> float:
    lp = np.log(market.pool_price)
    ret = lp[1:] - lp[:-1]
    return float(ret.std(ddof=1))


def estimate_naive_kappa(markets) -> tuple[float, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for m in markets:
        lp = np.log(m.pool_price)
        xs.append(lp[:-1])
        ys.append(lp[1:])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    return 1.0 - b, float(x.mean())


def estimate_mapping(diag: pd.DataFrame) -> dict:
    """Estimate sigma_o and the AR(1) observation-noise parameters.

    Uses true S only for this diagnostic (the mapping is not observable
    without it); the estimates are reported, not hidden.
    """
    diag = diag.sort_values(["window", "seed", "episode_step"])
    open_mask = diag["regime"] == REGIME_OPEN
    eps = np.log(diag["pool_price"] / diag["true_price"])[open_mask]
    open_sd = float(eps.std(ddof=1))

    ext_x: list[float] = []
    ext_y: list[float] = []
    stale_x: list[float] = []
    stale_y: list[float] = []
    for _, g in diag.groupby(["window", "seed"], sort=False):
        r = g["regime"].to_numpy()
        s = g["true_price"].to_numpy()
        p = g["pool_price"].to_numpy()
        eta = np.log(p / s)
        mask = (r[:-1] == REGIME_EXTENDED) & (r[1:] == REGIME_EXTENDED)
        ext_x.extend(eta[:-1][mask])
        ext_y.extend(eta[1:][mask])

        last = np.full(len(s), np.nan)
        cur = np.nan
        for i, ri in enumerate(r):
            if ri == REGIME_OPEN:
                cur = s[i]
            last[i] = cur
        eta_stale = np.log(p / last)
        mask = (
            ((r[:-1] == REGIME_WEEKEND) | (r[:-1] == REGIME_HOLIDAY))
            & ((r[1:] == REGIME_WEEKEND) | (r[1:] == REGIME_HOLIDAY))
        )
        x = eta_stale[:-1][mask]
        y = eta_stale[1:][mask]
        ok = np.isfinite(x) & np.isfinite(y)
        stale_x.extend(x[ok])
        stale_y.extend(y[ok])

    def ar1(xs: list[float], ys: list[float]) -> tuple[float, float]:
        x = np.asarray(xs)
        y = np.asarray(ys)
        b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        a = float(y.mean() - b * x.mean())
        resid_sd = float((y - a - b * x).std(ddof=1))
        return b, resid_sd

    ext_phi, ext_sd = ar1(ext_x, ext_y)
    stale_phi, stale_sd = ar1(stale_x, stale_y)
    return {
        "open_sd": open_sd,
        "ext_phi": ext_phi,
        "ext_sd": ext_sd,
        "stale_phi": stale_phi,
        "stale_sd": stale_sd,
        "n_open": int(open_mask.sum()),
        "n_ext_pairs": len(ext_x),
        "n_stale_pairs": len(stale_x),
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261020,
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = values[rng.integers(0, n, n)].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(values: np.ndarray, seed: int) -> dict:
    n = len(values)
    se = float(values.std(ddof=1) / math.sqrt(n))
    return {
        "mean": float(values.mean()),
        "se": se,
        "ci95": bootstrap_ci(values, seed=seed),
        "n_paths": int(n),
    }


def paired_diff(model: np.ndarray, truth: np.ndarray) -> dict:
    d = model - truth
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return {
        "mean": float(d.mean()),
        "se": se,
        "t": float(d.mean() / se) if se > 0 else math.nan,
        "ci95": bootstrap_ci(d, seed=20261021),
    }


# ---------------------------------------------------------------------------
# Step 6: freeze-episode (trajectory-level) conformal coverage
# ---------------------------------------------------------------------------

EVENT_TAUS = (2, 81)
R_EVENT = 96
N_FOLDS = 5
ALPHA = 0.1
# Asymptotic iid reference for the two-sample KS statistic under H0:
# E[sup|B|] = sqrt(pi/2) ln 2, and the 0.95 quantile of sup|B| is about 1.358.
KS_NULL_MEAN_CONST = math.sqrt(math.pi / 2.0) * math.log(2.0)
KS_NULL_Q95_CONST = 1.358
EPISODE_REGIMES = (REGIME_WEEKEND, REGIME_HOLIDAY, REGIME_HALT)
EVENT_SEEDS = (20261029, 20261030, 20261031)
METHODS = (
    "identified_raw",
    "identified_studentized",
    "identified_cqr",
    "naive_raw",
    "naive_studentized",
    "naive_cqr",
    "naive_mark_raw",
)
# Pre-decision labels available before the pre-anchor action; post_rebalance
# is deliberately excluded because it is a post-decision outcome.
MONDRIAN_KEYS = ("onset_action", "onset_distance", "onset_side")


def extract_freeze_episodes(market) -> list[dict]:
    """Freeze runs with both a pre-anchor and a post-anchor."""
    regime = market.regime
    out: list[dict] = []
    t = 0
    while t < len(regime):
        if regime[t] not in EPISODE_REGIMES:
            t += 1
            continue
        start = t
        while t < len(regime) and regime[t] in EPISODE_REGIMES:
            t += 1
        end = t - 1
        if start == 0 or t >= len(regime):
            continue
        out.append(
            {
                "start": int(start),
                "end": int(end),
                "tau": int(end - start + 1),
                "pre": int(start - 1),
                "post": int(t),
                "pre_regime": int(regime[start - 1]),
                "post_regime": int(regime[t]),
            }
        )
    return out


def last_open_before(regime: np.ndarray, pre: int) -> int:
    for t in range(pre, -1, -1):
        if regime[t] == REGIME_OPEN:
            return t
    raise ValueError("no OPEN step before the episode")


def kalman_pre_belief(
    market,
    pre: int,
    kappa: float,
    mu: float,
    sigma: float,
    h: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior over (log S_pre, eta_pre, log S_last_open) given P history.

    The state is linear-Gaussian on the informative segment since the last
    OPEN: log S follows the estimated AR(1), eta the estimated EXTENDED
    AR(1), and log P = log S + eta is observed without noise.  At the last
    OPEN the observation is log P = log S + N(0, open_sd^2) and eta = 0.
    """
    regime = market.regime
    logp = np.log(market.pool_price)
    last_open = last_open_before(regime, pre)
    var_o = h["open_sd"] ** 2
    mean = np.array([logp[last_open], 0.0, logp[last_open]])
    cov = np.zeros((3, 3))
    cov[0, 0] = var_o
    cov[2, 2] = var_o
    cov[0, 2] = cov[2, 0] = var_o
    if pre == last_open:
        return mean, cov

    transition = np.array(
        [[1.0 - kappa, 0.0, 0.0], [0.0, h["ext_phi"], 0.0], [0.0, 0.0, 1.0]]
    )
    intercept = np.array([kappa * mu, 0.0, 0.0])
    noise = np.diag([sigma**2, h["ext_sd"] ** 2, 0.0])
    obs = np.array([1.0, 1.0, 0.0])
    for t in range(last_open + 1, pre + 1):
        if regime[t] != REGIME_EXTENDED:
            raise ValueError(f"expected EXTENDED at {t}, got {regime[t]}")
        mean = transition @ mean + intercept
        cov = transition @ cov @ transition.T + noise
        z = float(logp[t])
        s = float(obs @ cov @ obs)
        gain = cov @ obs / s
        mean = mean + gain * (z - float(obs @ mean))
        ikh = np.eye(3) - np.outer(gain, obs)
        cov = ikh @ cov @ ikh.T
        cov = 0.5 * (cov + cov.T)
    return mean, cov


def sample_pre_state(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray):
    """Sample from a possibly rank-deficient Gaussian posterior."""
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    floor = max(float(np.max(vals)) * 1e-10, 1e-16)
    vals = np.maximum(vals, floor)
    return mean + vecs @ (np.sqrt(vals) * rng.normal(size=3))


def position_before(market, policy, pre: int) -> Position:
    """Observable policy position before the action at step ``pre``."""
    pos = Position()
    regime = market.regime
    for t in range(pre):
        r = int(regime[t])
        next_r = int(regime[t + 1]) if t + 1 < len(regime) else r
        action = policy(t, r, next_r, float(market.pool_price[t]), pos)
        apply_action(pos, action, float(market.pool_price[t]))
    return pos


def true_episode_value(market, episode: dict, policy) -> float:
    pre, post = episode["pre"], episode["post"]
    pos = position_before(market, policy, pre)
    il0 = pos.il(float(market.true_price[pre])) if pos.in_pos else 0.0
    return rollout_segment(
        market.true_price[pre : post + 1],
        market.pool_price[pre : post + 1],
        market.regime[pre:post],
        market.volume[pre:post],
        policy,
        pos,
        il0,
        t_offset=pre,
    )


def simulate_identified_episode(
    rng: np.random.Generator,
    market,
    episode: dict,
    policy,
    pos0: Position,
    kappa: float,
    mu: float,
    jump_prob: float,
    sigma: float,
    h: dict,
    belief_mean: np.ndarray,
    belief_cov: np.ndarray,
) -> float:
    """Model-based plug-in draw for one freeze episode, observed-only start."""
    pre, post = episode["pre"], episode["post"]
    ls, eta, ls_last = sample_pre_state(rng, belief_mean, belief_cov)
    s_list = [math.exp(ls)]
    p_list = [float(market.pool_price[pre])]
    for t in range(pre + 1, post + 1):
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < jump_prob else 0.0
        ls_new = ls + kappa * (mu - ls) + sigma * eps + jump
        r = int(market.regime[t])
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
        else:
            p_new = p_list[-1]
        s_list.append(math.exp(ls_new))
        p_list.append(p_new)
        ls = ls_new

    s = np.array(s_list)
    p = np.array(p_list)
    il0 = pos0.il(float(s[0])) if pos0.in_pos else 0.0
    return rollout_segment(
        s,
        p,
        market.regime[pre:post],
        market.volume[pre:post],
        policy,
        copy.deepcopy(pos0),
        il0,
        t_offset=pre,
    )


def simulate_naive_episode(
    rng: np.random.Generator,
    market,
    episode: dict,
    policy,
    pos0: Position,
    kappa_p: float,
    mu_p: float,
    jump_prob: float,
    sigma: float,
) -> float:
    """Naive plug-in draw: observed P is the state, no observation mapping."""
    pre, post = episode["pre"], episode["post"]
    p_list = [float(market.pool_price[pre])]
    for _ in range(pre + 1, post + 1):
        lp = math.log(p_list[-1])
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < jump_prob else 0.0
        p_list.append(math.exp(lp + kappa_p * (mu_p - lp) + sigma * eps + jump))
    p = np.array(p_list)
    il0 = pos0.il(float(p[0])) if pos0.in_pos else 0.0
    return rollout_segment(
        p,
        p,
        market.regime[pre:post],
        market.volume[pre:post],
        policy,
        copy.deepcopy(pos0),
        il0,
        t_offset=pre,
    )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    sa = np.sort(a)
    sb = np.sort(b)
    grid = np.unique(np.concatenate([sa, sb]))
    fa = np.searchsorted(sa, grid, side="right") / len(sa)
    fb = np.searchsorted(sb, grid, side="right") / len(sb)
    return float(np.max(np.abs(fa - fb)))


def bootstrap_ks(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Paired bootstrap uncertainty for the two-sample KS distance.

    The two score arrays are aligned by path (one tau=2 and one tau=81 event
    per path), so resampling the same index pair preserves their dependence.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = ks_distance(a[idx], b[idx])
    return {
        "se": float(vals.std(ddof=1)),
        "ci95": [
            float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975)),
        ],
    }


def bootstrap_branch_ks(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    branch: str,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Path-level paired bootstrap for the branch-internal KS distance.

    Calibration and test events are paired by path (same index) but their
    branch labels may differ across tau. Resampling paths jointly preserves
    the cross-tau dependence of the scores and the label composition.
    """
    rng = np.random.default_rng(seed)
    n = len(cal_scores)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = cal_scores[idx][cal_labels[idx] == branch]
        b = test_scores[idx][test_labels[idx] == branch]
        vals[i] = ks_distance(a, b) if len(a) and len(b) else math.nan
    return {
        "se": float(np.nanstd(vals, ddof=1)),
        "ci95": [
            float(np.nanquantile(vals, 0.025)),
            float(np.nanquantile(vals, 0.975)),
        ],
    }


def _coverage_detail(
    scores: np.ndarray,
    folds: np.ndarray,
    alpha: float,
    scale: np.ndarray | None = None,
    base_width: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Per-sample covered flags / widths, aligned with ``scores``."""
    covered = np.zeros(len(scores), dtype=bool)
    widths = np.zeros(len(scores), dtype=float)
    quantiles: list[float] = []
    for fold in range(N_FOLDS):
        cal = scores[folds != fold]
        test_idx = np.where(folds == fold)[0]
        q = conformal_quantile(cal, alpha)
        quantiles.append(q)
        covered[test_idx] = scores[test_idx] <= q
        if base_width is not None:
            # CQR: interval [q05 - Q, q95 + Q], width = base + 2Q.
            widths[test_idx] = base_width[test_idx] + 2.0 * q
        elif scale is None:
            widths[test_idx] = 2.0 * q
        else:
            widths[test_idx] = 2.0 * q * scale[test_idx]
    return covered, widths, quantiles


def _coverage_summary(
    covered: np.ndarray,
    widths: np.ndarray,
    quantiles: list[float],
) -> dict:
    cov = covered.astype(float)
    return {
        "n_test": int(len(cov)),
        "coverage": float(cov.mean()),
        "coverage_se": float(math.sqrt(cov.mean() * (1.0 - cov.mean()) / len(cov))),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "mean_q": float(np.mean(quantiles)),
        "min_q": float(np.min(quantiles)),
        "max_q": float(np.max(quantiles)),
    }


def cross_fitted_coverage(
    scores: np.ndarray,
    folds: np.ndarray,
    alpha: float,
    scale: np.ndarray | None = None,
    base_width: np.ndarray | None = None,
) -> dict:
    covered, widths, quantiles = _coverage_detail(
        scores, folds, alpha, scale, base_width
    )
    return _coverage_summary(covered, widths, quantiles)


def _branch_union_detail(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_folds: np.ndarray,
    test_folds: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float,
    test_scale: np.ndarray | None = None,
    test_base_width: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Branch-wise conformal with union/hull output.

    For each fold, compute a conformal quantile separately on the ``hold`` and
    ``rebalance`` calibration events, then use the larger quantile for every
    test event.  With symmetric scores the union of the two branch intervals
    is the interval with the larger radius; for CQR the hull is
    ``[q05 - Q, q95 + Q]`` with ``Q = max(q_hold, q_rebalance)``.  The branch
    label is used in calibration only; at test time the same union interval is
    issued for all events.
    """
    covered = np.zeros(len(test_scores), dtype=bool)
    widths = np.zeros(len(test_scores), dtype=float)
    quantiles: list[dict] = []
    for fold in range(N_FOLDS):
        cal = cal_scores[cal_folds != fold]
        cal_lab = cal_labels[cal_folds != fold]
        test_idx = np.where(test_folds == fold)[0]
        q_by_branch = {}
        for branch in ("hold", "rebalance"):
            cal_branch = cal[cal_lab == branch]
            q_by_branch[branch] = (
                conformal_quantile(cal_branch, alpha)
                if len(cal_branch)
                else math.inf
            )
        q_union = max(q_by_branch.values())
        quantiles.append({**q_by_branch, "union": q_union})
        covered[test_idx] = test_scores[test_idx] <= q_union
        if test_base_width is not None:
            widths[test_idx] = test_base_width[test_idx] + 2.0 * q_union
        elif test_scale is None:
            widths[test_idx] = 2.0 * q_union
        else:
            widths[test_idx] = 2.0 * q_union * test_scale[test_idx]
    return covered, widths, quantiles


def _branch_union_summary(
    covered: np.ndarray,
    widths: np.ndarray,
    quantiles: list[dict],
    labels: np.ndarray,
    marginal_width: float,
    cal_labels: np.ndarray,
) -> dict:
    out: dict = {
        "n_test": int(len(covered)),
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(widths)),
        "width_ratio_vs_marginal": float(np.mean(widths) / marginal_width),
        "mean_q_union": float(np.mean([q["union"] for q in quantiles])),
        "mean_q_hold": float(np.mean([q["hold"] for q in quantiles])),
        "mean_q_rebalance": float(
            np.mean([q["rebalance"] for q in quantiles])
        ),
        "branches": {},
    }
    for branch in ("hold", "rebalance"):
        idx = labels == branch
        n = int(idx.sum())
        if n == 0:
            continue
        cov = float(covered[idx].mean())
        cal_n = int((cal_labels == branch).sum())
        out["branches"][branch] = {
            "n": n,
            "cal_n": cal_n,
            "cal_fraction": float(cal_n / len(cal_labels)),
            "coverage": cov,
            "coverage_se": float(math.sqrt(cov * (1.0 - cov) / n)),
            "mean_width": float(widths[idx].mean()),
        }
    out["n_inf_union"] = int(
        sum(not math.isfinite(q["union"]) for q in quantiles)
    )
    out["any_inf"] = out["n_inf_union"] > 0
    return out


def _branch_kappa_block(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
    seed: int,
    overall_coverage: float,
) -> dict:
    """Branch-internal score-distribution shift and its plug-in bound.

    For each post-rebalance branch b, ``kappa_b`` is the KS distance between
    the calibration branch-b score distribution and the test branch-b score
    distribution.  If the calibration quantile ``q_b`` has calibration
    coverage at least ``1-alpha`` on branch b, then branch-b test coverage is
    at least ``1-alpha-kappa_b`` (the branch analogue of the marginal
    ``kappa_conf`` bound).  The block also reports the empirical trivial
    decomposition ``1-(1-c)/q_b`` from the evaluated method's observed
    overall coverage ``c``, marginal over branches (which reduces to the
    Prop-8 bound when ``c=1-alpha``; for the branch-union output this is its
    ``branch_union.coverage``, not the unmodified marginal pipeline).
    The KS statistic is strongly upward biased at the small branch sizes here,
    so the block also reports an iid null reference and the excess over it;
    neither bound is a finite-sample guarantee.
    """
    out: dict = {}
    cal_labels = cal_labels.astype(str)
    test_labels = test_labels.astype(str)
    for branch in ("hold", "rebalance"):
        ci = cal_labels == branch
        ti = test_labels == branch
        n_cal = int(ci.sum())
        n_test = int(ti.sum())
        if n_cal < GROUP_KAPPA_MIN_N or n_test < GROUP_KAPPA_MIN_N:
            continue
        k = ks_distance(cal_scores[ci], test_scores[ti])
        # sqrt((m+n)/(mn)) * KS converges to sup|B| under equal distributions.
        null_scale = math.sqrt((n_cal + n_test) / (n_cal * n_test))
        null_mean = KS_NULL_MEAN_CONST * null_scale
        null_q95 = KS_NULL_Q95_CONST * null_scale
        k_excess = max(0.0, k - null_mean)
        branch_fraction = n_test / len(test_scores)
        # Exact decomposition of the observed overall coverage: the branch-b
        # uncovered fraction is at most (1-c)/q_b.  With c=1-alpha this is the
        # Prop-8 bound; using the measured c keeps it an empirical identity.
        trivial_bound = max(
            0.0, 1.0 - (1.0 - overall_coverage) / branch_fraction
        )
        kappa_bound = max(0.0, 1.0 - alpha - k)
        boot = bootstrap_branch_ks(
            cal_scores,
            test_scores,
            cal_labels,
            test_labels,
            branch,
            seed=seed,
        )
        out[branch] = {
            "n_cal": n_cal,
            "n_test": n_test,
            "branch_fraction": branch_fraction,
            "overall_coverage": overall_coverage,
            "kappa_conf": k,
            "kappa_conf_se": boot["se"],
            "kappa_conf_ci95": boot["ci95"],
            "ks_null_mean_iid": null_mean,
            "ks_null_q95_iid": null_q95,
            "kappa_excess": k_excess,
            "kappa_bound_1malpha": kappa_bound,
            "trivial_bound_from_overall": trivial_bound,
        }
    return out


def _branch_union_joint_block(
    method_out: dict, direction: str, alpha: float
) -> dict:
    """Joint branch-union coverage/width summary against the marginal pipeline.

    The branch-union threshold is ``Q = max(q_hold, q_rebalance)`` per fold; the
    marginal threshold is the pooled conformal quantile ``q`` on the same
    calibration subset.  Because ``Q >= q`` (the max of the branch order
    statistics is at least the pooled order statistic), the union covers every
    test event the marginal interval covers; coverage dominance is therefore a
    finite-sample identity, not an empirical finding.  The price is width.

    Per branch this block reports the marginal and union coverage with binomial
    standard errors, the paired coverage gain, the branch width ratio, and a
    five-way taxonomy based on normal-approximation 95% intervals:

    - ``cost_only``: the marginal coverage is already clearly at or above
      nominal, so the union is pure width cost.
    - ``clear_gain``: the gain is clearly positive and the union is not clearly
      below nominal.
    - ``gain_but_fails``: the gain is clearly positive but the union is still
      clearly below nominal.
    - ``fails_no_clear_gain``: the union is clearly below nominal and the gain
      is not clear.
    - ``ambiguous``: none of the above intervals resolve the question.

    The intervals are descriptive, not simultaneous.  Cross-regime rows also
    carry the branch KS distance and the ``1-alpha-kappa`` bound.
    """
    bu = method_out[direction]["branch_union"]
    groups = method_out[direction]["conditional"]["by_group"]["post_rebalance"][
        "groups"
    ]
    kappa = method_out[direction].get("branch_kappa", {})
    nominal = 1.0 - alpha
    z = 1.96
    branches: dict = {}
    taxonomy_counts = {
        "cost_only": 0,
        "clear_gain": 0,
        "gain_but_fails": 0,
        "fails_no_clear_gain": 0,
        "ambiguous": 0,
    }
    for branch in ("hold", "rebalance"):
        if branch not in bu["branches"] or branch not in groups:
            continue
        n = int(bu["branches"][branch]["n"])
        if n == 0:
            continue
        marginal = float(groups[branch]["coverage"])
        union = float(bu["branches"][branch]["coverage"])
        gain = union - marginal
        se_marginal = math.sqrt(marginal * (1.0 - marginal) / n)
        se_union = math.sqrt(union * (1.0 - union) / n)
        se_gain = math.sqrt(max(gain * (1.0 - gain), 0.0) / n)
        cost_only = (marginal - z * se_marginal) >= nominal
        gain_clear = (gain - z * se_gain) > 0.0
        fail_clear = (union + z * se_union) < nominal
        if cost_only:
            taxonomy = "cost_only"
        elif fail_clear:
            taxonomy = (
                "gain_but_fails" if gain_clear else "fails_no_clear_gain"
            )
        elif gain_clear:
            taxonomy = "clear_gain"
        else:
            taxonomy = "ambiguous"
        taxonomy_counts[taxonomy] += 1
        marginal_width = float(groups[branch]["mean_width"])
        union_width = float(bu["branches"][branch]["mean_width"])
        entry = {
            "n_test": n,
            "cal_n": int(bu["branches"][branch]["cal_n"]),
            "cal_fraction": float(bu["branches"][branch]["cal_fraction"]),
            "marginal_coverage": marginal,
            "marginal_coverage_se": float(se_marginal),
            "union_coverage": union,
            "union_coverage_se": float(se_union),
            "coverage_gain": float(gain),
            "coverage_gain_se": float(se_gain),
            "gain_clear": bool(gain_clear),
            "marginal_clear_at_or_above_nominal": bool(cost_only),
            "union_clear_below_nominal": bool(fail_clear),
            "width_ratio_vs_marginal": float(union_width / marginal_width),
            "taxonomy": taxonomy,
        }
        if branch in kappa:
            k = float(kappa[branch]["kappa_conf"])
            kappa_bound = float(1.0 - alpha - k)
            entry["branch_kappa"] = k
            entry["kappa_bound_1malpha"] = kappa_bound
            entry["residual_gap"] = float(max(0.0, nominal - union))
            entry["kappa_bound_slack"] = float(union - kappa_bound)
            entry["residual_gap_over_kappa"] = (
                float(max(0.0, nominal - union) / k) if k > 0.0 else None
            )
        branches[branch] = entry

    worst_marginal = min(v["marginal_coverage"] for v in branches.values())
    worst_union = min(v["union_coverage"] for v in branches.values())
    return {
        "nominal": nominal,
        "n_branches": len(branches),
        "coverage_dominance": all(
            v["union_coverage"] >= v["marginal_coverage"] - 1e-12
            for v in branches.values()
        ),
        "overall_marginal_coverage": float(method_out[direction]["coverage"]),
        "overall_union_coverage": float(bu["coverage"]),
        "overall_coverage_gain": float(
            bu["coverage"] - method_out[direction]["coverage"]
        ),
        "width_ratio_vs_marginal": float(
            bu["width_ratio_vs_marginal"]
        ),
        "worst_branch_marginal_coverage": float(worst_marginal),
        "worst_branch_union_coverage": float(worst_union),
        "worst_gap_before": float(max(0.0, nominal - worst_marginal)),
        "worst_gap_after": float(max(0.0, nominal - worst_union)),
        "n_below_nominal_before": int(
            sum(v["marginal_coverage"] < nominal for v in branches.values())
        ),
        "n_below_nominal_after": int(
            sum(v["union_coverage"] < nominal for v in branches.values())
        ),
        "n_clear_below_before": int(
            sum(
                v["marginal_coverage"] + z * v["marginal_coverage_se"] < nominal
                for v in branches.values()
            )
        ),
        "n_clear_below_after": int(
            sum(
                v["union_coverage"] + z * v["union_coverage_se"] < nominal
                for v in branches.values()
            )
        ),
        "n_gain_clear": int(
            sum(v["gain_clear"] for v in branches.values())
        ),
        "n_cost_only": int(taxonomy_counts["cost_only"]),
        "taxonomy_counts": taxonomy_counts,
        "branches": branches,
    }


def _mondrian_detail(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_folds: np.ndarray,
    test_folds: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
    test_scale: np.ndarray | None = None,
    test_base_width: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
    """Mondrian conformal on a pre-decision label.

    Per fold, compute a separate conformal quantile for each calibration group
    and use the test point's group quantile.  If a test group has no finite
    calibration quantile (missing group, or n_cal below the finite-sample
    threshold ``ceil((n_cal+1)(1-alpha))``), fall back to the marginal quantile
    for that fold and record the fallback.  Only pre-decision labels may be
    passed here; the post-rebalance label is used by the summary for evaluation
    only, never for interval construction.
    """
    covered = np.zeros(len(test_scores), dtype=bool)
    widths = np.zeros(len(test_scores), dtype=float)
    q_used = np.full(len(test_scores), np.nan)
    fallback = np.zeros(len(test_scores), dtype=bool)
    quantiles: list[dict] = []
    test_labels = test_labels.astype(str)
    cal_labels = cal_labels.astype(str)
    for fold in range(N_FOLDS):
        cal = cal_scores[cal_folds != fold]
        cal_lab = cal_labels[cal_folds != fold]
        test_idx = np.where(test_folds == fold)[0]
        q_marg = conformal_quantile(cal, alpha)
        q_by_group: dict = {}
        for group in sorted(set(test_labels[test_idx].tolist())):
            cal_group = cal[cal_lab == group]
            q_group = (
                conformal_quantile(cal_group, alpha) if len(cal_group) else math.inf
            )
            used_fallback = not math.isfinite(q_group)
            q_use = q_marg if used_fallback else q_group
            q_by_group[group] = {
                "q": q_use,
                "q_group": q_group if math.isfinite(q_group) else None,
                "n_cal": int(len(cal_group)),
                "fallback": bool(used_fallback),
            }
            idx = test_idx[test_labels[test_idx] == group]
            covered[idx] = test_scores[idx] <= q_use
            q_used[idx] = q_use
            fallback[idx] = used_fallback
            if test_base_width is not None:
                widths[idx] = test_base_width[idx] + 2.0 * q_use
            elif test_scale is None:
                widths[idx] = 2.0 * q_use
            else:
                widths[idx] = 2.0 * q_use * test_scale[idx]
        quantiles.append({"marginal": q_marg, "groups": q_by_group})
    return covered, widths, quantiles, q_used, fallback


def _mondrian_summary(
    covered: np.ndarray,
    widths: np.ndarray,
    quantiles: list[dict],
    q_used: np.ndarray,
    fallback: np.ndarray,
    pre_labels: np.ndarray,
    post_labels: np.ndarray,
    marginal_width: float,
    alpha: float = ALPHA,
) -> dict:
    """Coverage/width summary for a pre-decision Mondrian interval."""
    cov = float(covered.mean())
    pre_groups = _group_coverage(covered, widths, pre_labels)
    post_groups = _group_coverage(covered, widths, post_labels)
    pre_covs = [v["coverage"] for v in pre_groups.values() if v["n"] >= COND_MIN_N]
    post_covs = [
        v["coverage"] for v in post_groups.values() if v["n"] >= COND_MIN_N
    ]
    pre_labels_str = pre_labels.astype(str)
    fallback_by_group = {}
    for label in sorted(set(pre_labels_str.tolist())):
        idx = pre_labels_str == label
        fallback_by_group[label] = {
            "n": int(idx.sum()),
            "n_fallback": int(fallback[idx].sum()),
        }
    return {
        "n_test": int(len(covered)),
        "coverage": cov,
        "coverage_se": float(math.sqrt(cov * (1.0 - cov) / len(covered))),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "width_ratio_vs_marginal": float(np.mean(widths) / marginal_width),
        "mean_q_marginal": float(np.mean([q["marginal"] for q in quantiles])),
        "mean_q_used": float(np.nanmean(q_used)),
        "n_fallback": int(fallback.sum()),
        "fallback_fraction": float(fallback.mean()),
        "fallback_by_group": fallback_by_group,
        "by_pre_group": pre_groups,
        "pre_worst_gap_n20": (
            max(0.0, (1.0 - alpha) - min(pre_covs)) if pre_covs else 0.0
        ),
        "by_post_rebalance": post_groups,
        "post_worst_gap_n20": (
            max(0.0, (1.0 - alpha) - min(post_covs)) if post_covs else 0.0
        ),
    }


def _mondrian_block(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_folds: np.ndarray,
    test_folds: np.ndarray,
    cal_groups: dict,
    test_groups: dict,
    post_labels: np.ndarray,
    marginal_width: float,
    test_scale: np.ndarray | None = None,
    test_base_width: np.ndarray | None = None,
) -> dict:
    """Run pre-decision Mondrian for every pre-decision grouping key."""
    out: dict = {}
    for key in MONDRIAN_KEYS:
        if key not in cal_groups or key not in test_groups:
            continue
        covered, widths, quantiles, q_used, fallback = _mondrian_detail(
            cal_scores,
            test_scores,
            cal_folds,
            test_folds,
            cal_groups[key],
            test_groups[key],
            ALPHA,
            test_scale,
            test_base_width,
        )
        out[key] = _mondrian_summary(
            covered,
            widths,
            quantiles,
            q_used,
            fallback,
            test_groups[key],
            post_labels,
            marginal_width,
        )
    return out


def episode_conformal_study(
    offline: list,
    policies: dict,
    kappa: float,
    mu: float,
    jump_prob: float,
    h: dict,
    seed: int = 20261029,
) -> dict:
    rng = np.random.default_rng(seed)
    results: dict = {
        "config": {
            "event_taus": list(EVENT_TAUS),
            "r_event": R_EVENT,
            "n_folds": N_FOLDS,
            "alpha": ALPHA,
            "seed": seed,
            "n_paths": len(offline),
            "episode_definition": "pre-anchor to post-anchor; target policy starts "
            "from the observable position inherited from the path before the episode",
            "state_groups": {
                "in_position": "observable position before the pre-anchor action; "
                "degenerate for both target policies (always in position), so the "
                "flat group is empty",
                "onset_action": "policy action at the pre-anchor decision: open "
                "(flat), rebalance (in position and |P_pre/anchor-1|>theta), or "
                "hold; computed from observed P and the inherited anchor",
                "onset_distance": "near if |P_pre/anchor-1|/theta>0.5, else far; "
                "flat events (not in position) are far by construction "
                "(currently none)",
                "onset_side": "P_pre above vs below the inherited anchor",
                "post_rebalance": "observed policy action at the post-anchor "
                "decision (rebalance vs hold); a post-hoc outcome, not available "
                "at prediction time, reported only as a boundary diagnostic",
            },
            "conditional_metrics": {
                "cond_min_n": COND_MIN_N,
                "group_kappa_min_n": GROUP_KAPPA_MIN_N,
                "worst_gap_n20": "max(0, (1-alpha) - min group coverage) over "
                "groups with n>=cond_min_n; a shortfall from nominal, not a "
                "between-group range",
                "group_kappa_conf": "per-group KS distance between calibration and "
                "test scores; descriptive plug-in, no bootstrap CI",
            },
            "branch_union": "branch-wise conformal on post_rebalance labels: "
            "per fold, compute separate conformal quantiles on hold and "
            "rebalance calibration events and issue the union/hull interval "
            "with the larger quantile; uses the post-decision branch label in "
            "calibration only, and is reported as the minimal width cost of "
            "branch-conditional coverage (not a test-time conditional method)",
            "branch_kappa": "per-branch KS distance between calibration and "
            "test score distributions, pooled within each post_rebalance "
            "branch (cross-tau directions only); kappa_excess subtracts the "
            "asymptotic iid KS mean sqrt(pi/2)ln2*sqrt((n_cal+n_test)/"
            "(n_cal*n_test)), and the paired path bootstrap resamples both "
            "the scores and the branch labels. Per branch it reports the KS "
            "bound 1-alpha-kappa_b and the empirical trivial decomposition "
            "1-(1-c)/q_b from the evaluated method's observed overall coverage "
            "c (marginal over branches; for the branch-union output this is "
            "its branch_union.coverage, not the unmodified marginal pipeline; "
            "equal to the Prop-8 bound when c=1-alpha). Branch labels are "
            "deterministic given the frozen paths, so branch counts and null "
            "references are seed-invariant; the three-seed summary stores "
            "kappa_conf, SE, bound and trivial decomposition only.",
            "branch_union_joint": "joint coverage/width summary of the "
            "branch-union against the unmodified marginal pipeline on the same "
            "test events: per post_rebalance branch, marginal and union "
            "coverage with binomial SE, the paired coverage gain (union minus "
            "marginal; union coverage is a superset because "
            "max(q_hold,q_rebalance) >= q_pooled), the branch width ratio, and "
            "a five-way taxonomy (cost_only: marginal lower 95% CI >= "
            "1-alpha; clear_gain: gain lower 95% CI > 0 and union not clearly "
            "below nominal; gain_but_fails: gain clear but union upper 95% CI "
            "< 1-alpha; fails_no_clear_gain: union clearly below nominal but "
            "gain not clear; ambiguous: neither). Cross-regime rows also carry "
            "the branch kappa, its 1-alpha-kappa bound, the bound slack and "
            "residual-gap/kappa ratio. Intervals are normal-approximation, "
            "descriptive and not simultaneous. The kappa_bound_1malpha stored "
            "here is unclipped, so it can be negative when kappa_b > 1-alpha "
            "(4 cross-regime branch cells). worst_gap_before/after use all "
            "branches and do not apply the n>=20 threshold of "
            "conditional.worst_gap_n20. The three-seed summary omits "
            "cal_n/cal_fraction because branch labels are deterministic given "
            "the frozen paths and seed-invariant.",
            "pre_decision_mondrian": "Mondrian conformal on pre-decision "
            "labels (onset_action / onset_distance / onset_side): per fold, a "
            "separate conformal quantile per calibration group, using the test "
            "point's group quantile. A group with no finite calibration "
            "quantile (n_cal < ceil((n_cal+1)(1-alpha))) falls back to that "
            "fold's marginal quantile; fallback counts are reported. "
            "post_rebalance labels are used only in the summary evaluation, "
            "never in interval construction.",
        },
        "policies": {},
    }

    for name, policy in policies.items():
        theta = THRESHOLDS[name]
        rows = {tau: [] for tau in EVENT_TAUS}
        for m in offline:
            sigma_path = estimate_sigma_open(m)
            sigma_all_path = estimate_sigma_all(m)
            kappa_p_path, mu_p_path = estimate_naive_kappa([m])
            for episode in extract_freeze_episodes(m):
                tau = episode["tau"]
                if tau not in EVENT_TAUS:
                    continue
                pre = episode["pre"]
                pos0 = position_before(m, policy, pre)
                p_pre = float(m.pool_price[pre])
                if pos0.in_pos:
                    dev = p_pre / pos0.anchor - 1.0
                    dev_ratio = abs(dev) / theta
                else:
                    dev = 0.0
                    dev_ratio = 0.0
                if not pos0.in_pos:
                    onset_action = "open"
                elif abs(dev) > theta:
                    onset_action = "rebalance"
                else:
                    onset_action = "hold"
                onset_near = bool(pos0.in_pos and dev_ratio > 0.5)
                onset_side = (
                    "above" if dev > 0.0 else ("below" if dev < 0.0 else "flat")
                )
                pos_post = position_before(m, policy, episode["post"])
                post_rebalance = bool(
                    pos_post.in_pos
                    and int(m.regime[episode["post"]])
                    in (REGIME_OPEN, REGIME_EXTENDED)
                    and abs(
                        float(m.pool_price[episode["post"]]) / pos_post.anchor - 1.0
                    )
                    > theta
                )
                truth = true_episode_value(m, episode, policy)
                belief_mean, belief_cov = kalman_pre_belief(
                    m, pre, kappa, mu, sigma_path, h
                )
                ident = np.array(
                    [
                        simulate_identified_episode(
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
                        for _ in range(R_EVENT)
                    ]
                )
                naive = np.array(
                    [
                        simulate_naive_episode(
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
                        for _ in range(R_EVENT)
                    ]
                )
                mark = rollout_segment(
                    m.pool_price[pre : episode["post"] + 1],
                    m.pool_price[pre : episode["post"] + 1],
                    m.regime[pre : episode["post"]],
                    m.volume[pre : episode["post"]],
                    policy,
                    copy.deepcopy(pos0),
                    pos0.il(float(m.pool_price[pre])) if pos0.in_pos else 0.0,
                    t_offset=pre,
                )
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

        policy_out: dict = {
            "episodes": {},
            "methods": {},
        }
        folds_by_tau = {}
        groups_by_tau = {}
        for tau in EVENT_TAUS:
            arr = rows[tau]
            n = len(arr)
            # One episode of each tau per path, both lists in the `offline`
            # order (window-major, seed-minor).  Group the four windows of the
            # same seed into one fold: position % len(OFFLINE_SEEDS) is the
            # seed index.
            folds = (np.arange(n) % len(OFFLINE_SEEDS)) % N_FOLDS
            folds_by_tau[tau] = folds
            groups_by_tau[tau] = {
                "in_position": np.array(
                    ["in_position" if r["in_position"] else "flat" for r in arr]
                ),
                "onset_action": np.array([r["onset_action"] for r in arr]),
                "onset_distance": np.array(
                    ["near" if r["onset_near"] else "far" for r in arr]
                ),
                "onset_side": np.array([r["onset_side"] for r in arr]),
                "post_rebalance": np.array(
                    ["rebalance" if r["post_rebalance"] else "hold" for r in arr]
                ),
            }
            group_counts = {
                name: {
                    label: int((labels == label).sum())
                    for label in sorted(set(labels.tolist()))
                }
                for name, labels in groups_by_tau[tau].items()
            }
            joint_counts = {
                f"{g}_x_post_rebalance": _joint_counts(
                    groups_by_tau[tau][g], groups_by_tau[tau]["post_rebalance"]
                )
                for g in ("onset_action", "onset_side")
            }
            truth = np.array([r["truth"] for r in arr])
            policy_out["episodes"][str(tau)] = {
                "n": n,
                "mean_true": float(truth.mean()),
                "se_true": float(truth.std(ddof=1) / math.sqrt(n)),
                "mean_ident": float(np.mean([r["ident_mean"] for r in arr])),
                "mean_ident_sd": float(np.mean([r["ident_sd"] for r in arr])),
                "mean_naive": float(np.mean([r["naive_mean"] for r in arr])),
                "mean_naive_sd": float(np.mean([r["naive_sd"] for r in arr])),
                "mean_mark": float(np.mean([r["mark"] for r in arr])),
                "ident_bias": float(
                    np.mean([r["ident_mean"] - r["truth"] for r in arr])
                ),
                "naive_bias": float(
                    np.mean([r["naive_mean"] - r["truth"] for r in arr])
                ),
                "mark_bias": float(np.mean([r["mark"] - r["truth"] for r in arr])),
                "groups": group_counts,
                "joint_counts": joint_counts,
            }

        for tau in EVENT_TAUS:
            arr = rows[tau]
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
            scores = {
                "identified_raw": np.abs(truth - ident_mean),
                "identified_studentized": np.abs(truth - ident_mean)
                / np.maximum(ident_sd, 1e-12),
                "identified_cqr": np.maximum(ident_q05 - truth, truth - ident_q95),
                "naive_raw": np.abs(truth - naive_mean),
                "naive_studentized": np.abs(truth - naive_mean)
                / np.maximum(naive_sd, 1e-12),
                "naive_cqr": np.maximum(naive_q05 - truth, truth - naive_q95),
                "naive_mark_raw": np.abs(truth - mark),
            }
            policy_out.setdefault("_scores", {})[tau] = {
                "scores": scores,
                "scale": {
                    "identified_studentized": ident_sd,
                    "naive_studentized": naive_sd,
                },
                "base_width": {
                    "identified_cqr": ident_q95 - ident_q05,
                    "naive_cqr": naive_q95 - naive_q05,
                },
            }

        for method in METHODS:
            method_out: dict = {}
            s2 = policy_out["_scores"][2]["scores"][method]
            s81 = policy_out["_scores"][81]["scores"][method]
            scale2 = policy_out["_scores"][2]["scale"].get(method)
            scale81 = policy_out["_scores"][81]["scale"].get(method)
            base2 = policy_out["_scores"][2]["base_width"].get(method)
            base81 = policy_out["_scores"][81]["base_width"].get(method)
            f2 = folds_by_tau[2]
            f81 = folds_by_tau[81]

            # Positive controls: calibration and test from the same tau.
            c2, w2, q2 = _coverage_detail(s2, f2, ALPHA, scale2, base2)
            c81, w81, q81 = _coverage_detail(s81, f81, ALPHA, scale81, base81)
            # Held-out duration class: calibrate on one tau, test on the other.
            c278, w278, q278 = _cross_tau_detail(
                s2, s81, f2, f81, ALPHA, scale81, base81
            )
            c812, w812, q812 = _cross_tau_detail(
                s81, s2, f81, f2, ALPHA, scale2, base2
            )
            method_out["within_tau2"] = _coverage_summary(c2, w2, q2)
            method_out["within_tau2"]["conditional"] = _conditional_block(
                c2, w2, groups_by_tau[2]
            )
            bu_c2, bu_w2, bu_q2 = _branch_union_detail(
                s2,
                s2,
                f2,
                f2,
                groups_by_tau[2]["post_rebalance"],
                ALPHA,
                scale2,
                base2,
            )
            method_out["within_tau2"]["branch_union"] = _branch_union_summary(
                bu_c2,
                bu_w2,
                bu_q2,
                groups_by_tau[2]["post_rebalance"],
                method_out["within_tau2"]["mean_width"],
                groups_by_tau[2]["post_rebalance"],
            )
            method_out["within_tau2"]["mondrian"] = _mondrian_block(
                s2,
                s2,
                f2,
                f2,
                groups_by_tau[2],
                groups_by_tau[2],
                groups_by_tau[2]["post_rebalance"],
                method_out["within_tau2"]["mean_width"],
                scale2,
                base2,
            )
            method_out["within_tau81"] = _coverage_summary(c81, w81, q81)
            method_out["within_tau81"]["conditional"] = _conditional_block(
                c81, w81, groups_by_tau[81]
            )
            bu_c81, bu_w81, bu_q81 = _branch_union_detail(
                s81,
                s81,
                f81,
                f81,
                groups_by_tau[81]["post_rebalance"],
                ALPHA,
                scale81,
                base81,
            )
            method_out["within_tau81"]["branch_union"] = _branch_union_summary(
                bu_c81,
                bu_w81,
                bu_q81,
                groups_by_tau[81]["post_rebalance"],
                method_out["within_tau81"]["mean_width"],
                groups_by_tau[81]["post_rebalance"],
            )
            method_out["within_tau81"]["mondrian"] = _mondrian_block(
                s81,
                s81,
                f81,
                f81,
                groups_by_tau[81],
                groups_by_tau[81],
                groups_by_tau[81]["post_rebalance"],
                method_out["within_tau81"]["mean_width"],
                scale81,
                base81,
            )
            method_out["cal_tau2_test_tau81"] = _coverage_summary(
                c278, w278, q278
            )
            method_out["cal_tau2_test_tau81"]["conditional"] = _conditional_block(
                c278,
                w278,
                groups_by_tau[81],
                s2,
                s81,
                groups_by_tau[2],
            )
            bu_c278, bu_w278, bu_q278 = _branch_union_detail(
                s2,
                s81,
                f2,
                f81,
                groups_by_tau[2]["post_rebalance"],
                ALPHA,
                scale81,
                base81,
            )
            method_out["cal_tau2_test_tau81"]["branch_union"] = (
                _branch_union_summary(
                    bu_c278,
                    bu_w278,
                    bu_q278,
                    groups_by_tau[81]["post_rebalance"],
                    method_out["cal_tau2_test_tau81"]["mean_width"],
                    groups_by_tau[2]["post_rebalance"],
                )
            )
            method_out["cal_tau2_test_tau81"]["branch_kappa"] = (
                _branch_kappa_block(
                    s2,
                    s81,
                    groups_by_tau[2]["post_rebalance"],
                    groups_by_tau[81]["post_rebalance"],
                    ALPHA,
                    seed + 2,
                    method_out["cal_tau2_test_tau81"]["branch_union"][
                        "coverage"
                    ],
                )
            )
            method_out["cal_tau2_test_tau81"]["mondrian"] = _mondrian_block(
                s2,
                s81,
                f2,
                f81,
                groups_by_tau[2],
                groups_by_tau[81],
                groups_by_tau[81]["post_rebalance"],
                method_out["cal_tau2_test_tau81"]["mean_width"],
                scale81,
                base81,
            )
            method_out["cal_tau81_test_tau2"] = _coverage_summary(
                c812, w812, q812
            )
            method_out["cal_tau81_test_tau2"]["conditional"] = _conditional_block(
                c812,
                w812,
                groups_by_tau[2],
                s81,
                s2,
                groups_by_tau[81],
            )
            bu_c812, bu_w812, bu_q812 = _branch_union_detail(
                s81,
                s2,
                f81,
                f2,
                groups_by_tau[81]["post_rebalance"],
                ALPHA,
                scale2,
                base2,
            )
            method_out["cal_tau81_test_tau2"]["branch_union"] = (
                _branch_union_summary(
                    bu_c812,
                    bu_w812,
                    bu_q812,
                    groups_by_tau[2]["post_rebalance"],
                    method_out["cal_tau81_test_tau2"]["mean_width"],
                    groups_by_tau[81]["post_rebalance"],
                )
            )
            method_out["cal_tau81_test_tau2"]["branch_kappa"] = (
                _branch_kappa_block(
                    s81,
                    s2,
                    groups_by_tau[81]["post_rebalance"],
                    groups_by_tau[2]["post_rebalance"],
                    ALPHA,
                    seed + 3,
                    method_out["cal_tau81_test_tau2"]["branch_union"][
                        "coverage"
                    ],
                )
            )
            method_out["cal_tau81_test_tau2"]["mondrian"] = _mondrian_block(
                s81,
                s2,
                f81,
                f2,
                groups_by_tau[81],
                groups_by_tau[2],
                groups_by_tau[2]["post_rebalance"],
                method_out["cal_tau81_test_tau2"]["mean_width"],
                scale2,
                base2,
            )
            for direction in (
                "within_tau2",
                "within_tau81",
                "cal_tau2_test_tau81",
                "cal_tau81_test_tau2",
            ):
                method_out[direction]["branch_union_joint"] = (
                    _branch_union_joint_block(method_out, direction, ALPHA)
                )
            kappa_conf = ks_distance(s2, s81)
            boot = bootstrap_ks(s2, s81, seed=seed + 1)
            method_out["kappa_conf_tau2_vs_tau81"] = kappa_conf
            method_out["kappa_conf_se"] = boot["se"]
            method_out["kappa_conf_ci95"] = boot["ci95"]
            method_out["bound_1malpha_mkconf"] = max(0.0, 1.0 - ALPHA - kappa_conf)
            policy_out["methods"][method] = method_out

        del policy_out["_scores"]
        results["policies"][name] = policy_out

    return results


def summarize_event_coverage(ev: dict) -> dict:
    """Compact coverage table for the seed-robustness block."""
    out: dict = {}
    for policy, policy_out in ev["policies"].items():
        out[policy] = {}
        for method in METHODS:
            m = policy_out["methods"][method]
            cond81 = m["cal_tau2_test_tau81"].get("conditional", {})
            cond2 = m["cal_tau81_test_tau2"].get("conditional", {})
            within2 = m["within_tau2"].get("conditional", {})
            within81 = m["within_tau81"].get("conditional", {})
            out[policy][method] = {
                "kappa_conf": m["kappa_conf_tau2_vs_tau81"],
                "kappa_conf_se": m["kappa_conf_se"],
                "within_tau2": m["within_tau2"]["coverage"],
                "within_tau81": m["within_tau81"]["coverage"],
                "cal_tau2_test_tau81": m["cal_tau2_test_tau81"]["coverage"],
                "cal_tau81_test_tau2": m["cal_tau81_test_tau2"]["coverage"],
                "within_onset_gap_tau2": within2.get("by_group", {})
                .get("onset_action", {})
                .get("worst_gap_n20"),
                "within_onset_gap_tau81": within81.get("by_group", {})
                .get("onset_action", {})
                .get("worst_gap_n20"),
                "within_post_gap_tau2": within2.get("by_group", {})
                .get("post_rebalance", {})
                .get("worst_gap_n20"),
                "within_post_gap_tau81": within81.get("by_group", {})
                .get("post_rebalance", {})
                .get("worst_gap_n20"),
                "within_distance_gap_tau2": within2.get("by_group", {})
                .get("onset_distance", {})
                .get("worst_gap_n20"),
                "within_distance_gap_tau81": within81.get("by_group", {})
                .get("onset_distance", {})
                .get("worst_gap_n20"),
                "within_side_gap_tau2": within2.get("by_group", {})
                .get("onset_side", {})
                .get("worst_gap_n20"),
                "within_side_gap_tau81": within81.get("by_group", {})
                .get("onset_side", {})
                .get("worst_gap_n20"),
                "cond_onset_gap_tau81": cond81.get("by_group", {})
                .get("onset_action", {})
                .get("worst_gap_n20"),
                "cond_onset_gap_tau2": cond2.get("by_group", {})
                .get("onset_action", {})
                .get("worst_gap_n20"),
                "cond_post_gap_tau81": cond81.get("by_group", {})
                .get("post_rebalance", {})
                .get("worst_gap_n20"),
                "cond_post_gap_tau2": cond2.get("by_group", {})
                .get("post_rebalance", {})
                .get("worst_gap_n20"),
                "cond_distance_gap_tau81": cond81.get("by_group", {})
                .get("onset_distance", {})
                .get("worst_gap_n20"),
                "cond_distance_gap_tau2": cond2.get("by_group", {})
                .get("onset_distance", {})
                .get("worst_gap_n20"),
                "cond_side_gap_tau81": cond81.get("by_group", {})
                .get("onset_side", {})
                .get("worst_gap_n20"),
                "cond_side_gap_tau2": cond2.get("by_group", {})
                .get("onset_side", {})
                .get("worst_gap_n20"),
                "mondrian": {
                    direction: {
                        key: {
                            "coverage": val["coverage"],
                            "pre_worst_gap_n20": val["pre_worst_gap_n20"],
                            "post_worst_gap_n20": val["post_worst_gap_n20"],
                            "post_rebalance_coverage": val["by_post_rebalance"]
                            .get("rebalance", {})
                            .get("coverage"),
                            "fallback_fraction": val["fallback_fraction"],
                            "width_ratio_vs_marginal": val[
                                "width_ratio_vs_marginal"
                            ],
                        }
                        for key, val in m[direction].get("mondrian", {}).items()
                    }
                    for direction in (
                        "within_tau2",
                        "within_tau81",
                        "cal_tau2_test_tau81",
                        "cal_tau81_test_tau2",
                    )
                },
                "branch_kappa": {
                    direction: {
                        branch: {
                            "kappa_conf": val["kappa_conf"],
                            "kappa_conf_se": val["kappa_conf_se"],
                            "kappa_bound_1malpha": val[
                                "kappa_bound_1malpha"
                            ],
                            "trivial_bound_from_overall": val[
                                "trivial_bound_from_overall"
                            ],
                        }
                        for branch, val in m[direction]
                        .get("branch_kappa", {})
                        .items()
                    }
                    for direction in (
                        "cal_tau2_test_tau81",
                        "cal_tau81_test_tau2",
                    )
                },
                "branch_union_joint": {
                    direction: {
                        "coverage": m[direction]["branch_union_joint"][
                            "overall_union_coverage"
                        ],
                        "coverage_gain": m[direction][
                            "branch_union_joint"
                        ]["overall_coverage_gain"],
                        "width_ratio_vs_marginal": m[direction][
                            "branch_union_joint"
                        ]["width_ratio_vs_marginal"],
                        "worst_gap_before": m[direction][
                            "branch_union_joint"
                        ]["worst_gap_before"],
                        "worst_gap_after": m[direction][
                            "branch_union_joint"
                        ]["worst_gap_after"],
                        "taxonomy_counts": m[direction][
                            "branch_union_joint"
                        ]["taxonomy_counts"],
                        "branches": {
                            branch: {
                                "n_test": val["n_test"],
                                "marginal_coverage": val[
                                    "marginal_coverage"
                                ],
                                "union_coverage": val["union_coverage"],
                                "coverage_gain": val["coverage_gain"],
                                "width_ratio_vs_marginal": val[
                                    "width_ratio_vs_marginal"
                                ],
                                "taxonomy": val["taxonomy"],
                            }
                            for branch, val in m[direction][
                                "branch_union_joint"
                            ]["branches"].items()
                        },
                    }
                    for direction in (
                        "within_tau2",
                        "within_tau81",
                        "cal_tau2_test_tau81",
                        "cal_tau81_test_tau2",
                    )
                },
            }
    return out


def _cross_tau_detail(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_folds: np.ndarray,
    test_folds: np.ndarray,
    alpha: float,
    test_scale: np.ndarray | None,
    test_base_width: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Per-test-sample covered flags / widths for cross-tau calibration."""
    covered = np.zeros(len(test_scores), dtype=bool)
    widths = np.zeros(len(test_scores), dtype=float)
    quantiles: list[float] = []
    for fold in range(N_FOLDS):
        cal = cal_scores[cal_folds != fold]
        test_idx = np.where(test_folds == fold)[0]
        q = conformal_quantile(cal, alpha)
        quantiles.append(q)
        covered[test_idx] = test_scores[test_idx] <= q
        if test_base_width is not None:
            widths[test_idx] = test_base_width[test_idx] + 2.0 * q
        elif test_scale is None:
            widths[test_idx] = 2.0 * q
        else:
            widths[test_idx] = 2.0 * q * test_scale[test_idx]
    return covered, widths, quantiles


def _cross_tau_coverage(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_folds: np.ndarray,
    test_folds: np.ndarray,
    alpha: float,
    test_scale: np.ndarray | None,
    test_base_width: np.ndarray | None = None,
) -> dict:
    covered, widths, quantiles = _cross_tau_detail(
        cal_scores,
        test_scores,
        cal_folds,
        test_folds,
        alpha,
        test_scale,
        test_base_width,
    )
    return _coverage_summary(covered, widths, quantiles)


COND_MIN_N = 20
GROUP_KAPPA_MIN_N = 10


def _group_coverage(
    covered: np.ndarray,
    widths: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Coverage / width / binomial SE within each observable state group."""
    out: dict = {}
    for label in sorted({str(v) for v in labels.tolist()}):
        idx = labels.astype(str) == label
        n = int(idx.sum())
        if n == 0:
            continue
        cov = float(covered[idx].mean())
        out[label] = {
            "n": n,
            "coverage": cov,
            "coverage_se": float(math.sqrt(cov * (1.0 - cov) / n)),
            "mean_width": float(widths[idx].mean()),
        }
    return out


def _joint_counts(a: np.ndarray, b: np.ndarray) -> dict:
    """Joint counts of two label vectors, keyed 'a_label|b_label'."""
    out: dict = {}
    for la in sorted(set(a.tolist())):
        for lb in sorted(set(b.tolist())):
            n_joint = int(((a == la) & (b == lb)).sum())
            if n_joint:
                out[f"{la}|{lb}"] = n_joint
    return out


def _group_kappa(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
) -> dict:
    """Per-group score-distribution shift and the plug-in coverage bound."""
    out: dict = {}
    for label in sorted(
        {str(v) for v in cal_labels.tolist()}
        & {str(v) for v in test_labels.tolist()}
    ):
        ci = cal_labels.astype(str) == label
        ti = test_labels.astype(str) == label
        n_cal = int(ci.sum())
        n_test = int(ti.sum())
        if n_cal < GROUP_KAPPA_MIN_N or n_test < GROUP_KAPPA_MIN_N:
            continue
        k = ks_distance(cal_scores[ci], test_scores[ti])
        out[label] = {
            "n_cal": n_cal,
            "n_test": n_test,
            "kappa_conf": k,
            "bound": max(0.0, 1.0 - alpha - k),
        }
    return out


def _conditional_block(
    covered: np.ndarray,
    widths: np.ndarray,
    groups: dict,
    cal_scores: np.ndarray | None = None,
    test_scores: np.ndarray | None = None,
    cal_groups: dict | None = None,
) -> dict:
    """Group-conditional coverage for one method and one calibration direction."""
    out: dict = {"by_group": {}}
    for name, labels in groups.items():
        block = _group_coverage(covered, widths, labels)
        covs = [
            v["coverage"] for v in block.values() if v["n"] >= COND_MIN_N
        ]
        out["by_group"][name] = {
            "groups": block,
            "worst_gap_n20": (
                max(0.0, (1.0 - ALPHA) - min(covs)) if covs else 0.0
            ),
            "min_coverage_n20": min(covs) if covs else None,
            "max_coverage_n20": max(covs) if covs else None,
        }
        if cal_scores is not None and cal_groups is not None:
            out["by_group"][name]["group_kappa_conf"] = _group_kappa(
                cal_scores,
                test_scores,
                cal_groups[name],
                labels,
                ALPHA,
            )
    return out


# ---------------------------------------------------------------------------
# Step 7 (D2'): value-error decomposition diagnostics
# ---------------------------------------------------------------------------

D2_TAUS = (2, 81)
D2_PHASES = ("anchor", "freeze", "post")
THRESHOLDS = {
    "Wide_Rebalance_2pct": 0.02,
    "Narrow_Rebalance_1pct": 0.01,
}


def _predict_state(
    m: np.ndarray,
    cov: np.ndarray,
    kappa: float,
    mu: float,
    sigma: float,
    h: dict,
    regime_next: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step prediction of x=(log S, eta, log S_last_open)."""
    if regime_next == REGIME_OPEN:
        phi_eta, q_eta = 0.0, 0.0
    elif regime_next == REGIME_EXTENDED:
        phi_eta, q_eta = h["ext_phi"], h["ext_sd"] ** 2
    elif regime_next in (REGIME_WEEKEND, REGIME_HOLIDAY):
        phi_eta, q_eta = h["stale_phi"], h["stale_sd"] ** 2
    elif regime_next == REGIME_HALT:
        phi_eta, q_eta = 1.0, 0.0
    else:
        raise ValueError(f"unknown regime: {regime_next}")
    a = np.array(
        [
            [1.0 - kappa, 0.0, 0.0],
            [0.0, phi_eta, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    c = np.array([kappa * mu, 0.0, 0.0])
    if regime_next == REGIME_OPEN:
        # At OPEN the anchor resets to the current latent price: it inherits
        # the same innovation as log S, so the S and anchor entries share
        # sigma^2 (perfect correlation), not an independent zero-variance copy.
        a[2, :] = a[0, :]
        c[2] = c[0]
        q = np.zeros((3, 3))
        q[0, 0] = q[2, 2] = q[0, 2] = q[2, 0] = sigma**2
    else:
        q = np.diag([sigma**2, q_eta, 0.0])
    return a @ m + c, a @ cov @ a.T + q


def _obs_model(regime_t: int, h: dict):
    if regime_t == REGIME_OPEN:
        return np.array([1.0, 0.0, 0.0]), h["open_sd"] ** 2
    if regime_t == REGIME_EXTENDED:
        return np.array([1.0, 1.0, 0.0]), 0.0
    if regime_t in (REGIME_WEEKEND, REGIME_HOLIDAY):
        return np.array([0.0, 1.0, 1.0]), 0.0
    return None, None


def _update_state(
    m: np.ndarray,
    cov: np.ndarray,
    y: float,
    obs_vec: np.ndarray,
    obs_var: float,
):
    s = float(obs_vec @ cov @ obs_vec) + obs_var
    if s <= 1e-18:
        return m, cov, np.zeros(3)
    gain = cov @ obs_vec / s
    m = m + gain * (y - float(obs_vec @ m))
    ikh = np.eye(3) - np.outer(gain, obs_vec)
    # Joseph form; the K R K^T term matters for OPEN observations (R>0).
    cov = ikh @ cov @ ikh.T + np.outer(gain, gain) * obs_var
    return m, 0.5 * (cov + cov.T), gain


def filter_episode(
    market,
    episode: dict,
    kappa: float,
    mu: float,
    sigma: float,
    h: dict,
):
    """Kalman-filter the identified model on the observed episode history.

    Returns the posterior at each step ``pre..post``, the one-step predictive
    of the observed price ``y_{t+1}`` for ``t=pre..post-1`` (with a degenerate
    flag for HALT), and the max spectral norm of the prior-to-posterior mean
    map ``I-KH`` over the updates (a contraction proxy, not the W1 constant).
    """
    regime = market.regime
    logp = np.log(market.pool_price)
    pre, post = episode["pre"], episode["post"]
    t0 = last_open_before(regime, pre)
    var_o = h["open_sd"] ** 2
    m = np.array([logp[t0], 0.0, logp[t0]])
    cov = np.zeros((3, 3))
    cov[0, 0] = var_o
    cov[2, 2] = var_o
    cov[0, 2] = cov[2, 0] = var_o

    posterior: dict = {}
    predictive: dict = {}
    contraction: list[float] = []
    if t0 >= pre:
        posterior[t0] = (m.copy(), cov.copy())
    for t in range(t0 + 1, post + 1):
        m, cov = _predict_state(m, cov, kappa, mu, sigma, h, int(regime[t]))
        obs_vec, obs_var = _obs_model(int(regime[t]), h)
        if obs_vec is None:
            # HALT: P_t = P_{t-1}; the predictive is a point mass.
            predictive[t - 1] = (float(logp[t - 1]), 0.0, True)
        else:
            predictive[t - 1] = (
                float(obs_vec @ m),
                float(obs_vec @ cov @ obs_vec) + float(obs_var),
                False,
            )
            m, cov, gain = _update_state(
                m, cov, float(logp[t]), obs_vec, obs_var
            )
            ikh = np.eye(3) - np.outer(gain, obs_vec)
            contraction.append(float(np.linalg.norm(ikh, 2)))
        if t >= pre:
            posterior[t] = (m.copy(), cov.copy())
    return posterior, predictive, contraction


def _folded_abs_mean(mu: float, sd: float, c: float) -> float:
    """E|X-c| for X ~ N(mu, sd^2)."""
    if sd <= 1e-15:
        return abs(mu - c)
    d = abs(mu - c) / sd
    return sd * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * d * d) + abs(
        mu - c
    ) * math.erf(d / math.sqrt(2.0))


def _normal_tv(m1: float, s1: float, m2: float, s2: float, n_grid: int = 2001) -> float:
    """TV between two 1-D Gaussians by numerical integration (or point mass)."""
    if s1 <= 0.0 or s2 <= 0.0:
        if s1 <= 0.0 and s2 <= 0.0:
            return 0.0 if abs(m1 - m2) <= 1e-12 else 1.0
        return 1.0
    lo = min(m1 - 8.0 * s1, m2 - 8.0 * s2)
    hi = max(m1 + 8.0 * s1, m2 + 8.0 * s2)
    x = np.linspace(lo, hi, n_grid)
    dx = float(x[1] - x[0])
    f = np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * math.sqrt(2.0 * math.pi))
    g = np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 * math.sqrt(2.0 * math.pi))
    return float(0.5 * np.sum(np.abs(f - g)) * dx)


def _p_rebalance(
    anchor: float,
    theta: float,
    mean: float,
    sd: float,
    degenerate: bool,
) -> float:
    """P(|P/anchor-1| > theta) under a Gaussian or point-mass predictive."""
    hi = math.log(anchor * (1.0 + theta))
    lo = math.log(anchor * (1.0 - theta))
    if degenerate or sd <= 0.0:
        return 1.0 if (mean > hi or mean < lo) else 0.0
    p_hi = 0.5 * (1.0 - math.erf((hi - mean) / (sd * math.sqrt(2.0))))
    p_lo = 0.5 * (1.0 + math.erf((lo - mean) / (sd * math.sqrt(2.0))))
    return float(min(1.0, max(0.0, p_hi + p_lo)))


def _dist_summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
        "frac_ge_0.99": float(np.mean(a >= 0.99)),
    }


def d2prime_belief_study(offline: list, kappa: float, mu: float, h: dict) -> dict:
    """W1 estimation term and predictive-TV information term, per tau/phase."""
    out = {
        "config": {
            "taus": list(D2_TAUS),
            "phases": {
                "anchor": "t = pre-anchor (last informative step before the freeze)",
                "freeze": "t = pre+1 .. post-1",
                "post": "t = post-anchor (first OPEN after the freeze)",
            },
            "w1": "E|log S_t - log P_t| under the identified posterior; equals "
            "W1(posterior, delta_{log P_t}) (naive treats P as S)",
            "tv": "TV between identified and naive one-step predictives of "
            "log P_{t+1}, evaluated at the observed history",
            "oracle_w1": "mean |log S_true_t - log P_t| along the same "
            "histories, using the simulator's latent price for validation only",
        },
        "filter_consistency": {},
        "post_update_check": {},
        "contraction_check": {},
        "by_tau": {},
    }

    consistency: list[dict] = []
    post_checks: list[dict] = []
    contraction_all: list[float] = []
    for m in offline:
        sigma_path = estimate_sigma_open(m)
        for episode in extract_freeze_episodes(m):
            if episode["tau"] not in D2_TAUS:
                continue
            posterior, _, contraction = filter_episode(
                m, episode, kappa, mu, sigma_path, h
            )
            mean, cov = posterior[episode["pre"]]
            ref_mean, ref_cov = kalman_pre_belief(
                m, episode["pre"], kappa, mu, sigma_path, h
            )
            consistency.append(
                {
                    "max_abs_mean": float(np.max(np.abs(mean - ref_mean))),
                    "max_abs_cov": float(np.max(np.abs(cov - ref_cov))),
                }
            )
            # Independent scalar-Kalman check of the R>0 (OPEN) update at
            # post-anchor; the pre-anchor cross-check never exercises R>0.
            m_prev, cov_prev = posterior[episode["post"] - 1]
            prior_mean, prior_cov = _predict_state(
                m_prev,
                cov_prev,
                kappa,
                mu,
                sigma_path,
                h,
                int(m.regime[episode["post"]]),
            )
            p_var = float(prior_cov[0, 0])
            var_o = h["open_sd"] ** 2
            ref_var = p_var * var_o / (p_var + var_o)
            ref_mean = float(prior_mean[0]) + (p_var / (p_var + var_o)) * (
                float(np.log(m.pool_price[episode["post"]]))
                - float(prior_mean[0])
            )
            post_mean, post_cov = posterior[episode["post"]]
            post_checks.append(
                {
                    "abs_mean_err": abs(float(post_mean[0]) - ref_mean),
                    "rel_var_err": abs(float(post_cov[0, 0]) - ref_var)
                    / max(ref_var, 1e-300),
                }
            )
            if contraction:
                contraction_all.extend(contraction)
    out["filter_consistency"] = {
        "n_episodes": len(consistency),
        "max_abs_mean": max(c["max_abs_mean"] for c in consistency),
        "max_abs_cov": max(c["max_abs_cov"] for c in consistency),
        "note": "filter_episode posterior at pre-anchor vs kalman_pre_belief "
        "(same Kalman equations; should be near machine precision). This "
        "covers only the noiseless EXTENDED segment (R=0); the OPEN update "
        "with R>0 is covered by post_update_check.",
    }
    out["post_update_check"] = {
        "n_episodes": len(post_checks),
        "max_abs_mean_err": max(c["abs_mean_err"] for c in post_checks),
        "max_rel_var_err": max(c["rel_var_err"] for c in post_checks),
        "note": "post-anchor OPEN update vs the analytic scalar Kalman "
        "identity var = P R/(P+R), mean = prior + P/(P+R)(y-prior), with P "
        "the filter's prior variance at post. Independent of _update_state; "
        "exercises the R>0 path that the pre-anchor cross-check misses.",
    }
    c = np.asarray(contraction_all, dtype=float)
    out["contraction_check"] = {
        "posterior_mean_map_norm": {
            "n": int(c.size),
            "mean": float(c.mean()),
            "median": float(np.median(c)),
            "p90": float(np.quantile(c, 0.90)),
            "max": float(c.max()),
        },
        "l_t_proxy": 1.0,
        "note": "I-KH spectral norm is a Euclidean proxy for the filter's "
        "mean-map Lipschitz constant, not the W1 constant; it is non-normal "
        "and can exceed 1, so a value >1 is NOT a violation of the W1 "
        "contraction. L_T=1 because additive transition noise can be "
        "coupled; this counts log S once, whereas the literal Euclidean "
        "norm of the OPEN transition (log S duplicated into the anchor "
        "coordinate) is sqrt(2)(1-kappa), a coordinate-duplication "
        "artifact. The finite-horizon episodes (H<=83) give a finite "
        "composition regardless; gamma L_Phi L_T < 1 is only needed for the "
        "infinite-horizon corollary. The per-step diagnostics below do not "
        "rely on it.",
    }

    for tau in D2_TAUS:
        w1 = {phase: [] for phase in D2_PHASES}
        oracle_w1 = {phase: [] for phase in D2_PHASES}
        tv = {"anchor": [], "freeze": []}
        w1_episode = {phase: [] for phase in D2_PHASES}
        oracle_w1_episode = {phase: [] for phase in D2_PHASES}
        tv_episode = {"anchor": [], "freeze": []}
        n_episodes = 0
        for m in offline:
            sigma_path = estimate_sigma_open(m)
            sigma_all_path = estimate_sigma_all(m)
            kappa_p, mu_p = estimate_naive_kappa([m])
            logp = np.log(m.pool_price)
            for episode in extract_freeze_episodes(m):
                if episode["tau"] != tau:
                    continue
                n_episodes += 1
                pre, post = episode["pre"], episode["post"]
                posterior, predictive, _ = filter_episode(
                    m, episode, kappa, mu, sigma_path, h
                )
                ep_w1 = {phase: [] for phase in D2_PHASES}
                ep_oracle = {phase: [] for phase in D2_PHASES}
                for t in range(pre, post + 1):
                    mean, cov = posterior[t]
                    val = _folded_abs_mean(
                        float(mean[0]),
                        math.sqrt(max(float(cov[0, 0]), 0.0)),
                        float(logp[t]),
                    )
                    phase = (
                        "anchor"
                        if t == pre
                        else ("post" if t == post else "freeze")
                    )
                    ep_w1[phase].append(val)
                    w1[phase].append(val)
                    ep_oracle[phase].append(
                        abs(math.log(float(m.true_price[t])) - float(logp[t]))
                    )
                    oracle_w1[phase].append(ep_oracle[phase][-1])
                ep_tv = {"anchor": [], "freeze": []}
                for t in range(pre, post):
                    pred_mean, pred_var, degenerate = predictive[t]
                    y_t = float(logp[t])
                    n_mean = y_t + kappa_p * (mu_p - y_t)
                    n_sd = sigma_all_path
                    if degenerate or pred_var <= 0.0:
                        val = 1.0
                    else:
                        val = _normal_tv(
                            pred_mean, math.sqrt(pred_var), n_mean, n_sd
                        )
                    phase = "anchor" if t == pre else "freeze"
                    ep_tv[phase].append(val)
                    tv[phase].append(val)
                for phase in D2_PHASES:
                    if ep_w1[phase]:
                        w1_episode[phase].append(float(np.mean(ep_w1[phase])))
                        oracle_w1_episode[phase].append(
                            float(np.mean(ep_oracle[phase]))
                        )
                for phase in ("anchor", "freeze"):
                    if ep_tv[phase]:
                        tv_episode[phase].append(float(np.mean(ep_tv[phase])))
        out["by_tau"][str(tau)] = {
            "n_episodes": n_episodes,
            "w1_step_pooled": {
                phase: _dist_summary(w1[phase]) for phase in D2_PHASES
            },
            "w1_episode_mean": {
                phase: _dist_summary(w1_episode[phase]) for phase in D2_PHASES
            },
            "oracle_w1_step_pooled": {
                phase: _dist_summary(oracle_w1[phase]) for phase in D2_PHASES
            },
            "oracle_w1_episode_mean": {
                phase: _dist_summary(oracle_w1_episode[phase])
                for phase in D2_PHASES
            },
            "tv_step_pooled": {
                phase: _dist_summary(tv[phase]) for phase in ("anchor", "freeze")
            },
            "tv_episode_mean": {
                phase: _dist_summary(tv_episode[phase])
                for phase in ("anchor", "freeze")
            },
        }
    return out


def d2prime_policy_study(
    offline: list,
    policies: dict,
    thresholds: dict,
    kappa: float,
    mu: float,
    h: dict,
) -> dict:
    """Threshold-margin and first-order action-flip diagnostics."""
    out = {
        "config": {
            "margin": "|P_t/anchor-1| - theta at decision steps (in position, "
            "OPEN/EXTENDED); m>0 means the observed path rebalances",
            "flip": "|p_id - p_naive| = minimal-coupling probability that the "
            "rebalance decision at t+1 differs, where p_model = "
            "P(|P_{t+1}/anchor-1| > theta) under that model's predictive; "
            "positions are taken from the observed trajectory (first-order)",
        },
        "by_policy": {},
    }
    for name, policy in policies.items():
        theta = thresholds[name]
        out["by_policy"][name] = {"threshold": theta, "by_tau": {}}
        for tau in D2_TAUS:
            margins: list[float] = []
            post_margins: list[float] = []
            flips: list[float] = []
            p_id_all: list[float] = []
            p_naive_all: list[float] = []
            n_episodes = 0
            for m in offline:
                sigma_path = estimate_sigma_open(m)
                sigma_all_path = estimate_sigma_all(m)
                kappa_p, mu_p = estimate_naive_kappa([m])
                logp = np.log(m.pool_price)
                for episode in extract_freeze_episodes(m):
                    if episode["tau"] != tau:
                        continue
                    n_episodes += 1
                    pre, post = episode["pre"], episode["post"]
                    _, predictive, _ = filter_episode(
                        m, episode, kappa, mu, sigma_path, h
                    )
                    pos = position_before(m, policy, pre)
                    for t in range(pre, post):
                        r = int(m.regime[t])
                        next_r = int(m.regime[t + 1])
                        if pos.in_pos and r in (REGIME_OPEN, REGIME_EXTENDED):
                            margins.append(
                                abs(float(m.pool_price[t]) / pos.anchor - 1.0)
                                - theta
                            )
                        action = policy(
                            t, r, next_r, float(m.pool_price[t]), pos
                        )
                        apply_action(pos, action, float(m.pool_price[t]))
                        if pos.in_pos and int(m.regime[t + 1]) in (
                            REGIME_OPEN,
                            REGIME_EXTENDED,
                        ):
                            pred_mean, pred_var, degenerate = predictive[t]
                            y_t = float(logp[t])
                            n_mean = y_t + kappa_p * (mu_p - y_t)
                            n_sd = sigma_all_path
                            p_id = _p_rebalance(
                                pos.anchor,
                                theta,
                                pred_mean,
                                math.sqrt(max(pred_var, 0.0)),
                                degenerate,
                            )
                            p_naive = _p_rebalance(
                                pos.anchor, theta, n_mean, n_sd, False
                            )
                            flips.append(abs(p_id - p_naive))
                            p_id_all.append(p_id)
                            p_naive_all.append(p_naive)
                    if pos.in_pos and int(m.regime[post]) in (
                        REGIME_OPEN,
                        REGIME_EXTENDED,
                    ):
                        post_margins.append(
                            abs(float(m.pool_price[post]) / pos.anchor - 1.0)
                            - theta
                        )

            margin_summary = _dist_summary(margins)
            if margins:
                a = np.asarray(margins, dtype=float)
                margin_summary.update(
                    {
                        "frac_rebalance": float(np.mean(a > 0.0)),
                        "frac_abs_le_0.0025": float(np.mean(np.abs(a) <= 0.0025)),
                        "frac_abs_le_0.005": float(np.mean(np.abs(a) <= 0.005)),
                        "frac_abs_le_0.01": float(np.mean(np.abs(a) <= 0.01)),
                        "min_abs": float(np.min(np.abs(a))),
                    }
                )
            post_summary = _dist_summary(post_margins)
            if post_margins:
                a = np.asarray(post_margins, dtype=float)
                post_summary.update(
                    {
                        "frac_rebalance": float(np.mean(a > 0.0)),
                        "frac_abs_le_0.005": float(np.mean(np.abs(a) <= 0.005)),
                    }
                )
            flip_summary = _dist_summary(flips)
            if flips:
                flip_summary.update(
                    {
                        "mean_p_id": float(np.mean(p_id_all)),
                        "mean_p_naive": float(np.mean(p_naive_all)),
                        "frac_gt_0.5": float(np.mean(np.asarray(flips) > 0.5)),
                    }
                )
            out["by_policy"][name]["by_tau"][str(tau)] = {
                "n_episodes": n_episodes,
                "margin": margin_summary,
                "post_anchor_margin": post_summary,
                "flip": flip_summary,
            }
    return out


def d2prime_study(
    offline: list,
    policies: dict,
    thresholds: dict,
    kappa: float,
    mu: float,
    h: dict,
) -> dict:
    return {
        "belief": d2prime_belief_study(offline, kappa, mu, h),
        "policy": d2prime_policy_study(offline, policies, thresholds, kappa, mu, h),
        "note": "D2' decomposition diagnostics: W1 estimation term, predictive-TV "
        "information term, and boundary/action-flip frequency. These are "
        "along-observed-history sensitivities, not a uniform sup over beliefs; "
        "the third block is a diagnostic refinement for observation-type "
        "policies, not an independent bound term.",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    diag = pd.read_csv(TRUE_PATHS)
    diag = diag.sort_values(["window", "seed", "episode_step"], ignore_index=True)

    # Offline paths = frozen 128.
    offline = [
        simulate_market([seed, wi])
        for wi in range(len(WINDOWS))
        for seed in OFFLINE_SEEDS
    ]

    # Cross-check every regenerated path against the frozen diagnostics.
    # Volume is not stored in `true_paths.csv.gz`; matching all prices to
    # machine precision pins the same RNG stream, which also generates volume.
    checks = []
    for wi, window in enumerate(WINDOWS):
        max_true = 0.0
        max_pool = 0.0
        max_sigma = 0.0
        for si, seed in enumerate(OFFLINE_SEEDS):
            m = offline[wi * len(OFFLINE_SEEDS) + si]
            g = diag[(diag["window"] == window) & (diag["seed"] == seed)]
            max_true = max(
                max_true,
                float(np.max(np.abs(m.true_price[:-1] - g["true_price"].to_numpy()))),
            )
            max_pool = max(
                max_pool,
                float(np.max(np.abs(m.pool_price[:-1] - g["pool_price"].to_numpy()))),
            )
            max_sigma = max(max_sigma, float(abs(m.sigma - g["sigma"].iloc[0])))
        checks.append(
            {
                "window": window,
                "n_paths": len(OFFLINE_SEEDS),
                "max_abs_true_price": max_true,
                "max_abs_pool_price": max_pool,
                "max_sigma_abs_diff": max_sigma,
                "volume_note": "volume is not in true_paths.csv.gz; the exact "
                "price match pins the same RNG stream that generates volume",
            }
        )

    sigma_open = np.array([estimate_sigma_open(m) for m in offline])
    sigma_all = np.array([estimate_sigma_all(m) for m in offline])
    kappa_p, mu_p = estimate_naive_kappa(offline)
    mapping = estimate_mapping(diag)

    # Parameters used by the estimated plug-in.
    kappa_est = 0.00285          # G0 step 2 pooled estimate
    mu_est = 4.614               # G0 step 2 intercept / kappa_hat
    jump_est = 0.003             # G0 step 2 jump-rate estimate
    h = mapping

    rng = np.random.default_rng(20261022)
    results: dict = {
        "config": {
            "n_offline_paths": len(offline),
            "offline_seeds": [int(OFFLINE_SEEDS.start), int(OFFLINE_SEEDS.stop - 1)],
            "n_eval_paths": len(WINDOWS) * len(EVAL_SEEDS),
            "r_main": R_MAIN,
            "r_block": R_BLOCK,
            "n_blocks": N_BLOCKS,
            "n_eval_block": N_EVAL_BLOCK,
            "policies": list(POLICIES),
            "event_seeds": list(EVENT_SEEDS),
        },
        "data_checks": checks,
        "observation_mapping_estimate": mapping,
        "parameter_estimates": {
            "kappa_plug_in": kappa_est,
            "mu_plug_in": mu_est,
            "jump_prob_plug_in": jump_est,
            "sigma_open_mean": float(sigma_open.mean()),
            "sigma_open_sd": float(sigma_open.std(ddof=1)),
            "sigma_all_mean": float(sigma_all.mean()),
            "sigma_all_sd": float(sigma_all.std(ddof=1)),
            "naive_kappa_p": kappa_p,
            "naive_mu_p": mu_p,
        },
        "policies": {},
        "block_study": {},
        "limits": [
            "True S is used for scoring only and never enters a feature or policy.",
            "kappa/mu/jump are the G0 step-2 estimates; the block study varies only sigma estimation.",
            "The observation-mapping parameters are estimated with true S as a diagnostic, not from P alone.",
            "The estimated model resamples the 128 per-path sigma estimates; measurement noise in sigma is not deconvolved.",
            "The block study uses new simulator paths to estimate sigma/kappa_P; it is a simulation study, not the frozen dataset.",
            "D2' diagnostics are along-observed-history sensitivities, not a uniform sup over beliefs.",
            "D2' W1 compares the identified posterior over log S_t with the naive point belief delta_{log P_t} (naive treats P as S); it is a one-dimensional log-S marginal, not the joint law over (S, eta, anchor).",
            "D2' W1/TV/flip are plug-in estimates with kappa/mu/h/sigma treated as fixed; no confidence intervals are attached to the three terms (128 paths).",
            "D2' naive AR(1) parameters (kappa_P, mu_P, sigma_all) are fitted per path (336 observations), so per-path kappa_P is noisy (e.g. 0.015 vs pooled 0.0028); this matches the step-5/6 per-path naive model, not step 5's pooled naive check.",
            "D2' post-anchor W1 is checked against an oracle |log S_true - log P|; the pre-anchor filter cross-check covers only the R=0 segment, while post_update_check covers the OPEN R>0 update analytically.",
            "D2' L_T=1 is stated in a metric that counts log S once; the literal Euclidean operator norm of the OPEN transition (which duplicates log S into the anchor coordinate) is sqrt(2)(1-kappa), a coordinate-duplication artifact.",
            "D2' predictive TV is numerical integration of Gaussian one-step predictives; HALT gives TV=1 because the identified model knows P is frozen and the naive model does not.",
            "Jumps (prob 0.003) are omitted in the D2' filter and predictive; both sides omit them.",
            "D2' action-flip probability is first-order: positions come from the observed trajectory and the flip probability is the minimal-coupling Bernoulli disagreement |p_id-p_naive|.",
            "Le Cam deficiency is not estimated in D2'; the predictive TV is the information-term estimate, and Le Cam only controls Bayes-optimal values.",
            "Event-level conditional coverage uses observable pre-decision groups (onset action/distance/side); in_position is degenerate for both target policies (always in position), so the flat group is empty.",
            "post_rebalance is a post-hoc boundary diagnostic, not a prediction-time condition; it is not a deployable conditional guarantee and cannot be fixed by pre-decision Mondrian calibration.",
            "Conditional group coverage is descriptive: small groups are shown with binomial SE, worst_gap_n20 excludes groups with n<20, group kappa_conf is a plug-in KS distance without a bootstrap CI, and Prop 7 remains a marginal (not conditional) guarantee.",
            "branch_union uses post_rebalance labels in calibration only; it is the minimal branch-aware fix that shows the width cost of branch-conditional coverage. It is not a test-time conditional method, and its branch labels are post-decision outcomes.",
            "branch_kappa is a plug-in KS distance on branch sizes 16-112. The KS sampling floor under equal distributions is large (asymptotic iid mean 0.13-0.24 and 95% quantile 0.21-0.38 over these branch sizes), so raw kappa_conf is not evidence of shift without the null reference; kappa_excess is descriptive. The KS bound and the empirical trivial decomposition are both plug-in, not finite-sample guarantees; the trivial decomposition uses the evaluated method's observed overall coverage (marginal over branches) and is exact only when all non-coverage sits in that branch. The paired path bootstrap captures cross-tau dependence but treats branch labels as resampled outcomes. The iid KS null ignores the cross-tau path pairing; positive dependence typically narrows its dispersion, so the reference is conservative.",
            "branch_union_joint compares the branch-union and the unmodified marginal pipeline on the same test events and folds. Coverage dominance (union coverage >= marginal coverage) is structural, because max(q_hold,q_rebalance) >= q_pooled per fold; the reported gain is one-sided and its SE is the binomial SE of the gain. The five-way taxonomy uses normal-approximation 95% intervals on branch sizes 16-112, is descriptive rather than simultaneous, and labels most small gains ambiguous. The width ratio is a calibration-side quantity for scale-family scores but includes the target base width for CQR. Cross-regime residual_gap/kappa is zero when the union covers and is not a calibrated error rate; the kappa bound is descriptive, not a prospective certificate.",
            "Pre-decision Mondrian uses only onset_action/onset_distance/onset_side labels in interval construction; onset_action=rebalance has n=8 at tau=2, so most folds have no finite group quantile and fall back to the marginal quantile. The pre-group fix is therefore partial and fallback counts are reported.",
            "Mondrian post_rebalance coverage is an evaluation diagnostic only; conditioning on that post-decision branch is impossible at prediction time, so pre-decision Mondrian is expected to leave the post-branch gap near its trivial bound.",
        ],
    }

    # Main table: evaluate on the frozen 128 paths (in-sample for the model).
    for name, policy in POLICIES.items():
        true_vals = np.array(
            [rollout(m.true_price, m.pool_price, m.regime, m.volume, policy) for m in offline]
        )
        oracle_vals = []
        est_vals = []
        est_true_sigma_vals = []
        naive_model_vals = []
        naive_mark_vals = []

        for m in offline:
            reps_oracle = []
            reps_est = []
            reps_true_sigma = []
            reps_naive = []
            for _ in range(R_MAIN):
                s, p, reg = simulate_model(
                    rng,
                    m.sigma,
                    KAPPA,
                    MU_LOG,
                    0.005,
                    h["open_sd"],
                    h["ext_phi"],
                    h["ext_sd"],
                    h["stale_phi"],
                    h["stale_sd"],
                )
                reps_oracle.append(rollout(s, p, reg, m.volume, policy))

                sig_draw = float(rng.choice(sigma_open))
                s, p, reg = simulate_model(
                    rng,
                    sig_draw,
                    kappa_est,
                    mu_est,
                    jump_est,
                    h["open_sd"],
                    h["ext_phi"],
                    h["ext_sd"],
                    h["stale_phi"],
                    h["stale_sd"],
                )
                reps_est.append(rollout(s, p, reg, m.volume, policy))

                s, p, reg = simulate_model(
                    rng,
                    m.sigma,
                    kappa_est,
                    mu_est,
                    jump_est,
                    h["open_sd"],
                    h["ext_phi"],
                    h["ext_sd"],
                    h["stale_phi"],
                    h["stale_sd"],
                )
                reps_true_sigma.append(rollout(s, p, reg, m.volume, policy))

                sig_naive = float(rng.choice(sigma_all))
                s, p, reg = simulate_naive(rng, sig_naive, kappa_p, mu_p, jump_est)
                reps_naive.append(rollout(s, p, reg, m.volume, policy))

            oracle_vals.append(float(np.mean(reps_oracle)))
            est_vals.append(float(np.mean(reps_est)))
            est_true_sigma_vals.append(float(np.mean(reps_true_sigma)))
            naive_model_vals.append(float(np.mean(reps_naive)))
            naive_mark_vals.append(
                rollout(m.pool_price, m.pool_price, m.regime, m.volume, policy)
            )

        oracle_vals = np.array(oracle_vals)
        est_vals = np.array(est_vals)
        est_true_sigma_vals = np.array(est_true_sigma_vals)
        naive_model_vals = np.array(naive_model_vals)
        naive_mark_vals = np.array(naive_mark_vals)

        results["policies"][name] = {
            "definition": (
                "wide 30%, enter on OPEN/EXTENDED, rebalance at 2%, hold through freezes"
                if name == "Wide_Rebalance_2pct"
                else "narrow 5%, enter on OPEN/EXTENDED, rebalance at 1%, hold through freezes"
            ),
            "true": summarize(true_vals, seed=20261023),
            "oracle": {
                **summarize(oracle_vals, seed=20261024),
                "paired_diff_vs_true": paired_diff(oracle_vals, true_vals),
            },
            "estimated": {
                **summarize(est_vals, seed=20261025),
                "paired_diff_vs_true": paired_diff(est_vals, true_vals),
            },
            "estimated_true_sigma": {
                **summarize(est_true_sigma_vals, seed=20261026),
                "paired_diff_vs_true": paired_diff(est_true_sigma_vals, true_vals),
            },
            "naive_model": {
                **summarize(naive_model_vals, seed=20261027),
                "paired_diff_vs_true": paired_diff(naive_model_vals, true_vals),
            },
            "naive_mark": {
                **summarize(naive_mark_vals, seed=20261028),
                "paired_diff_vs_true": paired_diff(naive_mark_vals, true_vals),
            },
        }

    # Block study: out-of-sample offline blocks, fixed 512-path eval set.
    n_eval_seeds = N_EVAL_BLOCK // len(WINDOWS)
    eval_markets = [
        simulate_market([seed, wi])
        for wi in range(len(WINDOWS))
        for seed in list(EVAL_SEEDS)[:n_eval_seeds]
    ]

    for name, policy in POLICIES.items():
        true_eval = np.array(
            [rollout(m.true_price, m.pool_price, m.regime, m.volume, policy) for m in eval_markets]
        )
        est_blocks = []
        naive_blocks = []
        sigma_means = []
        for b in range(N_BLOCKS):
            seeds = range(BLOCK_SEEDS_START + b * 128, BLOCK_SEEDS_START + (b + 1) * 128)
            block = [
                simulate_market([seed, wi])
                for wi in range(len(WINDOWS))
                for seed in seeds
            ]
            sig_open_block = np.array([estimate_sigma_open(m) for m in block])
            sig_all_block = np.array([estimate_sigma_all(m) for m in block])
            kappa_p_block, mu_p_block = estimate_naive_kappa(block)
            sigma_means.append(float(sig_open_block.mean()))

            est_vals_b = []
            naive_vals_b = []
            for m in eval_markets:
                reps_est = []
                reps_naive = []
                for _ in range(R_BLOCK):
                    sig_draw = float(rng.choice(sig_open_block))
                    s, p, reg = simulate_model(
                        rng,
                        sig_draw,
                        kappa_est,
                        mu_est,
                        jump_est,
                        h["open_sd"],
                        h["ext_phi"],
                        h["ext_sd"],
                        h["stale_phi"],
                        h["stale_sd"],
                    )
                    reps_est.append(rollout(s, p, reg, m.volume, policy))

                    sig_naive = float(rng.choice(sig_all_block))
                    s, p, reg = simulate_naive(
                        rng, sig_naive, kappa_p_block, mu_p_block, jump_est
                    )
                    reps_naive.append(rollout(s, p, reg, m.volume, policy))
                est_vals_b.append(float(np.mean(reps_est)))
                naive_vals_b.append(float(np.mean(reps_naive)))
            est_blocks.append(float(np.mean(est_vals_b)))
            naive_blocks.append(float(np.mean(naive_vals_b)))

        est_blocks = np.array(est_blocks)
        naive_blocks = np.array(naive_blocks)
        true_mean = float(true_eval.mean())
        est_bias = est_blocks - true_mean
        naive_bias = naive_blocks - true_mean
        results["block_study"][name] = {
            "true_eval_mean": true_mean,
            "true_eval_se": float(true_eval.std(ddof=1) / math.sqrt(len(true_eval))),
            "estimated_bias_mean": float(est_bias.mean()),
            "estimated_bias_se": float(est_bias.std(ddof=1) / math.sqrt(N_BLOCKS)),
            "estimated_bias_t": float(
                est_bias.mean() / (est_bias.std(ddof=1) / math.sqrt(N_BLOCKS))
            ),
            "naive_bias_mean": float(naive_bias.mean()),
            "naive_bias_se": float(naive_bias.std(ddof=1) / math.sqrt(N_BLOCKS)),
            "naive_bias_t": float(
                naive_bias.mean() / (naive_bias.std(ddof=1) / math.sqrt(N_BLOCKS))
            ),
            "estimated_blocks": est_blocks.tolist(),
            "naive_blocks": naive_blocks.tolist(),
            "sigma_open_block_mean": float(np.mean(sigma_means)),
        }

    # Step 6: freeze-episode (trajectory-level) conformal coverage.
    # The first seed is the primary run; the others quantify MC-draw noise.
    event_runs = {
        seed: episode_conformal_study(
            offline,
            POLICIES,
            kappa_est,
            mu_est,
            jump_est,
            h,
            seed=seed,
        )
        for seed in EVENT_SEEDS
    }
    results["event_conformal"] = event_runs[EVENT_SEEDS[0]]
    results["event_conformal_seed_robustness"] = {
        str(seed): summarize_event_coverage(run) for seed, run in event_runs.items()
    }

    # Step 7 (D2'): value-error decomposition diagnostics.
    results["d2prime"] = d2prime_study(
        offline, POLICIES, THRESHOLDS, kappa_est, mu_est, h
    )

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
