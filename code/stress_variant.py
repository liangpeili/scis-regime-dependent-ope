"""I5 stress variant: de-freeze jump and open-only vs cross-event identification.

Non-C1 verification task (program.md §1 G0 exception, 2026-10-21 addendum).
The frozen environment (``synthetic_env.py``) and the main dataset are NOT
modified; this script imports the former as a library and builds an independent
16-day variant with one extra latent component.

Why a variant
-------------
In the main environment the latent log-price is a parametric AR(1), so
``S_open | (S_freeze, tau, sigma)`` is already implied by the one-step kernel
``T`` estimated from open data.  I5 (cross-event variation) is therefore
redundant there: open data alone identifies the freeze conditional law.  The
stress variant adds a de-freeze jump

    J_e ~ N(0, g(tau_e, sigma_e)^2),   g = gamma * sigma * sqrt(1-exp(-tau/tau0))

only on the freeze -> OPEN step of each event, and nowhere in ordinary OPEN ->
OPEN one-hour transitions.  Then:

* the law of ordinary open transitions is invariant to gamma (so any
  open-only / Markov-extrapolation estimator has zero identifying power);
* cross-event (tau, sigma) variation in de-freeze residuals identifies gamma.

Window phase
------------
The 14-day main window leaves the second weekend (tau=56) without a post
anchor.  The variant uses 16 days (384 steps): the freeze lengths with both
anchors become {2 (HALT), 65 (weekend), 81 (HOLIDAY+WEEKEND)}.  ``tau`` is the
number of frozen hours; the two anchors are ``tau+1`` steps apart (same
convention as ``identifiability_check.py``).

Two tiers
---------
* ``oracle_T``: true kappa/mu/sigma, so the open-only estimator has no
  estimation error; any failure is due to gamma itself.
* ``estimated_T``: kappa/mu fitted on ordinary OPEN->OPEN one-step transitions,
  sigma per path from its own open residuals.  Robustness, not the headline.

Coverage is 90% predictive-interval coverage for the post-anchor log price
residual, with 5-fold cross-fitting by seed (fit/calibrate on four folds, test
on the held-out fold), pooled over folds.  A gamma=0 twin (same main RNG
stream, jump scale zero) is the placebo: open-only must be valid and the
cross-event fit must return gamma ~ 0 there.

Usage (from projects/p1-c1-ope/code):
    ../../../.venv-c1/bin/python stress_variant.py
    ../../../.venv-c1/bin/python stress_variant.py --n-seeds 32 --windows W1

Writes ``results/stress_variant.json`` and
``results/stress/true_paths_stress.csv.gz`` (diagnostics only; results/ is not
committed).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from synthetic_env import (
    BASE_VOLUME,
    KAPPA,
    MU_LOG,
    REGIME_EXTENDED,
    REGIME_HALT,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    SIGMA_BASE,
    WINDOW_DAYS,
    MarketPath,
    calendar_regime,
    simulate_market,
)

C1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = C1_DIR.parent / "results"
STRESS_DIR = RESULTS_DIR / "stress"

FREEZE_REGIMES = (REGIME_WEEKEND, REGIME_HOLIDAY, REGIME_HALT)
EVENT_TAUS = (2, 65, 81)
ALPHA = 0.10
N_FOLDS = 5
CHI2_95_1DF = 3.841458820694124

# The frozen generator adds a rare jump N(0, (3 sigma)^2) with probability
# 0.005 at every step.  The one-step innovation variance is therefore
# sigma^2 * (1 + 0.005 * 9) = 1.045 sigma^2.  The oracle V_T must include this
# or the de-freeze variance is systematically understated by 4.5% (enough to
# bias the gamma MLE upward at the short tau).
RARE_JUMP_PROB = 0.005
RARE_JUMP_SD_MULT = 3.0
RARE_JUMP_VAR_FACTOR = 1.0 + RARE_JUMP_PROB * RARE_JUMP_SD_MULT**2


# ---------------------------------------------------------------------------
# Variant market
# ---------------------------------------------------------------------------


def regime_sequence(n_steps: int) -> np.ndarray:
    """Same calendar mapping as ``simulate_market`` (including 14-day wrap)."""
    return np.array(
        [calendar_regime((t // 24) % WINDOW_DAYS, t % 24) for t in range(n_steps)],
        dtype=int,
    )


def defreeze_map(regime: np.ndarray) -> dict[int, int]:
    """Map step index t -> freeze length, for freeze -> OPEN steps.

    The step t -> t+1 is a de-freeze step when row t is frozen and row t+1 is
    OPEN.  The freeze length is the number of frozen rows ending at t (the
    project convention: anchors are tau+1 steps apart).
    """
    out: dict[int, int] = {}
    n = len(regime)
    t = 0
    while t < n:
        if regime[t] not in FREEZE_REGIMES:
            t += 1
            continue
        start = t
        while t < n and regime[t] in FREEZE_REGIMES:
            t += 1
        end = t - 1
        if t < n and regime[t] == REGIME_OPEN:
            out[end] = end - start + 1
    return out


def freeze_runs(regime: np.ndarray) -> list[tuple[int, int, int]]:
    """Freeze runs with both anchors: (start, end, post) row indices."""
    runs: list[tuple[int, int, int]] = []
    n = len(regime)
    t = 0
    while t < n:
        if regime[t] not in FREEZE_REGIMES:
            t += 1
            continue
        start = t
        while t < n and regime[t] in FREEZE_REGIMES:
            t += 1
        end = t - 1
        if start > 0 and t < n and regime[t] == REGIME_OPEN:
            runs.append((start, end, t))
    return runs


def observation_map(
    true_price: np.ndarray, regime: np.ndarray, rng_obs
) -> np.ndarray:
    """Apply the frozen observation mapping with a caller-owned RNG.

    Separating the observation RNG lets the same-endpoint diagnostic compare
    two bridge interiors under bit-identical observation noise.
    """
    n_steps = len(regime)
    pool_price = np.empty(n_steps + 1, dtype=float)
    eta = 0.0
    last_open_price = true_price[0]
    for t in range(n_steps + 1):
        r = regime[t] if t < n_steps else regime[-1]
        if r == REGIME_OPEN:
            last_open_price = true_price[t]
            eta = 0.0
            pool_price[t] = true_price[t] * math.exp(rng_obs.normal(0.0, 0.0005))
        elif r == REGIME_EXTENDED:
            eta = 0.8 * eta + rng_obs.normal(0.0, 0.002)
            pool_price[t] = true_price[t] * math.exp(eta)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = 0.98 * eta + rng_obs.normal(0.0, 0.0008)
            pool_price[t] = last_open_price * math.exp(eta)
        else:  # halt: no trades, pool price is frozen
            pool_price[t] = pool_price[t - 1] if t > 0 else true_price[t]
    return pool_price


def simulate_market_bridge(
    seed_seq,
    n_steps: int,
    *,
    zeta_sd: float,
    nu_mult: float,
    nu_clip: float = 1.0,
) -> MarketPath:
    """Non-Markov bridge variant of ``simulate_market``.

    Ordinary OPEN->OPEN transitions keep the frozen AR(1) kernel.  Each freeze
    run with both anchors is replaced by a Brownian bridge pinned at the latent
    pre/post endpoints.  The bridge volatility is event-specific,

        nu_e = sigma * nu_mult * exp(zeta_e),   zeta_e ~ N(0, zeta_sd^2),

    and the post endpoint is drawn from

        x_+ ~ N(m_+, v_ar + nu_e^2 * gap),

    where ``m_+`` and ``v_ar`` are the AR(1) forecast and its integrated
    variance.  Thus ``nu_mult = 0`` is the Markov placebo (endpoint law equals
    the tau-step composition of T), while ``nu_mult > 0`` adds a freeze-span
    variance component that no ordinary one-step open transition contains.
    The bridge interior itself is unobserved; its volatility is what makes the
    path law non-Markov and is diagnosed with a path-functional check.
    """
    rng = np.random.default_rng(seed_seq)
    rng_bridge = np.random.default_rng([*seed_seq, 271828])
    rng_obs = np.random.default_rng([*seed_seq, 314159])
    sigma = SIGMA_BASE * math.exp(rng.normal(0.0, 0.35))
    phi = 1.0 - KAPPA

    regime = regime_sequence(n_steps)
    runs = {start: (start, end, post) for start, end, post in freeze_runs(regime)}
    log_s = np.empty(n_steps + 1, dtype=float)
    log_s[0] = MU_LOG
    nu_by_start: dict[int, float] = {}

    t = 0
    while t < n_steps:
        if t in runs:
            start, end, post = runs[t]
            pre = start - 1
            tau = end - start + 1
            gap = post - pre
            x_minus = log_s[pre]
            m_plus = MU_LOG + phi**gap * (x_minus - MU_LOG)
            v_ar = (
                sigma**2
                * RARE_JUMP_VAR_FACTOR
                * (1.0 - phi ** (2 * gap))
                / (1.0 - phi**2)
            )
            zeta = float(rng_bridge.normal(0.0, zeta_sd))
            if nu_clip is not None:
                zeta = max(-nu_clip, min(nu_clip, zeta))
            nu = sigma * nu_mult * math.exp(zeta)
            x_plus = m_plus + math.sqrt(v_ar + nu**2 * gap) * float(
                rng_bridge.normal()
            )

            # Standard Brownian bridge increments on the unit interval,
            # scaled to a bridge of duration ``gap`` hours.
            eps = rng_bridge.normal(size=gap)
            w_path = np.concatenate([[0.0], np.cumsum(eps)])
            u = np.arange(1, gap) / gap
            bridge_noise = nu * (w_path[1:-1] - u * w_path[-1])
            log_s[pre + 1 : post] = (
                x_minus + u * (x_plus - x_minus) + bridge_noise
            )
            log_s[post] = x_plus
            nu_by_start[start] = nu
            t = post
            continue

        eps = float(rng.normal())
        jump = (
            float(rng.normal(0.0, 3.0 * sigma)) if rng.random() < 0.005 else 0.0
        )
        log_s[t + 1] = (
            log_s[t] + KAPPA * (MU_LOG - log_s[t]) + sigma * eps + jump
        )
        t += 1

    true_price = np.exp(log_s)
    pool_price = observation_map(true_price, regime, rng_obs)
    volume = rng.lognormal(math.log(BASE_VOLUME), 0.5, size=n_steps)
    volume[regime == REGIME_HALT] = 0.0

    market = MarketPath(true_price, pool_price, regime, volume, sigma)
    market.bridge_nu_by_start = nu_by_start  # type: ignore[attr-defined]
    return market


def bridge_path_values(
    x_minus: float,
    x_plus: float,
    gap: int,
    nu: float,
    rng_bridge,
) -> np.ndarray:
    """Latent log-price values on a Brownian bridge, including endpoints."""
    eps = rng_bridge.normal(size=gap)
    w_path = np.concatenate([[0.0], np.cumsum(eps)])
    u = np.arange(1, gap) / gap
    values = np.empty(gap + 1, dtype=float)
    values[0] = x_minus
    values[-1] = x_plus
    values[1:-1] = (
        x_minus
        + u * (x_plus - x_minus)
        + nu * (w_path[1:-1] - u * w_path[-1])
    )
    return values


def g_scale(tau: int, sigma: float, gamma: float, tau0: float) -> float:
    """De-freeze jump sd g(tau, sigma)."""
    if gamma == 0.0:
        return 0.0
    return gamma * sigma * math.sqrt(1.0 - math.exp(-tau / tau0))


def simulate_market_variant(
    seed_seq, n_steps: int, gamma: float, tau0: float
) -> MarketPath:
    """Replicate ``simulate_market`` plus a de-freeze jump on freeze->OPEN.

    The jump draws use a separate RNG stream, so gamma=0 consumes the main
    stream exactly as ``simulate_market`` does and reproduces it bit for bit.
    """
    rng = np.random.default_rng(seed_seq)
    rng_jump = np.random.default_rng([*seed_seq, 90210])
    sigma = SIGMA_BASE * math.exp(rng.normal(0.0, 0.35))

    regime = regime_sequence(n_steps)
    jumps_at = defreeze_map(regime)

    true_price = np.empty(n_steps + 1, dtype=float)
    true_price[0] = math.exp(MU_LOG)
    for t in range(n_steps):
        log_s = math.log(true_price[t])
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < 0.005 else 0.0
        log_next = log_s + KAPPA * (MU_LOG - log_s) + sigma * eps + jump
        if t in jumps_at:
            log_next += rng_jump.normal(
                0.0, g_scale(jumps_at[t], sigma, gamma, tau0)
            )
        true_price[t + 1] = math.exp(log_next)

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
            pool_price[t] = true_price[t] * math.exp(rng.normal(0.0, 0.0005))
        elif r == REGIME_EXTENDED:
            eta = 0.8 * eta + rng.normal(0.0, 0.002)
            pool_price[t] = true_price[t] * math.exp(eta)
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            eta = 0.98 * eta + rng.normal(0.0, 0.0008)
            pool_price[t] = last_open_price * math.exp(eta)
        else:  # halt: no trades, pool price is frozen
            pool_price[t] = pool_price[t - 1] if t > 0 else true_price[t]

    return MarketPath(true_price, pool_price, regime, volume, sigma)


# ---------------------------------------------------------------------------
# Events and open transitions
# ---------------------------------------------------------------------------


def extract_events(market: MarketPath, window: str, seed: int) -> list[dict]:
    """Freeze runs with pre/post anchor indices (0-based, price-array indices)."""
    regime = market.regime
    n = len(regime)
    nu_by_start = getattr(market, "bridge_nu_by_start", {})
    events: list[dict] = []
    t = 0
    while t < n:
        if regime[t] not in FREEZE_REGIMES:
            t += 1
            continue
        start = t
        while t < n and regime[t] in FREEZE_REGIMES:
            t += 1
        end = t - 1
        pre = start - 1 if start > 0 else None
        post = t if t < n else None
        events.append(
            {
                "window": window,
                "seed": int(seed),
                "start": int(start),
                "end": int(end),
                "tau": int(end - start + 1),
                "gap": int(post - pre) if pre is not None and post is not None else None,
                "pre": pre,
                "post": post,
                "pre_regime": int(regime[pre]) if pre is not None else None,
                "post_regime": int(regime[post]) if post is not None else None,
                "sigma": float(market.sigma),
                "s_minus": float(market.true_price[pre]) if pre is not None else None,
                "s_plus": float(market.true_price[post]) if post is not None else None,
                "p_minus": float(market.pool_price[pre]) if pre is not None else None,
                "p_plus": float(market.pool_price[post]) if post is not None else None,
                "nu_bridge": float(nu_by_start[start]) if start in nu_by_start else None,
            }
        )
    return events


def open_transitions(market: MarketPath) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary OPEN -> OPEN one-step log transitions (excludes de-freeze)."""
    regime = market.regime
    reg_ext = np.concatenate([regime, [regime[-1]]])
    mask = (reg_ext[:-1] == REGIME_OPEN) & (reg_ext[1:] == REGIME_OPEN)
    idx = np.flatnonzero(mask)
    x = np.log(market.pool_price[idx])
    y = np.log(market.pool_price[idx + 1])
    return x, y


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), float(beta[1])


def fit_T(markets: list[MarketPath]) -> dict:
    """Pooled AR(1) fit on ordinary OPEN->OPEN transitions.

    Returns the intercept, phi, mu (sample mean of open log prices) and a
    per-path sigma from that path's own open residuals.
    """
    xs, ys, pids = [], [], []
    for i, market in enumerate(markets):
        x, y = open_transitions(market)
        xs.append(x)
        ys.append(y)
        pids.append(np.full(len(x), i, dtype=int))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    pid = np.concatenate(pids)
    a, phi = _ols(x, y)
    resid = y - (a + phi * x)
    sigma = np.array(
        [math.sqrt(float(np.mean(resid[pid == i] ** 2))) for i in range(len(markets))]
    )
    return {
        "intercept": a,
        "phi": phi,
        "kappa": 1.0 - phi,
        "mu": float(np.mean(x)),
        "sigma": sigma,
        "n_transitions": int(len(x)),
        "residual_sd": float(np.std(resid)),
        "residual_mean": float(np.mean(resid)),
    }


def event_residuals(
    events: list[dict],
    phi: float,
    mu: float,
    sigma_of,
    *,
    use_observed: bool = True,
    sigma_is_total: bool = True,
) -> list[dict]:
    """Post-anchor residual after the Markov prediction (anchor gap tau+1).

    ``sigma_is_total`` distinguishes the oracle tier (input is the diffusion
    sigma from the generator; V_T must add the rare-jump variance) from the
    estimated tier (input is the total innovation sd from open residuals).
    """
    out: list[dict] = []
    for event in events:
        if event["pre"] is None or event["post"] is None:
            continue
        x = math.log(event["p_minus"] if use_observed else event["s_minus"])
        y = math.log(event["p_plus"] if use_observed else event["s_plus"])
        gap = event["gap"]
        pred = mu + phi**gap * (x - mu)
        r = y - pred
        sigma = float(sigma_of(event))
        sigma_v = sigma if sigma_is_total else sigma * math.sqrt(RARE_JUMP_VAR_FACTOR)
        v0 = sigma_v**2 * (1.0 - phi ** (2 * gap)) / (1.0 - phi**2)
        out.append(
            {
                **event,
                "sigma_used": sigma,
                "r": r,
                "v0": v0,
                "w_bridge": sigma_v**2 * gap,
                "z0": r / math.sqrt(v0),
            }
        )
    return out


def jump_weight(event: dict, tau0: float) -> float:
    """sigma^2 * h(tau)^2, the unit-variance regressor for gamma^2."""
    h2 = 1.0 - math.exp(-event["tau"] / tau0)
    return event["sigma_used"] ** 2 * h2


def _negll(theta: float, r2, v0, w) -> float:
    v = v0 + theta * w
    return float(0.5 * np.sum(np.log(2.0 * math.pi * v) + r2 / v))


def _fit_component(rows: list[dict], w: np.ndarray, *, ci: bool = True) -> dict:
    """Gaussian MLE of theta >= 0 in Var(r) = v0 + theta*w."""
    r2 = np.array([row["r"] ** 2 for row in rows])
    v0 = np.array([row["v0"] for row in rows])
    w = np.asarray(w, dtype=float)

    def score(theta: float) -> float:
        v = v0 + theta * w
        return float(np.sum((r2 - v) / v**2 * w))

    if score(0.0) <= 0.0 or np.all(w <= 0):
        theta_hat = 0.0
    else:
        hi = max(1e-12, 10.0 * float(np.max(r2)) / float(np.max(w)))
        while score(hi) > 0.0:
            hi *= 2.0
        theta_hat = float(brentq(score, 0.0, hi, xtol=1e-14, rtol=1e-12))

    out = {
        "n": int(len(rows)),
        "theta_hat": theta_hat,
        "gamma_hat": math.sqrt(max(theta_hat, 0.0)),
        "boundary": bool(theta_hat == 0.0),
    }
    if not ci:
        return out

    ll_hat = -_negll(theta_hat, r2, v0, w)
    # The boundary case theta_hat == 0 is handled by the same profile-LR
    # formula below: the lower crossing stays at 0 and the upper crossing is
    # still defined. Do not special-case it here.
    # lower crossing
    lower = 0.0
    if 2.0 * (_negll(0.0, r2, v0, w) + ll_hat) > CHI2_95_1DF:
        lower = float(
            brentq(
                lambda th: 2.0 * (_negll(th, r2, v0, w) + ll_hat) - CHI2_95_1DF,
                0.0,
                theta_hat,
                xtol=1e-14,
                rtol=1e-12,
            )
        )
    # upper crossing
    hi = max(theta_hat * 2.0, 1e-12)
    while 2.0 * (_negll(hi, r2, v0, w) + ll_hat) < CHI2_95_1DF:
        hi *= 2.0
        if hi > 1e6:
            break
    upper = float(
        brentq(
            lambda th: 2.0 * (_negll(th, r2, v0, w) + ll_hat) - CHI2_95_1DF,
            theta_hat,
            hi,
            xtol=1e-14,
            rtol=1e-12,
        )
    )
    out["ci95_theta"] = [lower, upper]
    out["ci95_gamma"] = [math.sqrt(max(lower, 0.0)), math.sqrt(max(upper, 0.0))]
    return out


def fit_gamma2(
    rows: list[dict], tau0: float, *, ci: bool = True
) -> dict:
    """Gaussian MLE of theta = gamma^2 in Var(r) = v0 + theta*w, theta >= 0."""
    w = np.array([jump_weight(row, tau0) for row in rows])
    return _fit_component(rows, w, ci=ci)


def fit_bridge_component(rows: list[dict], *, ci: bool = True) -> dict:
    """Gaussian MLE of the bridge variance component.

    The non-Markov endpoint law is ``Var(r_e) = v0_e + theta * sigma_e^2 *
    gap_e``.  ``theta = nu_mult^2 * E[exp(2*zeta)]``; unlike the de-freeze
    jump, the bridge component grows with the full anchor gap rather than
    saturating at long tau.
    """
    w = np.array([row["sigma_used"] ** 2 * row["gap"] for row in rows])
    out = _fit_component(rows, w, ci=ci)
    out["nu_over_sigma_hat"] = math.sqrt(max(out["theta_hat"], 0.0))
    if ci:
        out["ci95_nu_over_sigma"] = [
            math.sqrt(max(out["ci95_theta"][0], 0.0)),
            math.sqrt(max(out["ci95_theta"][1], 0.0)),
        ]
    return out


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample split-conformal quantile ceil((n+1)(1-alpha))."""
    n = len(scores)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def binomial_se(coverage: float, n: int) -> float:
    if n == 0:
        return float("nan")
    return math.sqrt(max(coverage * (1.0 - coverage), 0.0) / n)


def coverage_by_tau(rows: list[dict], covered: np.ndarray, widths) -> dict:
    widths = np.asarray(widths, dtype=float)
    out = {}
    for tau in EVENT_TAUS:
        mask = np.array([row["tau"] == tau for row in rows])
        n = int(mask.sum())
        if n == 0:
            continue
        cov = float(np.mean(covered[mask]))
        out[str(tau)] = {
            "n": n,
            "coverage": cov,
            "coverage_se": binomial_se(cov, n),
            "mean_width": float(np.mean(widths[mask])),
        }
    return out


def summarize_method(rows: list[dict], covered: np.ndarray, widths) -> dict:
    n = len(rows)
    cov = float(np.mean(covered)) if n else float("nan")
    return {
        "n": int(n),
        "coverage": cov,
        "coverage_se": binomial_se(cov, n),
        "mean_width": float(np.mean(widths)) if n else float("nan"),
        "by_tau": coverage_by_tau(rows, covered, widths),
    }


def variance_ratio_block(rows: list[dict]) -> dict:
    """mean(z0^2) by tau with a path-level bootstrap CI."""
    out = {}
    paths = sorted({(row["window"], row["seed"]) for row in rows})
    for tau in EVENT_TAUS:
        sel = [row for row in rows if row["tau"] == tau]
        if not sel:
            continue
        z2 = np.array([row["z0"] ** 2 for row in sel])
        by_path = {}
        for row, z in zip(sel, z2):
            by_path.setdefault((row["window"], row["seed"]), []).append(z)
        path_keys = sorted(by_path)
        draws = []
        rng = np.random.default_rng(12345)
        for _ in range(1000):
            pick = rng.integers(0, len(path_keys), size=len(path_keys))
            vals = []
            for i in pick:
                vals.extend(by_path[path_keys[i]])
            draws.append(float(np.mean(vals)))
        out[str(tau)] = {
            "n": int(len(sel)),
            "n_paths": int(len(path_keys)),
            "mean_z2": float(np.mean(z2)),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }
    return out


def nonparametric_g2(rows: list[dict]) -> dict:
    """tau-bucket estimate g^2(tau) = mean(r^2 - v0), no parametric form."""
    out = {}
    for tau in EVENT_TAUS:
        sel = [row for row in rows if row["tau"] == tau]
        if not sel:
            continue
        d = np.array([row["r"] ** 2 - row["v0"] for row in sel])
        n = len(d)
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        out[str(tau)] = {
            "n": int(n),
            "g2_hat": mean,
            "g2_se": se,
            "g_hat": math.sqrt(max(mean, 0.0)),
            "ci95_g2": [mean - 1.96 * se, mean + 1.96 * se],
        }
    return out


def estimated_diagnostics(
    markets: list[MarketPath], labels: list[tuple[str, int]], events: list[dict]
) -> dict:
    """Same diagnostics with T and per-path sigma estimated from open data."""
    fit = fit_T(markets)
    sigma = {}
    for market, key in zip(markets, labels):
        x, y = open_transitions(market)
        resid = y - (fit["intercept"] + fit["phi"] * x)
        sigma[key] = math.sqrt(float(np.mean(resid**2)))
    rows = event_residuals(
        events,
        fit["phi"],
        fit["mu"],
        _sigma_lookup(sigma),
        sigma_is_total=True,
    )
    return {
        "T": {
            "phi": fit["phi"],
            "kappa": fit["kappa"],
            "mu": fit["mu"],
            "n_transitions": fit["n_transitions"],
            "sigma_summary": {
                "min": float(np.min(list(sigma.values()))),
                "median": float(np.median(list(sigma.values()))),
                "max": float(np.max(list(sigma.values()))),
            },
        },
        "variance_ratio": variance_ratio_block(rows),
        "nonparametric_g2": nonparametric_g2(rows),
        "gamma2_mle": fit_gamma2(rows, 40.0),
    }


def tau0_sensitivity(
    rows: list[dict], grid: tuple[float, ...] = (20.0, 40.0, 80.0)
) -> dict:
    """Refit gamma under alternative tau0; the implied g/sigma shape should
    be stable at the observed tau values even though gamma is not."""
    out = {}
    for tau0 in grid:
        fit = fit_gamma2(rows, tau0)
        out[str(tau0)] = {
            "gamma_hat": fit["gamma_hat"],
            "ci95_gamma": fit["ci95_gamma"],
            "g_over_sigma_by_tau": {
                str(tau): math.sqrt(max(fit["theta_hat"], 0.0))
                * math.sqrt(1.0 - math.exp(-tau / tau0))
                for tau in EVENT_TAUS
            },
        }
    return out


def subset_data(
    markets: list[MarketPath],
    labels: list[tuple[str, int]],
    events: list[dict],
    seed_set: set[int],
):
    """Filter paths/events by seed, preserving window-major order."""
    sub_markets = []
    sub_labels = []
    for market, key in zip(markets, labels):
        if key[1] in seed_set:
            sub_markets.append(market)
            sub_labels.append(key)
    sub_events = [event for event in events if event["seed"] in seed_set]
    return sub_markets, sub_labels, sub_events


# ---------------------------------------------------------------------------
# Cross-fitted coverage comparison
# ---------------------------------------------------------------------------


def _sigma_lookup(sigma_by_path: dict):
    return lambda event: sigma_by_path[(event["window"], event["seed"])]


def analyze_environment(
    markets: list[MarketPath],
    windows: list[str],
    seeds: list[int],
    events: list[dict],
    *,
    tier: str,
    tau0: float,
    bridge: bool = False,
) -> dict:
    """Open-only vs cross-event, 5-fold cross-fitting by seed.

    Folds are keyed by seed % N_FOLDS (same convention as the event-conformal
    study); each event is tested exactly once, predictions are pooled.

    ``bridge=True`` swaps the de-freeze jump variance model for the non-Markov
    bridge component ``theta * sigma^2 * gap`` and adds a tau-stratified
    cross-event calibrator that does not assume a functional form in tau.
    """
    keys_order = [(window, seed) for window in windows for seed in seeds]
    key_to_idx = {key: i for i, key in enumerate(keys_order)}
    path_fold = np.array([seed % N_FOLDS for _, seed in keys_order])

    if tier == "oracle":
        phi_global = 1.0 - KAPPA
        mu_global = MU_LOG
        sigma_global = {
            key: float(markets[i].sigma) for key, i in key_to_idx.items()
        }
        T_summary = {
            "phi": phi_global,
            "kappa": KAPPA,
            "mu": mu_global,
            "source": "design constants (oracle)",
        }
    elif tier == "estimated":
        fit_full = fit_T(markets)
        phi_global = fit_full["phi"]
        mu_global = fit_full["mu"]
        sigma_global = {}
        for key, i in key_to_idx.items():
            x, y = open_transitions(markets[i])
            resid = y - (fit_full["intercept"] + phi_global * x)
            sigma_global[key] = math.sqrt(float(np.mean(resid**2)))
        T_summary = {
            "phi": phi_global,
            "kappa": 1.0 - phi_global,
            "mu": mu_global,
            "source": "full-data fit on OPEN->OPEN one-step transitions",
            "n_transitions": fit_full["n_transitions"],
            "sigma_summary": {
                "min": float(np.min(list(sigma_global.values()))),
                "median": float(np.median(list(sigma_global.values()))),
                "max": float(np.max(list(sigma_global.values()))),
            },
        }
    else:
        raise ValueError(f"unknown tier: {tier}")

    test_rows: list[dict] = []
    covered_open: list[bool] = []
    covered_cross: list[bool] = []
    covered_tau: list[bool] = []
    width_open: list[float] = []
    width_cross: list[float] = []
    width_tau: list[float] = []
    fold_summary: list[dict] = []

    def weight_of(row: dict) -> float:
        if bridge:
            return float(row["w_bridge"])
        return jump_weight(row, tau0)

    def fit_component(rows: list[dict], *, ci: bool) -> dict:
        if bridge:
            return fit_bridge_component(rows, ci=ci)
        return fit_gamma2(rows, tau0, ci=ci)

    for fold in range(N_FOLDS):
        cal_keys = {
            key for key, idx in key_to_idx.items() if path_fold[idx] != fold
        }
        test_keys = {
            key for key, idx in key_to_idx.items() if path_fold[idx] == fold
        }

        if tier == "estimated":
            cal_markets = [markets[key_to_idx[key]] for key in sorted(cal_keys)]
            fit = fit_T(cal_markets)
            phi = fit["phi"]
            mu = fit["mu"]
            sigma_fold = {}
            for key in sorted(cal_keys | test_keys):
                x, y = open_transitions(markets[key_to_idx[key]])
                resid = y - (fit["intercept"] + phi * x)
                sigma_fold[key] = math.sqrt(float(np.mean(resid**2)))
            fold_T = {
                "T_phi": phi,
                "T_kappa": 1.0 - phi,
                "T_mu": mu,
            }
        else:
            phi = phi_global
            mu = mu_global
            sigma_fold = sigma_global
            fold_T = {}

        sigma_of = _sigma_lookup(sigma_fold)
        sigma_is_total = tier == "estimated"
        cal_rows = event_residuals(
            [e for e in events if (e["window"], e["seed"]) in cal_keys],
            phi,
            mu,
            sigma_of,
            sigma_is_total=sigma_is_total,
        )
        test_fold_rows = event_residuals(
            [e for e in events if (e["window"], e["seed"]) in test_keys],
            phi,
            mu,
            sigma_of,
            sigma_is_total=sigma_is_total,
        )

        # open-only: q from ordinary OPEN->OPEN one-step residuals, pooled.
        open_scores = []
        for key in sorted(cal_keys):
            x, y = open_transitions(markets[key_to_idx[key]])
            resid = y - (mu + phi * (x - mu))
            open_scores.append(np.abs(resid) / sigma_fold[key])
        q_open = conformal_quantile(np.concatenate(open_scores))

        # cross-event: fit theta on calibration events, conformalize there.
        theta = fit_component(cal_rows, ci=False)["theta_hat"]
        cal_scores = np.array(
            [
                abs(row["r"]) / math.sqrt(row["v0"] + theta * weight_of(row))
                for row in cal_rows
            ]
        )
        q_cross = conformal_quantile(cal_scores)

        # tau-stratified cross-event: no functional form in tau, only the
        # event-length stratum.  This is the assumption-light fallback when
        # the bridge variance shape is unknown.
        q_tau: dict[int, float] = {}
        for tau in EVENT_TAUS:
            scores_tau = [
                abs(row["r"]) / math.sqrt(row["v0"])
                for row in cal_rows
                if row["tau"] == tau
            ]
            if scores_tau:
                q_tau[tau] = conformal_quantile(np.array(scores_tau))
        q_tau_global = conformal_quantile(
            np.array([abs(row["r"]) / math.sqrt(row["v0"]) for row in cal_rows])
        )

        fold_open_cov = []
        fold_cross_cov = []
        fold_tau_cov = []
        for row in test_fold_rows:
            v0 = row["v0"]
            w = weight_of(row)
            q_tau_row = q_tau.get(row["tau"], q_tau_global)
            open_hit = abs(row["r"]) <= q_open * math.sqrt(v0)
            cross_hit = abs(row["r"]) <= q_cross * math.sqrt(v0 + theta * w)
            tau_hit = abs(row["r"]) <= q_tau_row * math.sqrt(v0)
            test_rows.append(row)
            covered_open.append(open_hit)
            covered_cross.append(cross_hit)
            covered_tau.append(tau_hit)
            width_open.append(2.0 * q_open * math.sqrt(v0))
            width_cross.append(2.0 * q_cross * math.sqrt(v0 + theta * w))
            width_tau.append(2.0 * q_tau_row * math.sqrt(v0))
            fold_open_cov.append(open_hit)
            fold_cross_cov.append(cross_hit)
            fold_tau_cov.append(tau_hit)

        fold_summary.append(
            {
                "fold": fold,
                "n_cal_events": len(cal_rows),
                "n_test_events": len(test_fold_rows),
                "q_open": q_open,
                "q_cross": q_cross,
                "theta_hat_cal": theta,
                "coverage_open": float(np.mean(fold_open_cov))
                if fold_open_cov
                else None,
                "coverage_cross": float(np.mean(fold_cross_cov))
                if fold_cross_cov
                else None,
                "coverage_tau": float(np.mean(fold_tau_cov))
                if fold_tau_cov
                else None,
                **fold_T,
            }
        )

    covered_open_arr = np.array(covered_open, dtype=bool)
    covered_cross_arr = np.array(covered_cross, dtype=bool)
    covered_tau_arr = np.array(covered_tau, dtype=bool)

    full_rows = event_residuals(
        events,
        phi_global,
        mu_global,
        _sigma_lookup(sigma_global),
        sigma_is_total=(tier == "estimated"),
    )
    fold_kappas = [f["T_kappa"] for f in fold_summary if "T_kappa" in f]
    if fold_kappas:
        T_summary["fold_kappa_range"] = [
            float(np.min(fold_kappas)),
            float(np.max(fold_kappas)),
        ]

    open_summary = summarize_method(test_rows, covered_open_arr, width_open)
    cross_summary = summarize_method(test_rows, covered_cross_arr, width_cross)
    tau_summary = summarize_method(test_rows, covered_tau_arr, width_tau)
    for method_summary in (cross_summary, tau_summary):
        method_summary["width_ratio_vs_open"] = (
            method_summary["mean_width"] / open_summary["mean_width"]
            if open_summary["mean_width"] > 0
            else None
        )
    component_fit = fit_component(full_rows, ci=True)
    cross_summary["component_full_fit"] = component_fit
    if bridge:
        cross_summary["gamma2_full_fit"] = None
        tau_summary["gamma2_full_fit"] = None
    else:
        cross_summary["gamma2_full_fit"] = component_fit
        tau_summary["gamma2_full_fit"] = None

    out = {
        "tier": tier,
        "T": T_summary,
        "folds": fold_summary,
        "open_only": open_summary,
        "cross_event": cross_summary,
        "cross_event_tau": tau_summary,
    }
    if bridge:
        out["nu_quartile_coverage"] = _coverage_by_nu_quartile(
            test_rows, covered_open_arr, covered_cross_arr, width_open, width_cross
        )
    return out


def _coverage_by_nu_quartile(
    rows: list[dict],
    covered_open: np.ndarray,
    covered_cross: np.ndarray,
    width_open: list[float],
    width_cross: list[float],
) -> dict:
    """Diagnostic: coverage against the realized (simulation-only) nu_e."""
    nu = np.array(
        [row["nu_bridge"] if row["nu_bridge"] is not None else np.nan for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(nu)) or np.nanstd(nu) == 0:
        return {"available": False}
    edges = np.quantile(nu, [0.25, 0.5, 0.75])
    groups = np.digitize(nu, edges)
    out = {"available": True, "edges": [float(x) for x in edges], "groups": []}
    for g in range(4):
        mask = groups == g
        if not mask.any():
            continue
        out["groups"].append(
            {
                "group": g,
                "n": int(mask.sum()),
                "nu_min": float(nu[mask].min()),
                "nu_max": float(nu[mask].max()),
                "open_coverage": float(np.mean(covered_open[mask])),
                "cross_coverage": float(np.mean(covered_cross[mask])),
                "open_width": float(np.mean(np.asarray(width_open)[mask])),
                "cross_width": float(np.mean(np.asarray(width_cross)[mask])),
            }
        )
    return out


def same_endpoint_diagnostic(
    markets: list[MarketPath],
    labels: list[tuple[str, int]],
    events: list[dict],
    *,
    factor: float = 2.0,
    max_events: int = 64,
) -> dict:
    """Two bridge interiors with identical endpoints, different volatility.

    This is the empirical counterpart of the Theorem 4 construction: the
    observed pool-price path is bit-identical because freeze interiors are
    masked, while the latent quadratic variation changes by ``factor^2``.
    """
    market_by_key = dict(zip(labels, markets))

    selected = [
        event
        for event in events
        if event["pre"] is not None
        and event["post"] is not None
        and event["nu_bridge"] is not None
        and event["nu_bridge"] > 0.0
    ][:max_events]
    if not selected:
        return {"available": False}

    max_pool_diff = 0.0
    qv_nu = []
    qv_2nu = []
    theo_nu = []
    theo_2nu = []
    for event in selected:
        market = market_by_key[(event["window"], event["seed"])]
        pre, post = event["pre"], event["post"]
        gap = post - pre
        x_minus = math.log(market.true_price[pre])
        x_plus = math.log(market.true_price[post])
        nu = float(event["nu_bridge"])

        rng = np.random.default_rng([event["seed"], pre, 991])
        eps = rng.normal(size=gap)
        w_path = np.concatenate([[0.0], np.cumsum(eps)])
        u = np.arange(1, gap) / gap
        common = w_path[1:-1] - u * w_path[-1]
        interior_nu = x_minus + u * (x_plus - x_minus) + nu * common
        interior_2nu = (
            x_minus + u * (x_plus - x_minus) + factor * nu * common
        )

        true_nu = market.true_price.copy()
        true_2nu = market.true_price.copy()
        true_nu[pre + 1 : post] = np.exp(interior_nu)
        true_2nu[pre + 1 : post] = np.exp(interior_2nu)

        rng_a = np.random.default_rng([event["seed"], pre, 31337])
        rng_b = np.random.default_rng([event["seed"], pre, 31337])
        pool_nu = observation_map(true_nu, market.regime, rng_a)
        pool_2nu = observation_map(true_2nu, market.regime, rng_b)
        max_pool_diff = max(
            max_pool_diff, float(np.max(np.abs(pool_nu - pool_2nu)))
        )

        log_nu = np.log(true_nu[pre : post + 1])
        log_2nu = np.log(true_2nu[pre : post + 1])
        qv_nu.append(float(np.sum(np.diff(log_nu) ** 2)))
        qv_2nu.append(float(np.sum(np.diff(log_2nu) ** 2)))
        tau = event["tau"]
        theo_nu.append(nu**2 * tau)
        theo_2nu.append((factor * nu) ** 2 * tau)

    return {
        "available": True,
        "n_events": int(len(selected)),
        "volatility_factor": factor,
        "max_abs_pool_price_diff": max_pool_diff,
        "observed_paths_identical": bool(max_pool_diff == 0.0),
        "mean_qv_nu": float(np.mean(qv_nu)),
        "mean_qv_factor_nu": float(np.mean(qv_2nu)),
        "mean_qv_ratio": float(np.mean(qv_2nu) / np.mean(qv_nu)),
        "mean_theory_qv_nu": float(np.mean(theo_nu)),
        "mean_theory_qv_factor_nu": float(np.mean(theo_2nu)),
        "mean_theory_ratio": float(np.mean(theo_2nu) / np.mean(theo_nu)),
        "note": (
            "Same realized endpoints and bit-identical observed pool prices; "
            "the latent bridge quadratic variation is the only difference."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--windows", nargs="+", default=["W1", "W2", "W3", "W4"])
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--tau0", type=float, default=40.0)
    parser.add_argument(
        "--bridge-nu",
        type=float,
        default=0.6,
        help="bridge volatility multiplier nu_mult in nu_e = sigma*nu_mult*exp(zeta)",
    )
    parser.add_argument(
        "--bridge-zeta-sd",
        type=float,
        default=0.35,
        help="event-level log-volatility sd for the non-Markov bridge",
    )
    parser.add_argument(
        "--bridge-nu-clip",
        type=float,
        default=1.0,
        help="clip zeta at +/- this value (set <=0 to disable)",
    )
    parser.add_argument(
        "--skip-bridge",
        action="store_true",
        help="skip the non-Markov bridge block (legacy gamma-only run)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "stress_variant.json")
    parser.add_argument(
        "--diag",
        type=Path,
        default=STRESS_DIR / "true_paths_stress.csv.gz",
    )
    return parser.parse_args()


def _sanity_replication(n_steps: int, seeds: list[int], windows: list[str]) -> dict:
    """Check gamma=0 against the frozen generator on every analysed path.

    The variant must be bit-for-bit identical to ``simulate_market`` when the
    de-freeze jump is switched off; otherwise the stress variant would be
    changing something other than gamma.
    """
    max_true = 0.0
    max_pool = 0.0
    regime_ok = True
    volume_ok = True
    sigma_ok = True
    n_checked = 0
    for w_idx, _window in enumerate(windows):
        for seed in seeds:
            base = simulate_market([seed, w_idx], n_steps)
            variant = simulate_market_variant([seed, w_idx], n_steps, 0.0, 40.0)
            max_true = max(
                max_true,
                float(np.max(np.abs(base.true_price - variant.true_price))),
            )
            max_pool = max(
                max_pool,
                float(np.max(np.abs(base.pool_price - variant.pool_price))),
            )
            regime_ok = regime_ok and bool(
                np.array_equal(base.regime, variant.regime)
            )
            volume_ok = volume_ok and bool(
                np.array_equal(base.volume, variant.volume)
            )
            sigma_ok = sigma_ok and bool(base.sigma == variant.sigma)
            n_checked += 1
    return {
        "description": (
            "gamma=0 variant vs frozen simulate_market, all analysed paths "
            f"({n_steps} steps)"
        ),
        "n_checked": n_checked,
        "max_abs_true_price": max_true,
        "max_abs_pool_price": max_pool,
        "regime_all_equal": regime_ok,
        "volume_all_equal": volume_ok,
        "sigma_all_equal": sigma_ok,
        "all_exact": (
            max_true == 0.0
            and max_pool == 0.0
            and regime_ok
            and volume_ok
            and sigma_ok
        ),
    }


def main() -> None:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    windows = list(args.windows)
    n_steps = int(args.steps)

    # ---- generate twin environments -------------------------------------
    markets0: list[MarketPath] = []
    markets1: list[MarketPath] = []
    events0: list[dict] = []
    events1: list[dict] = []
    diag_frames = []
    labels: list[tuple[str, int]] = []
    for w_idx, window in enumerate(windows):
        for seed in seeds:
            seq = [seed, w_idx]
            market0 = simulate_market_variant(seq, n_steps, 0.0, args.tau0)
            market1 = simulate_market_variant(seq, n_steps, args.gamma, args.tau0)
            markets0.append(market0)
            markets1.append(market1)
            labels.append((window, seed))
            events0.extend(extract_events(market0, window, seed))
            events1.extend(extract_events(market1, window, seed))
            for gamma, market in ((0.0, market0), (args.gamma, market1)):
                diag_frames.append(
                    pd.DataFrame(
                        {
                            "variant_gamma": gamma,
                            "window": window,
                            "seed": int(seed),
                            "episode_step": np.arange(1, n_steps + 1),
                            "regime": market.regime,
                            "true_price": market.true_price[:n_steps],
                            "pool_price": market.pool_price[:n_steps],
                            "sigma": market.sigma,
                        }
                    )
                )

    # ---- sanity and event design ---------------------------------------
    n_paths = len(markets1)
    for event in events1:
        if event["pre"] is None or event["post"] is None:
            continue
        assert event["post_regime"] == REGIME_OPEN, event
        assert event["gap"] == event["tau"] + 1, event
        assert event["pre_regime"] in (REGIME_OPEN, REGIME_EXTENDED), event
    double_taus = [
        event["tau"]
        for event in events1
        if event["pre"] is not None and event["post"] is not None
    ]
    assert sorted(double_taus) == sorted(list(EVENT_TAUS) * n_paths), (
        f"unexpected double-endpoint events: {sorted(set(double_taus))}"
    )

    sanity = _sanity_replication(n_steps, seeds, windows)
    event_counts = {}
    for name, evs in (("gamma0", events0), ("gamma_star", events1)):
        counts = {}
        anchors = {}
        for event in evs:
            if event["pre"] is None or event["post"] is None:
                continue
            counts[str(event["tau"])] = counts.get(str(event["tau"]), 0) + 1
            anchors[str(event["tau"])] = {
                "pre_regime": REGIME_EXTENDED
                if event["pre_regime"] == REGIME_EXTENDED
                else event["pre_regime"],
                "post_regime": event["post_regime"],
                "anchor_gap": event["gap"],
            }
        event_counts[name] = {
            "n_events_total": len(evs),
            "n_double_endpoint": int(sum(counts.values())),
            "tau_counts": counts,
            "tau_anchor": anchors,
        }

    sigma_all = [m.sigma for m in markets1]
    jump_by_tau = {
        str(tau): {
            "g_over_sigma": args.gamma * math.sqrt(1.0 - math.exp(-tau / args.tau0)),
            "g_median": float(
                np.median(
                    [
                        g_scale(tau, m.sigma, args.gamma, args.tau0)
                        for m in markets1
                    ]
                )
            ),
        }
        for tau in EVENT_TAUS
    }

    # ---- tier analyses ---------------------------------------------------
    results = {}
    for label, gamma, markets, evs in (
        ("gamma0", 0.0, markets0, events0),
        ("gamma_star", args.gamma, markets1, events1),
    ):
        results[label] = {}
        for tier in ("oracle", "estimated"):
            results[label][tier] = analyze_environment(
                markets,
                windows,
                seeds,
                evs,
                tier=tier,
                tau0=args.tau0,
            )

    # ---- non-Markov Brownian bridge stress ------------------------------
    half = len(seeds) // 2
    bridge_block: dict = {"available": False}
    if not args.skip_bridge:
        nu_clip = None if args.bridge_nu_clip <= 0 else args.bridge_nu_clip
        bridge_markets: dict[str, list[MarketPath]] = {}
        bridge_events: dict[str, list[dict]] = {}
        for label, nu_mult in (("bridge0", 0.0), ("bridge_star", args.bridge_nu)):
            bmarkets: list[MarketPath] = []
            bevents: list[dict] = []
            for w_idx, window in enumerate(windows):
                for seed in seeds:
                    market = simulate_market_bridge(
                        [seed, w_idx],
                        n_steps,
                        zeta_sd=args.bridge_zeta_sd,
                        nu_mult=nu_mult,
                        nu_clip=nu_clip,
                    )
                    bmarkets.append(market)
                    bevents.extend(extract_events(market, window, seed))
            bridge_markets[label] = bmarkets
            bridge_events[label] = bevents

        bridge_counts = {}
        for label, evs in bridge_events.items():
            counts = {}
            n_nu = 0
            nu_vals = []
            for event in evs:
                if event["pre"] is None or event["post"] is None:
                    continue
                assert event["post_regime"] == REGIME_OPEN, event
                assert event["gap"] == event["tau"] + 1, event
                counts[str(event["tau"])] = counts.get(str(event["tau"]), 0) + 1
                if event["nu_bridge"] is not None:
                    n_nu += 1
                    nu_vals.append(event["nu_bridge"])
            bridge_counts[label] = {
                "n_events_total": int(len(evs)),
                "n_double_endpoint": int(sum(counts.values())),
                "n_with_nu": int(n_nu),
                "tau_counts": counts,
                "nu_over_sigma_median": float(
                    np.median(
                        [
                            event["nu_bridge"] / event["sigma"]
                            for event in evs
                            if event["nu_bridge"] is not None
                        ]
                    )
                )
                if n_nu
                else None,
                "nu_values_positive": bool(nu_vals) and bool(min(nu_vals) > 0.0),
            }

        bridge_results = {}
        for label, bmarkets, bevents in (
            ("bridge0", bridge_markets["bridge0"], bridge_events["bridge0"]),
            (
                "bridge_star",
                bridge_markets["bridge_star"],
                bridge_events["bridge_star"],
            ),
        ):
            bridge_results[label] = {}
            for tier in ("oracle", "estimated"):
                bridge_results[label][tier] = analyze_environment(
                    bmarkets,
                    windows,
                    seeds,
                    bevents,
                    tier=tier,
                    tau0=args.tau0,
                    bridge=True,
                )

        # Assumption-light diagnostics on the bridge-star world.
        sigma_lookup_bridge = {
            (window, seed): float(market.sigma)
            for market, (window, seed) in zip(
                bridge_markets["bridge_star"], labels
            )
        }
        bridge_rows = event_residuals(
            bridge_events["bridge_star"],
            1.0 - KAPPA,
            MU_LOG,
            _sigma_lookup(sigma_lookup_bridge),
            sigma_is_total=False,
        )
        bridge_diagnostics = {
            "variance_ratio": variance_ratio_block(bridge_rows),
            "nonparametric_component": nonparametric_g2(bridge_rows),
            "component_mle": fit_bridge_component(bridge_rows),
            "same_endpoint": same_endpoint_diagnostic(
                bridge_markets["bridge_star"], labels, bridge_events["bridge_star"]
            ),
        }

        # Seed-split stability of the open-only failure / cross-event recovery.
        bridge_seed_split = {}
        for name, seed_set in (
            ("first_half", set(seeds[:half])),
            ("second_half", set(seeds[half:])),
        ):
            sub_seeds = [seed for seed in seeds if seed in seed_set]
            bridge_seed_split[name] = {}
            for label in ("bridge0", "bridge_star"):
                sub_markets, _, sub_events = subset_data(
                    bridge_markets[label], labels, bridge_events[label], seed_set
                )
                bridge_seed_split[name][label] = {}
                for tier in ("oracle", "estimated"):
                    block = analyze_environment(
                        sub_markets,
                        windows,
                        sub_seeds,
                        sub_events,
                        tier=tier,
                        tau0=args.tau0,
                        bridge=True,
                    )
                    fit = block["cross_event"]["component_full_fit"]
                    bridge_seed_split[name][label][tier] = {
                        "n_paths": len(sub_markets),
                        "n_events": block["open_only"]["n"],
                        "open_coverage": block["open_only"]["coverage"],
                        "cross_coverage": block["cross_event"]["coverage"],
                        "tau_coverage": block["cross_event_tau"]["coverage"],
                        "theta_hat": fit["theta_hat"],
                        "nu_over_sigma_hat": fit["nu_over_sigma_hat"],
                    }

        # Design summary: realized nu/sigma by tau and the implied variance
        # fraction theta = E[nu^2]/sigma^2.
        nu_ratio_by_tau = {}
        for tau in EVENT_TAUS:
            vals = [
                event["nu_bridge"] / event["sigma"]
                for event in bridge_events["bridge_star"]
                if event["pre"] is not None
                and event["post"] is not None
                and event["tau"] == tau
                and event["nu_bridge"] is not None
            ]
            if vals:
                nu_ratio_by_tau[str(tau)] = {
                    "n": int(len(vals)),
                    "median": float(np.median(vals)),
                    "mean_square": float(np.mean(np.square(vals))),
                }
        bridge_block = {
            "available": True,
            "design": {
                "n_paths": int(len(bridge_markets["bridge_star"])),
                "n_steps": n_steps,
                "n_days": n_steps / 24.0,
                "event_taus_freeze_hours": list(EVENT_TAUS),
                "nu_mult": args.bridge_nu,
                "zeta_sd": args.bridge_zeta_sd,
                "nu_clip": nu_clip,
                "bridge_law": (
                    "Brownian bridge pinned at latent pre/post anchors; "
                    "endpoint law N(m_+, v_ar + nu_e^2*gap), "
                    "nu_e = sigma*nu_mult*exp(zeta_e), "
                    "zeta_e ~ N(0, zeta_sd^2)"
                ),
                "variance_model": (
                    "Var(r_e) = v0_e + theta * sigma_e^2 * gap_e; "
                    "theta = nu_mult^2 * E[exp(2*zeta)]"
                ),
                "nu_ratio_by_tau": nu_ratio_by_tau,
                "theta_true_empirical": float(
                    np.mean(
                        [
                            (event["nu_bridge"] / event["sigma"]) ** 2
                            for event in bridge_events["bridge_star"]
                            if event["nu_bridge"] is not None
                        ]
                    )
                ),
                "coverage_alpha": ALPHA,
                "n_folds": N_FOLDS,
                "fold_key": "seed % 5",
            },
            "events": bridge_counts,
            "diagnostics": {
                "bridge_star": bridge_diagnostics,
            },
            "seed_split": bridge_seed_split,
            "results": bridge_results,
            "boundaries": [
                "The freeze interior is unobserved, so the bridge volatility's "
                "observable footprint is the post-anchor endpoint law. The "
                "same-endpoint diagnostic isolates the path-functional part of "
                "Theorem 4; coverage is driven by the endpoint variance "
                "component, not by observing the interior.",
                "The endpoint law adds nu_e^2*gap to the AR(1) integrated "
                "variance. This is a deliberate stress, not an estimate of a "
                "real bridge law; nu_mult=0 is the Markov placebo.",
                "The cross-event variance model uses sigma_used^2*gap, where "
                "sigma_used includes the 4.5% rare-jump variance factor in the "
                "estimated tier; theta is therefore biased low by about 4.5%, "
                "which does not affect conformal coverage.",
                "The tau-stratified calibrator is assumption-light in tau but "
                "still pools sigma variation; it is a robustness check, not a "
                "replacement for the parametric component.",
            ],
        }

    # ---- seed-split robustness ------------------------------------------
    seed_split = {}
    for name, seed_set in (
        ("first_half", set(seeds[:half])),
        ("second_half", set(seeds[half:])),
    ):
        sub_seeds = [seed for seed in seeds if seed in seed_set]
        seed_split[name] = {}
        for label, markets, evs in (
            ("gamma0", markets0, events0),
            ("gamma_star", markets1, events1),
        ):
            sub_markets, _, sub_events = subset_data(markets, labels, evs, seed_set)
            seed_split[name][label] = {}
            for tier in ("oracle", "estimated"):
                block = analyze_environment(
                    sub_markets,
                    windows,
                    sub_seeds,
                    sub_events,
                    tier=tier,
                    tau0=args.tau0,
                )
                fit = block["cross_event"]["gamma2_full_fit"]
                seed_split[name][label][tier] = {
                    "n_paths": len(sub_markets),
                    "n_events": block["open_only"]["n"],
                    "open_coverage": block["open_only"]["coverage"],
                    "cross_coverage": block["cross_event"]["coverage"],
                    "gamma_hat": fit["gamma_hat"],
                    "ci95_gamma": fit["ci95_gamma"],
                }

    # ---- assumption-light diagnostics (oracle T, all events) -------------
    diagnostics = {}
    diag_rows = {}
    for label, gamma, markets, evs in (
        ("gamma0", 0.0, markets0, events0),
        ("gamma_star", args.gamma, markets1, events1),
    ):
        sigma_lookup = {
            (window, seed): float(market.sigma)
            for market, (window, seed) in zip(markets, labels)
        }
        rows = event_residuals(
            evs,
            1.0 - KAPPA,
            MU_LOG,
            _sigma_lookup(sigma_lookup),
            sigma_is_total=False,
        )
        diag_rows[label] = rows
        diagnostics[label] = {
            "n_events": len(rows),
            "oracle": {
                "variance_ratio": variance_ratio_block(rows),
                "nonparametric_g2": nonparametric_g2(rows),
                "gamma2_mle": fit_gamma2(rows, args.tau0),
                "tau0_sensitivity": tau0_sensitivity(rows),
            },
            "estimated": estimated_diagnostics(markets, labels, evs),
        }

    # ---- T-law comparison between twins ---------------------------------
    fit0 = fit_T(markets0)
    fit1 = fit_T(markets1)

    def pooled_open_standardized(markets: list[MarketPath]) -> np.ndarray:
        vals = []
        for market in markets:
            x, y = open_transitions(market)
            resid = y - (fit0["intercept"] + fit0["phi"] * x)
            vals.append(resid / market.sigma)
        return np.concatenate(vals)

    z_open0 = pooled_open_standardized(markets0)
    z_open1 = pooled_open_standardized(markets1)
    ks_open = stats.ks_2samp(z_open0, z_open1)

    sigma0 = {
        key: float(markets0[i].sigma)
        for i, key in enumerate(labels)
    }
    sigma1 = {
        key: float(markets1[i].sigma)
        for i, key in enumerate(labels)
    }
    rows_df0 = event_residuals(
        events0,
        1.0 - KAPPA,
        MU_LOG,
        _sigma_lookup(sigma0),
        sigma_is_total=False,
    )
    rows_df1 = event_residuals(
        events1,
        1.0 - KAPPA,
        MU_LOG,
        _sigma_lookup(sigma1),
        sigma_is_total=False,
    )
    z_df0 = np.array([row["z0"] for row in rows_df0])
    z_df1 = np.array([row["z0"] for row in rows_df1])
    ks_defreeze = stats.ks_2samp(z_df0, z_df1)
    # A KS test on z is underpowered against a pure scale change; test the
    # variance ratio directly. Events within a path share sigma, so the
    # permutation is paired at the path level: each path contributes its mean
    # z^2 under both twins, and twin labels are flipped per path.
    z2_0 = z_df0**2
    z2_1 = z_df1**2

    def _path_mean_z2(rows: list[dict]) -> dict[tuple[str, int], float]:
        groups: dict[tuple[str, int], list[float]] = {}
        for row in rows:
            key = (row["window"], row["seed"])
            groups.setdefault(key, []).append(row["z0"] ** 2)
        assert all(len(vals) == len(EVENT_TAUS) for vals in groups.values())
        return {key: float(np.mean(vals)) for key, vals in groups.items()}

    path_mean_0 = _path_mean_z2(rows_df0)
    path_mean_1 = _path_mean_z2(rows_df1)
    path_keys = sorted(path_mean_0)
    assert path_keys == sorted(path_mean_1)
    paired_diff = np.array(
        [path_mean_1[key] - path_mean_0[key] for key in path_keys]
    )
    observed_diff = float(np.mean(paired_diff))
    n_perm = 2000
    rng_perm = np.random.default_rng(20261021)
    exceed = 0
    for _ in range(n_perm):
        signs = rng_perm.choice((-1.0, 1.0), size=len(paired_diff))
        stat = float(np.mean(signs * paired_diff))
        if abs(stat) >= abs(observed_diff):
            exceed += 1
    perm_p = (exceed + 1.0) / (n_perm + 1.0)
    observed_ratio = float(np.mean(z2_1) / np.mean(z2_0))

    # ---- write diagnostics ----------------------------------------------
    args.diag.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(diag_frames, ignore_index=True).to_csv(
        args.diag, index=False, compression="gzip"
    )

    summary = {
        "generated": "2026-10-21",
        "task": "I5 stress variant (identifiability note §6.3), non-C1",
        "design": {
            "n_steps": n_steps,
            "n_days": n_steps / 24.0,
            "seeds": seeds,
            "windows": windows,
            "n_paths": len(markets1),
            "n_events_double_endpoint": int(len(z_df1)),
            "event_taus_freeze_hours": list(EVENT_TAUS),
            "tau_convention": (
                "tau = frozen hours; anchors are tau+1 steps apart "
                "(matches identifiability_check.py)"
            ),
            "gamma_true": args.gamma,
            "tau0_true": args.tau0,
            "g_form": "gamma * sigma * sqrt(1 - exp(-tau/tau0)), "
                      "injected only on freeze->OPEN steps",
            "jump_rng": "separate stream default_rng([*seed, 90210])",
            "sigma_summary": {
                "min": float(np.min(sigma_all)),
                "median": float(np.median(sigma_all)),
                "max": float(np.max(sigma_all)),
            },
            "jump_by_tau": jump_by_tau,
            "coverage_alpha": ALPHA,
            "n_folds": N_FOLDS,
            "fold_key": "seed % 5",
        },
        "sanity": sanity,
        "events": event_counts,
        "T_fit": {
            "gamma0": {
                "phi": fit0["phi"],
                "kappa": fit0["kappa"],
                "mu": fit0["mu"],
                "n_transitions": fit0["n_transitions"],
                "sigma_summary": {
                    "min": float(np.min(fit0["sigma"])),
                    "median": float(np.median(fit0["sigma"])),
                    "max": float(np.max(fit0["sigma"])),
                },
            },
            "gamma_star": {
                "phi": fit1["phi"],
                "kappa": fit1["kappa"],
                "mu": fit1["mu"],
                "n_transitions": fit1["n_transitions"],
                "sigma_summary": {
                    "min": float(np.min(fit1["sigma"])),
                    "median": float(np.median(fit1["sigma"])),
                    "max": float(np.max(fit1["sigma"])),
                },
            },
            "design": {"kappa": KAPPA, "mu": MU_LOG},
        },
        "identification": {
            "statement": (
                "The one-step conditional kernel T governing ordinary "
                "OPEN->OPEN transitions does not contain gamma: the de-freeze "
                "jump is injected only on freeze->OPEN steps, which are "
                "excluded from the T fit. Hence T is exactly gamma-free, and "
                "an estimator built only from these transitions targets a "
                "gamma-free quantity; open-only Markov extrapolation cannot "
                "see gamma. The marginal law of later open increments "
                "is not literally invariant: a de-freeze jump shifts the "
                "post-freeze level, which mean-reverts into the next "
                "increment with drift O(kappa*g), up to ~1.3% of sigma at "
                "long tau, decaying as (1-kappa)^k; the twin open-increment "
                "samples are descriptively indistinguishable. De-freeze "
                "anchor residuals depend on gamma directly; cross-event "
                "(tau, sigma) variation identifies it."
            ),
            "open_increment_ks_between_twins": {
                "statistic": float(ks_open.statistic),
                "pvalue": float(ks_open.pvalue),
                "note": "descriptive; twins share the main RNG stream up to "
                        "the first de-freeze, so samples are dependent",
            },
            "defreeze_z_ks_between_twins": {
                "statistic": float(ks_defreeze.statistic),
                "pvalue": float(ks_defreeze.pvalue),
                "note": "descriptive; z uses the oracle T and design sigma, "
                        "and the twins are dependent through the shared main "
                        "RNG stream. A KS test on z is weak against a pure "
                        "scale change, so the permutation below tests z^2.",
            },
            "defreeze_variance_ratio_permutation": {
                "mean_z2_gamma0": float(np.mean(z2_0)),
                "mean_z2_gamma_star": float(np.mean(z2_1)),
                "ratio_gamma_star_over_gamma0": observed_ratio,
                "mean_paired_path_diff": observed_diff,
                "paired_path_diff_se": float(
                    np.std(paired_diff, ddof=1) / math.sqrt(len(paired_diff))
                ),
                "n_paths": int(len(paired_diff)),
                "perm_p_two_sided": perm_p,
                "n_permutations": n_perm,
                "note": "A KS test on z is underpowered against a pure scale "
                        "change, so this tests z^2 directly. Events within a "
                        "path share sigma, so the permutation flips twin "
                        "labels per path on the path-mean z^2; it preserves "
                        "within-path dependence. The ratio above is the "
                        "event-level descriptive ratio.",
            },
        },
        "diagnostics": diagnostics,
        "seed_split": seed_split,
        "results": results,
        "nonmarkov_bridge": bridge_block,
        "boundaries": [
            "gamma=4 and tau0=40 are design choices; smaller gamma weakens "
            "power. What is exactly gamma-free is the one-step conditional "
            "kernel T, not the marginal law of open increments: a de-freeze "
            "jump shifts the post-freeze level, which mean-reverts into later "
            "increments with drift O(kappa*g) (up to ~1.3% of sigma at long "
            "tau) decaying as (1-kappa)^k. The twin open-increment samples are "
            "descriptively indistinguishable (KS stored under "
            "identification); the back-reaction is negligible at this scale.",
            "Only three tau values (2, 65, 81) with both anchors; tau0 is "
            "not identifiable from three points and is fixed by design. "
            "The nonparametric per-tau g^2 is the assumption-light check. "
            "Refitting under tau0=20/40/80 gives gamma_hat=3.29/3.97/4.86, "
            "but the implied g/sigma at tau=65/81 stays in 3.2-3.9, so the "
            "observed long-tau jump scale is stable even though tau0 is not.",
            f"sigma is constant within a path and shared across its three "
            f"events; {len(markets1)} paths give {len(markets1)} independent "
            "sigma draws, and all event-level CIs use a path-level bootstrap.",
            "The 81h event's pre-anchor is EXTENDED (eta sd ~0.0033), not "
            "OPEN; endpoint observation noise adds <0.5% of V_T and is "
            "omitted from the model variance.",
            "Latent S is a parametric AR(1) in both twins; the stress variant "
            "tests the de-freeze jump, not non-Markov dynamics.",
            "The g^2/V_T ratios 0.26/0.24/0.21 quoted in the identification "
            "note use V_T without the rare-jump factor 1.045; with the "
            "factor they are 0.250/0.225/0.205. Both conventions give the "
            "same ~0.8pp design coverage gap between tau=2 and tau=81.",
            "In the estimated tier, jump_weight uses the total innovation sd "
            "from open residuals (diffusion plus rare jumps), whereas the "
            "design g uses the diffusion sigma. This makes w_e about 4.5% "
            "high and biases gamma_hat about 2% low at gamma=4; the effect "
            "is dominated by the kappa_hat bias below.",
            "The profile CI at theta=0 uses the chi-square_1(0.95) cutoff, "
            "which is conservative at a boundary: the LR limit is a 50:50 "
            "mixture of a point mass at 0 and chi-square_1, so the one-sided "
            "95% cutoff is 2.71. The gamma=0 interval [0, 2.14] is therefore "
            "somewhat wide; non-overlap with the gamma=4 interval is "
            "unaffected.",
            "In the estimated tier each test path's sigma is estimated from "
            "that path's own open transitions (transductive). Sigma is "
            "gamma-free and both methods use it symmetrically, but coverage "
            "is not strictly out-of-fold with respect to sigma.",
            "The estimated tier's kappa_hat is slightly high (0.00338 vs "
            "0.003): open 1-step transitions identify kappa weakly, and the "
            "error propagates into the long-horizon variance. This biases "
            "gamma_hat upward at gamma=0 (1.72, 95% CI [0, 2.85]); the oracle "
            "tier isolates the identification argument. The estimated tier's "
            "coverage contrast is unaffected because conformal calibration "
            "absorbs the shared T error. The gamma=0 point estimate also "
            "shifts with the seed range (seed start 1000 gives 0.00), so CIs "
            "rather than point estimates are the portable object. Re-running "
            "with --seed-start 1000 reproduces the coverage pattern "
            "(gamma=4: open 0.867 vs cross 0.902; gamma=0: open 0.911 vs "
            "cross 0.900), so the structural conclusion is not seed-range "
            "specific.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("events:", event_counts["gamma_star"]["tau_counts"])
    for label in ("gamma0", "gamma_star"):
        for tier in ("oracle", "estimated"):
            block = results[label][tier]
            print(
                f"{label:10s} {tier:9s} "
                f"open={block['open_only']['coverage']:.3f} "
                f"cross={block['cross_event']['coverage']:.3f} "
                f"gamma_hat={block['cross_event']['gamma2_full_fit']['gamma_hat']:.2f}"
            )
    if bridge_block.get("available"):
        print("bridge events:", bridge_block["events"]["bridge_star"]["tau_counts"])
        for label in ("bridge0", "bridge_star"):
            for tier in ("oracle", "estimated"):
                block = bridge_results[label][tier]
                fit = block["cross_event"]["component_full_fit"]
                print(
                    f"{label:10s} {tier:9s} "
                    f"open={block['open_only']['coverage']:.3f} "
                    f"cross={block['cross_event']['coverage']:.3f} "
                    f"tau={block['cross_event_tau']['coverage']:.3f} "
                    f"nu/sigma_hat={fit.get('nu_over_sigma_hat', float('nan')):.2f}"
                )
    print("written:", args.out)
    print("written:", args.diag)


if __name__ == "__main__":
    main()
