"""G0 identification check for P1 (identifiability note §6.2, steps 1-4).

This is a *diagnostic* script, not part of the C1 coverage loop.  It reads the
frozen dataset and the diagnostic true paths; `true_paths.csv.gz` is used only
to score the estimators and is never fed to a feature/model.  The script writes
`results/identifiability_check.json` and prints a short table.

Checks
------
1. h_open noise: residual log P_open - log S_open, by window; compare with
   sigma_o = 0.0005 and check independence/serial correlation.
2. Short-range T: AR(1) fit on log S (OPEN+EXTENDED rows), plus the EIV
   attenuation correction when only log P_open is observed.
3. Freeze conditional law: 256 double-endpoint events (tau in {2,81}); compare
   the observed-endpoint regression with the true-S regression, correcting for
   regime-specific endpoint noise.  tau=81's pre-anchor is EXTENDED (AR(1)
   stationary sd ~0.0033), tau=2's anchors are OPEN (0.0005).
4. Belief width/coverage: Gaussian model-based belief vs. the empirical
   conditional law, with and without the EIV correction.  Jumps are reported
   separately because the Gaussian formula is not the full conditional law.

Run from the repo root:
    .venv-c1/bin/python projects/p1-c1-ope/code/identifiability_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

C1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = C1_DIR.parent / "results"
DATASET = RESULTS_DIR / "dataset.csv.gz"
DIAGNOSTICS = RESULTS_DIR / "true_paths.csv.gz"
OUT_JSON = RESULTS_DIR / "identifiability_check.json"

REGIME_OPEN = 0
REGIME_EXTENDED = 1
REGIME_WEEKEND = 2
REGIME_HOLIDAY = 3
REGIME_HALT = 4
FREEZE_REGIMES = (REGIME_WEEKEND, REGIME_HOLIDAY, REGIME_HALT)
REGIME_NAMES = {
    REGIME_OPEN: "OPEN",
    REGIME_EXTENDED: "EXTENDED",
    REGIME_WEEKEND: "WEEKEND",
    REGIME_HOLIDAY: "HOLIDAY",
    REGIME_HALT: "HALT",
}

SIGMA_O = 0.0005
SIGMA_ETA_EXTENDED = 0.002 / np.sqrt(1.0 - 0.8**2)
SIGMA_ETA_STALE = 0.0008 / np.sqrt(1.0 - 0.98**2)
KAPPA_TRUE = 0.003
MU_LOG = np.log(100.0)
NOMINAL = 0.90
Z90 = 1.6448536269514722  # standard-normal 95th percentile


def endpoint_noise_sd(regime: int) -> float:
    """Marginal log-observation noise sd for an endpoint anchor."""
    if regime == REGIME_OPEN:
        return SIGMA_O
    if regime == REGIME_EXTENDED:
        return SIGMA_ETA_EXTENDED
    if regime in (REGIME_WEEKEND, REGIME_HOLIDAY):
        return SIGMA_ETA_STALE
    return np.nan  # HALT: P is frozen, not a fresh noisy observation


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Simple OLS y = a + b x; returns (b, a, residuals)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x.mean()
    ym = y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        raise ValueError("OLS: zero variance in x")
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    a = float(ym - b * xm)
    return b, a, y - (a + b * x)


def ols_with_se(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, np.ndarray, float, float]:
    """OLS with conventional (homoskedastic) standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    b, a, res = ols(x, y)
    n = len(x)
    sxx = float(((x - x.mean()) ** 2).sum())
    sigma2 = float((res**2).sum() / max(n - 2, 1))
    se_b = float(np.sqrt(sigma2 / sxx))
    se_a = float(np.sqrt(sigma2 * (1.0 / n + x.mean() ** 2 / sxx)))
    return b, a, res, se_b, se_a


def robust_sd(residuals: np.ndarray) -> float:
    med = np.median(residuals)
    return float(1.4826 * np.median(np.abs(residuals - med)))


def cluster_se(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> float:
    """Cluster-robust standard error for the slope, clustered by `groups`."""
    b, _, res = ols(x, y)
    xc = np.asarray(x, dtype=float) - np.mean(x)
    frame = pd.DataFrame({"score": xc * res, "group": np.asarray(groups)})
    meat = float((frame.groupby("group")["score"].sum() ** 2).sum())
    bread = float((xc**2).sum())
    n_groups = int(frame["group"].nunique())
    correction = n_groups / max(n_groups - 1, 1)
    return float(np.sqrt(meat * correction) / bread)


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def load_paths() -> pd.DataFrame:
    diag = pd.read_csv(DIAGNOSTICS)
    diag = diag.sort_values(["window", "seed", "episode_step"], ignore_index=True)
    diag["step"] = diag["episode_step"] - 1
    return diag


def add_next_step_columns(diag: pd.DataFrame) -> pd.DataFrame:
    diag = diag.copy()
    grouped = diag.groupby(["window", "seed"], sort=False)
    diag["next_regime"] = grouped["regime"].shift(-1)
    diag["next_true_price"] = grouped["true_price"].shift(-1)
    diag["next_pool_price"] = grouped["pool_price"].shift(-1)
    return diag


def extract_events(diag: pd.DataFrame) -> list[dict]:
    """Freeze runs of WEEKEND/HOLIDAY/HALT, with pre/post anchor indices."""
    events: list[dict] = []
    for (window, seed), path in diag.groupby(["window", "seed"], sort=False):
        path = path.sort_values("step")
        steps = path["episode_step"].to_numpy()
        assert np.all(np.diff(steps) == 1), (
            f"non-contiguous episode_step for window={window}, seed={seed}"
        )
        regime = path["regime"].to_numpy()
        true_price = path["true_price"].to_numpy()
        pool_price = path["pool_price"].to_numpy()
        sigma = float(path["sigma"].iloc[0])
        t = 0
        n = len(regime)
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
                    "sigma": sigma,
                    "pre": pre,
                    "post": post,
                    "pre_regime": int(regime[pre]) if pre is not None else None,
                    "post_regime": int(regime[post]) if post is not None else None,
                    "s_minus": float(true_price[pre]) if pre is not None else np.nan,
                    "s_plus": float(true_price[post]) if post is not None else np.nan,
                    "p_minus": float(pool_price[pre]) if pre is not None else np.nan,
                    "p_plus": float(pool_price[post]) if post is not None else np.nan,
                }
            )
    return events


def step1_h_open(diag: pd.DataFrame) -> dict:
    open_rows = diag[diag["regime"] == REGIME_OPEN].copy()
    eps = np.log(open_rows["pool_price"].to_numpy()) - np.log(
        open_rows["true_price"].to_numpy()
    )
    open_rows["eps"] = eps

    by_window = {
        str(w): {"eps": describe(g["eps"].to_numpy())}
        for w, g in open_rows.groupby("window", sort=True)
    }
    # Lag-1 correlation within consecutive OPEN steps (same path).
    ordered = add_next_step_columns(diag).sort_values(
        ["window", "seed", "episode_step"]
    )
    pair_mask = (ordered["regime"] == REGIME_OPEN) & (
        ordered["next_regime"] == REGIME_OPEN
    )
    pairs = ordered[pair_mask].copy()
    pairs["eps_t"] = np.log(pairs["pool_price"]) - np.log(pairs["true_price"])
    pairs["eps_next"] = np.log(pairs["next_pool_price"]) - np.log(
        pairs["next_true_price"]
    )
    lag1 = float(np.corrcoef(pairs["eps_t"], pairs["eps_next"])[0, 1])
    corr_with_s = float(
        np.corrcoef(eps, np.log(open_rows["true_price"].to_numpy()))[0, 1]
    )
    return {
        "n_open_rows": int(len(open_rows)),
        "residual": describe(eps),
        "sd_vs_sigma_o": float(np.std(eps, ddof=1) / SIGMA_O),
        "lag1_corr_within_open": lag1,
        "corr_with_log_true_price": corr_with_s,
        "by_window": by_window,
        "note": "sigma_o is identified from OPEN rows with true S as a diagnostic; "
        "the estimating target is the residual sd, not a blind deconvolution.",
    }


def step2_short_range_T(diag: pd.DataFrame) -> dict:
    d = add_next_step_columns(diag)
    informative = d["regime"].isin([REGIME_OPEN, REGIME_EXTENDED]) & d[
        "next_regime"
    ].isin([REGIME_OPEN, REGIME_EXTENDED])
    pairs = d[informative].dropna(subset=["next_true_price", "next_pool_price"]).copy()
    x_true = np.log(pairs["true_price"].to_numpy())
    y_true = np.log(pairs["next_true_price"].to_numpy())
    b_true, a_true, res_true, se_b_true, _ = ols_with_se(x_true, y_true)
    groups = (
        pairs["window"].astype(str) + "_" + pairs["seed"].astype(str)
    ).to_numpy()
    se_b_true_cluster = cluster_se(x_true, y_true, groups)
    kappa_hat = 1.0 - b_true
    mu_hat = a_true / kappa_hat if kappa_hat > 0 else np.nan

    # Jump diagnostic: normalize by the path-level sigma, otherwise the
    # lognormal spread of sigma across paths looks like excess kurtosis.
    sigma_pair = pairs["sigma"].to_numpy()
    res_norm = res_true / sigma_pair
    sd_rob_norm = robust_sd(res_norm)
    p3 = float(np.mean(np.abs(res_norm) > 3.0 * sd_rob_norm))
    p4 = float(np.mean(np.abs(res_norm) > 4.0 * sd_rob_norm))
    # P(|Z|>4)=6.3e-5; a jump step has variance 10 sigma^2, so its residual
    # exceeds 4 sigma with prob P(|Z|>4/sqrt(10))=0.206.  The excess is a rough
    # jump-rate estimate.
    jump_rate_hat = max(0.0, (p4 - 6.3e-5) / 0.206)
    # A second, moment-based estimate: for a small jump mixture with variance
    # ratio 9, excess kurtosis ~= 243 * p_jump (m4 = 3 + 297p, Var = 1 + 8p).
    z = res_norm - res_norm.mean()
    m2 = float(np.mean(z**2))
    m4 = float(np.mean(z**4))
    excess_kurt = m4 / m2**2 - 3.0
    jump_rate_kurt = max(0.0, excess_kurt / 243.0)

    # Within-path (fixed-effects) slope.  A naive per-path OLS is biased
    # downward by the Nickell bias in a short dynamic panel; the pooled fit
    # above mixes between-path variation.  Report the within estimate and the
    # standard Nickell correction as a diagnostic.
    pairs["lx"] = x_true
    pairs["ly"] = y_true
    grouped = pairs.groupby(["window", "seed"], sort=False)
    pairs["dx"] = pairs["lx"] - grouped["lx"].transform("mean")
    pairs["dy"] = pairs["ly"] - grouped["ly"].transform("mean")
    b_within = float(
        (pairs["dx"] * pairs["dy"]).sum() / (pairs["dx"] ** 2).sum()
    )
    t_bar = float(grouped.size().mean())
    nickell_bias = float(-(1.0 + b_within) / (t_bar - 1.0))
    b_within_corrected = b_within - nickell_bias

    # Observed OPEN->OPEN: attenuation from sigma_o.
    open_pairs = pairs[
        (pairs["regime"] == REGIME_OPEN)
        & (pairs["next_regime"] == REGIME_OPEN)
    ]
    x_obs = np.log(open_pairs["pool_price"].to_numpy())
    y_obs = np.log(open_pairs["next_pool_price"].to_numpy())
    b_obs, a_obs, res_obs, se_b_obs, _ = ols_with_se(x_obs, y_obs)
    open_groups = (
        open_pairs["window"].astype(str) + "_" + open_pairs["seed"].astype(str)
    ).to_numpy()
    se_b_obs_cluster = cluster_se(x_obs, y_obs, open_groups)
    var_y = float(np.var(x_obs, ddof=1))
    var_s = max(var_y - SIGMA_O**2, 1e-12)
    reliability = var_s / var_y
    b_corr = b_obs / reliability

    # Same-subset true-S fit, so the observed-vs-true comparison is apples-to-apples.
    b_true_open, a_true_open, _, se_b_true_open, _ = ols_with_se(
        np.log(open_pairs["true_price"].to_numpy()),
        np.log(open_pairs["next_true_price"].to_numpy()),
    )
    return {
        "n_pairs_informative": int(len(pairs)),
        "true_S_fit": {
            "slope": b_true,
            "slope_se": se_b_true,
            "slope_se_cluster": se_b_true_cluster,
            "intercept": a_true,
            "kappa_hat": kappa_hat,
            "mu_hat_log": mu_hat,
            "residual_sd": float(np.std(res_true, ddof=1)),
            "residual_sd_robust": robust_sd(res_true),
            "mean_abs_res_over_sigma": float(np.mean(np.abs(res_norm))),
            "sd_res_over_sigma": float(np.std(res_norm, ddof=1)),
            "r2": float(1.0 - np.var(res_true, ddof=1) / np.var(y_true, ddof=1)),
            "jump_rate_hat_threshold": jump_rate_hat,
            "jump_rate_hat_kurtosis": jump_rate_kurt,
            "excess_kurtosis_normalized": excess_kurt,
            "p_abs_res_gt_3sd_normalized": p3,
            "p_abs_res_gt_4sd_normalized": p4,
        },
        "within_path_fit": {
            "slope_within": b_within,
            "kappa_within": 1.0 - b_within,
            "t_bar": t_bar,
            "nickell_bias_approx": nickell_bias,
            "slope_nickell_corrected": b_within_corrected,
            "kappa_nickell_corrected": 1.0 - b_within_corrected,
        },
        "observed_OPEN_fit": {
            "n_pairs": int(len(open_pairs)),
            "slope_observed": b_obs,
            "slope_se": se_b_obs,
            "slope_se_cluster": se_b_obs_cluster,
            "slope_eiv_corrected": b_corr,
            "kappa_hat_corrected": 1.0 - b_corr,
            "reliability": reliability,
            "residual_sd_observed": float(np.std(res_obs, ddof=1)),
            "same_subset_true_S_slope": b_true_open,
            "same_subset_true_S_slope_se": se_b_true_open,
        },
        "generator_values": {
            "kappa": KAPPA_TRUE,
            "sigma_base": 0.008,
            "jump_prob": 0.005,
        },
        "note": "The observed EXTENDED->EXTENDED fit is omitted because eta is "
        "serially correlated; the OPEN-only EIV correction is the clean check.",
    }


def _endpoint_arrays(events: list[dict], tau: int) -> dict:
    rows = [
        e for e in events if e["tau"] == tau and e["pre"] is not None and e["post"] is not None
    ]
    out = {
        "n": len(rows),
        "s_minus": np.array([e["s_minus"] for e in rows]),
        "s_plus": np.array([e["s_plus"] for e in rows]),
        "p_minus": np.array([e["p_minus"] for e in rows]),
        "p_plus": np.array([e["p_plus"] for e in rows]),
        "sigma": np.array([e["sigma"] for e in rows]),
        "pre_regime": np.array([e["pre_regime"] for e in rows]),
        "post_regime": np.array([e["post_regime"] for e in rows]),
    }
    return out


def _observed_belief(
    p_minus: np.ndarray,
    sigma_minus: float,
    slope: float,
    intercept: float,
    cond_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed-only Gaussian belief for log S^+ given observed log P^-.

    Model: s^- ~ N(mu_s, var_s); p^- = s^- + eps, eps ~ N(0, sigma_minus^2);
    s^+ | s^- ~ N(intercept + slope*s^-, cond_var).  The prior moments are
    estimated from p^- and the known noise variance, not from the latent s^-.
    Returns (mean, var) of s^+ | p^-.
    """
    y = np.log(p_minus)
    mu_s = float(np.mean(y))
    var_p = float(np.var(y, ddof=1))
    var_s = max(var_p - sigma_minus**2, 1e-12)
    reliability = var_s / var_p
    post_mean = mu_s + reliability * (y - y.mean())
    post_var = var_s * (1.0 - reliability)
    pred_mean = intercept + slope * post_mean
    pred_var = slope**2 * post_var + cond_var
    return pred_mean, pred_var


def _coverage(
    y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, z: float = Z90
) -> dict:
    sd = np.sqrt(np.maximum(var, 0.0))
    lo = mean - z * sd
    hi = mean + z * sd
    covered = (y_true >= lo) & (y_true <= hi)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(hi - lo)),
        "n": int(len(y_true)),
    }


def step3_freeze_conditional(events: list[dict]) -> dict:
    tau_counts = pd.Series([e["tau"] for e in events]).value_counts().sort_index()
    n_double = sum(
        1 for e in events if e["pre"] is not None and e["post"] is not None
    )
    out = {
        "n_events_total": len(events),
        "n_double_endpoint": int(n_double),
        "tau_counts": {int(k): int(v) for k, v in tau_counts.items()},
        "by_tau": {},
    }
    for tau in (2, 81):
        arr = _endpoint_arrays(events, tau)
        if arr["n"] == 0:
            continue
        x_true = np.log(arr["s_minus"])
        y_true = np.log(arr["s_plus"])
        b_true, a_true, res_true, se_b_true, _ = ols_with_se(x_true, y_true)
        var_true = float(np.var(res_true, ddof=1))
        sd_true = float(np.sqrt(var_true))

        # Regime-specific endpoint noise.  tau=2: OPEN/OPEN; tau=81:
        # EXTENDED/OPEN.  Assert rather than silently using the wrong noise.
        pre_reg = int(arr["pre_regime"][0])
        post_reg = int(arr["post_regime"][0])
        assert np.all(arr["pre_regime"] == pre_reg), "mixed pre-endpoint regimes"
        assert np.all(arr["post_regime"] == post_reg), "mixed post-endpoint regimes"
        sd_minus = endpoint_noise_sd(pre_reg)
        sd_plus = endpoint_noise_sd(post_reg)

        x_obs = np.log(arr["p_minus"])
        y_obs = np.log(arr["p_plus"])
        b_obs, a_obs, res_obs, se_b_obs, _ = ols_with_se(x_obs, y_obs)
        var_y_obs = float(np.var(x_obs, ddof=1))
        var_s_minus = max(var_y_obs - sd_minus**2, 1e-12)
        reliability = var_s_minus / var_y_obs
        b_corr = b_obs / reliability
        a_corr = float(y_obs.mean() - b_corr * x_obs.mean())

        # Observed residual variance = Var(s+ | s-) + sigma_+^2
        # + slope_true^2 * Var(s- | p-).
        var_cond_obs = float(np.var(res_obs, ddof=1))
        var_s_given_p = var_s_minus * (1.0 - reliability)
        var_cond_corr = var_cond_obs - sd_plus**2 - b_true**2 * var_s_given_p
        var_cond_corr = max(var_cond_corr, 1e-12)

        # Theoretical Gaussian conditional sd (no jumps) for reference.
        # The anchors are separated by tau freeze steps PLUS the two anchor
        # steps themselves: t_post - t_pre = tau + 1.
        anchor_gap = tau + 1
        decay = (1.0 - KAPPA_TRUE) ** anchor_gap
        theo_slope = float(decay)
        theo_sd = float(
            np.sqrt(np.mean(arr["sigma"] ** 2))
            * np.sqrt((1.0 - decay**2) / (1.0 - (1.0 - KAPPA_TRUE) ** 2))
        )

        # Belief comparison, all scored against the latent s_plus.
        oracle_mean = a_true + b_true * x_true
        oracle = _coverage(y_true, oracle_mean, np.full_like(oracle_mean, var_true))

        eiv_mean, eiv_var = _observed_belief(
            arr["p_minus"], sd_minus, b_corr, a_corr, var_cond_corr
        )
        eiv = _coverage(y_true, eiv_mean, eiv_var)

        naive_mean = a_obs + b_obs * x_obs
        naive = _coverage(y_true, naive_mean, np.full_like(naive_mean, var_cond_obs))

        # Empirical conditional quantiles (jump-aware oracle) for the residual.
        q05, q50, q95 = np.quantile(res_true, [0.05, 0.50, 0.95])

        out["by_tau"][str(tau)] = {
            "n": int(arr["n"]),
            "freeze_steps": tau,
            "anchor_gap_steps": anchor_gap,
            "pre_regime": REGIME_NAMES[pre_reg],
            "post_regime": REGIME_NAMES[post_reg],
            "endpoint_noise_sd": {
                "pre": float(sd_minus),
                "post": float(sd_plus),
            },
            "true_S_fit": {
                "slope": b_true,
                "slope_se": se_b_true,
                "intercept": a_true,
                "residual_sd": sd_true,
                "r2": float(
                    1.0 - var_true / float(np.var(y_true, ddof=1))
                ),
            },
            "observed_fit": {
                "slope_observed": b_obs,
                "slope_se": se_b_obs,
                "slope_eiv_corrected": b_corr,
                "intercept_eiv_corrected": a_corr,
                "reliability_pre": reliability,
                "residual_sd_observed": float(np.sqrt(var_cond_obs)),
                "residual_sd_eiv_corrected": float(np.sqrt(var_cond_corr)),
            },
            "theory_gaussian_no_jumps": {
                "slope": theo_slope,
                "residual_sd": theo_sd,
            },
            "empirical_residual_quantiles": {
                "q05": float(q05),
                "q50": float(q50),
                "q95": float(q95),
            },
            "belief_90pct": {
                "oracle": oracle,
                "eiv_corrected": eiv,
                "naive_observed": naive,
                "true_residual_sd_over_theory": float(sd_true / theo_sd),
            },
        }
    return out


def step3_eiv_stress(
    events: list[dict],
    sigma_adds: tuple[float, ...] = (0.005, 0.02, 0.05, 0.1),
    seed: int = 20261020,
    n_rep: int = 200,
) -> dict:
    """Artificial-noise stress for the EIV correction (diagnostic only).

    The frozen dataset is untouched; we add independent log-noise to the
    observed endpoints and rerun the EIV vs naive comparison.  This is the
    zero-cost way to check whether the correction matters when the endpoint
    noise is comparable to the conditional spread.

    Each sigma level is replicated ``n_rep`` times: a single noise draw has
    sampling covariance with the fixed latent x of order ``sigma_add*sd(x)/sqrt(n)``,
    which at n=128 is large enough to swamp the attenuation at moderate noise.
    We therefore report replicate means and Monte-Carlo standard errors.
    """
    rng = np.random.default_rng(seed)
    out = {
        "sigma_added": [float(s) for s in sigma_adds],
        "n_rep": int(n_rep),
        "note": "Artificial endpoint noise on top of the frozen observations; "
        "the latent s_plus is unchanged. Diagnostic only. Reported values are "
        "means over n_rep independent noise draws; sd/se are across replicates.",
        "by_tau": {},
    }
    for tau in (2, 81):
        arr = _endpoint_arrays(events, tau)
        if arr["n"] == 0:
            continue
        y_true = np.log(arr["s_plus"])
        x_base = np.log(arr["p_minus"])
        y_base = np.log(arr["p_plus"])
        b_true, a_true, res_true = ols(np.log(arr["s_minus"]), y_true)
        pre_reg = int(arr["pre_regime"][0])
        post_reg = int(arr["post_regime"][0])
        sd_minus = endpoint_noise_sd(pre_reg)
        sd_plus = endpoint_noise_sd(post_reg)
        rows = []
        for add in sigma_adds:
            sd_minus_eff = float(np.sqrt(sd_minus**2 + add**2))
            sd_plus_eff = float(np.sqrt(sd_plus**2 + add**2))
            rec = {
                "reliability": [],
                "slope_naive": [],
                "slope_eiv": [],
                "slope_predicted": [],
                "cov_naive": [],
                "cov_eiv": [],
                "width_naive": [],
                "width_eiv": [],
            }
            for _ in range(n_rep):
                x_obs = x_base + rng.normal(0.0, add, size=arr["n"])
                y_obs = y_base + rng.normal(0.0, add, size=arr["n"])
                b_obs, a_obs, res_obs = ols(x_obs, y_obs)
                var_y = float(np.var(x_obs, ddof=1))
                var_s = max(var_y - sd_minus_eff**2, 1e-12)
                reliability = var_s / var_y
                b_corr = b_obs / reliability
                a_corr = float(y_obs.mean() - b_corr * x_obs.mean())
                var_cond_obs = float(np.var(res_obs, ddof=1))
                var_s_given_p = var_s * (1.0 - reliability)
                var_cond_corr = max(
                    var_cond_obs - sd_plus_eff**2 - b_true**2 * var_s_given_p,
                    1e-12,
                )

                eiv_mean, eiv_var = _observed_belief(
                    np.exp(x_obs), sd_minus_eff, b_corr, a_corr, var_cond_corr
                )
                naive_mean = a_obs + b_obs * x_obs
                eiv_cov = _coverage(y_true, eiv_mean, eiv_var)
                naive_cov = _coverage(
                    y_true, naive_mean, np.full_like(naive_mean, var_cond_obs)
                )
                rec["reliability"].append(reliability)
                rec["slope_naive"].append(b_obs)
                rec["slope_eiv"].append(b_corr)
                rec["slope_predicted"].append(b_true * reliability)
                rec["cov_naive"].append(naive_cov["coverage"])
                rec["cov_eiv"].append(eiv_cov["coverage"])
                rec["width_naive"].append(naive_cov["mean_width"])
                rec["width_eiv"].append(eiv_cov["mean_width"])

            def stat(xs):
                a = np.asarray(xs, dtype=float)
                return {
                    "mean": float(a.mean()),
                    "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    "mc_se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                }

            rows.append(
                {
                    "sigma_added": float(add),
                    "sd_minus_effective": sd_minus_eff,
                    "sd_plus_effective": sd_plus_eff,
                    "true_slope": b_true,
                    "reliability": stat(rec["reliability"]),
                    "slope_naive": stat(rec["slope_naive"]),
                    "slope_eiv_corrected": stat(rec["slope_eiv"]),
                    "slope_attenuation_prediction": stat(rec["slope_predicted"]),
                    "naive_slope_bias": stat(
                        [b - b_true for b in rec["slope_naive"]]
                    ),
                    "eiv_slope_bias": stat(
                        [b - b_true for b in rec["slope_eiv"]]
                    ),
                    "naive_belief_90pct": {
                        **stat(rec["cov_naive"]),
                        "width": stat(rec["width_naive"]),
                    },
                    "eiv_belief_90pct": {
                        **stat(rec["cov_eiv"]),
                        "width": stat(rec["width_eiv"]),
                    },
                    "frac_eiv_bias_smaller": float(
                        np.mean(
                            [
                                abs(be - b_true) < abs(bn - b_true)
                                for bn, be in zip(rec["slope_naive"], rec["slope_eiv"])
                            ]
                        )
                    ),
                }
            )
        out["by_tau"][str(tau)] = {
            "n": int(arr["n"]),
            "pre_regime": REGIME_NAMES[pre_reg],
            "post_regime": REGIME_NAMES[post_reg],
            "rows": rows,
        }
    return out


def main() -> None:
    diag = load_paths()
    dataset = pd.read_csv(
        DATASET,
        usecols=["strategy", "window", "seed", "episode_step", "current_regime"],
    )
    # Cross-check: one pool price column in the dataset matches the diagnostic.
    sample = (
        dataset[["window", "seed", "episode_step", "current_regime"]]
        .drop_duplicates()
        .merge(
            diag[["window", "seed", "episode_step", "regime"]],
            on=["window", "seed", "episode_step"],
            how="inner",
        )
    )
    regime_match = bool(
        (sample["current_regime"] == sample["regime"]).all()
    )

    events = extract_events(diag)
    payload = {
        "meta": {
            "description": "P1 G0 identification check (identifiability note §6.2)",
            "dataset_rows": int(len(dataset)),
            "diagnostic_rows": int(len(diag)),
            "n_paths": int(diag.groupby(["window", "seed"]).ngroups),
            "regime_match_dataset_vs_diagnostics": regime_match,
            "truth_paths_usage": "diagnostic/scoring only; never a feature",
        },
        "step1_h_open": step1_h_open(diag),
        "step2_short_range_T": step2_short_range_T(diag),
        "step3_freeze_conditional": step3_freeze_conditional(events),
        "step3_eiv_stress": step3_eiv_stress(events),
        "limits": [
            "Population identification, not a finite-sample nonparametric deconvolution.",
            "sigma_o and the latent S are used as diagnostics; the observed-only EIV check uses p_minus and the known endpoint noise, not the latent s_minus.",
            "tau=9 and tau=56 have only one anchor; they are excluded from step 3.",
            "The Gaussian belief is a model-based approximation; jumps can widen the true conditional law.",
            "The endpoint-noise stress is artificial and does not modify the frozen dataset.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
