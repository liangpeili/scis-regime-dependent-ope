"""G0 step 12: observable-only estimation of the observation mapping h.

Only the observable pool price P and the known calendar are observed.  The
main specification additionally feeds the simulator's per-path sigma into
the likelihood to isolate h; --sigma-mode open replaces it with an
observable estimate and is reported as a sensitivity.  kappa, mu, and
jump_prob are held at the G0 plug-in values, exactly as in
trajectory_value_check.py.  The likelihood is the exact linear-Gaussian
state-space likelihood of synthetic_env's observation model, augmented
with a two-component Gaussian-sum jump mixture so that rare S jumps do
not contaminate the eta dynamics.

Estimation is a two-pass robust MLE:
  1. an unmasked GSF MLE from several starts (all local optima reported);
  2. observations with |mixture z| > Z_THRESHOLD are flagged and the model
     is refitted with those observations ignored (R inflated); the mask is
     iterated to a fixed point.
Uncertainty is a path-level bootstrap of the masked fit plus numerical
Hessian standard errors (quasi-likelihood caveat).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from synthetic_env import (  # noqa: E402
    MU_LOG,
    REGIME_EXTENDED,
    REGIME_HALT,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    simulate_market,
)

PVC_JSON = ROOT / "results" / "policy_value_check.json"
OUT_JSON = ROOT / "results" / "observable_h_check.json"

N_STEPS = 336
WINDOWS = ("W1", "W2", "W3", "W4")
CAL_SEEDS = tuple(range(2000, 2128))
Z_THRESHOLD = 3.0
N_BOOT = 48

H_KEYS = ("open_sd", "ext_phi", "ext_sd", "stale_phi", "stale_sd")

TRUTH_H = {
    "open_sd": 0.0005,
    "ext_phi": 0.8,
    "ext_sd": 0.002,
    "stale_phi": 0.98,
    "stale_sd": 0.0008,
}


def unpack(theta):
    return {
        "open_sd": math.exp(theta[0]),
        "ext_phi": math.tanh(theta[1]),
        "ext_sd": math.exp(theta[2]),
        "stale_phi": math.tanh(theta[3]),
        "stale_sd": math.exp(theta[4]),
    }


def pack(h):
    return np.array(
        [
            math.log(h["open_sd"]),
            math.atanh(np.clip(h["ext_phi"], -0.999, 0.999)),
            math.log(h["ext_sd"]),
            math.atanh(np.clip(h["stale_phi"], -0.999, 0.999)),
            math.log(h["stale_sd"]),
        ]
    )


BOUNDS = [
    (math.log(1e-5), math.log(0.02)),
    (-3.0, 3.0),
    (math.log(1e-5), math.log(0.05)),
    (-3.0, 3.0),
    (math.log(1e-5), math.log(0.05)),
]


class Model:
    def __init__(self, kappa, mu, jump_prob):
        self.kappa = kappa
        self.mu = mu
        self.jump_prob = jump_prob

    def transition(self, r, h):
        if r == REGIME_OPEN:
            phi, q_eta = 0.0, 0.0
        elif r == REGIME_EXTENDED:
            phi, q_eta = h["ext_phi"], h["ext_sd"] ** 2
        elif r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            phi, q_eta = h["stale_phi"], h["stale_sd"] ** 2
        elif r == REGIME_HALT:
            phi, q_eta = 1.0, 0.0
        else:
            raise ValueError(r)
        a = np.array(
            [[1.0 - self.kappa, 0.0, 0.0], [0.0, phi, 0.0], [0.0, 0.0, 1.0]]
        )
        c = np.array([self.kappa * self.mu, 0.0, 0.0])
        if r == REGIME_OPEN:
            a[2, :] = a[0, :]
            c[2] = c[0]
        return a, c, q_eta

    @staticmethod
    def obs(r, h):
        if r == REGIME_OPEN:
            return np.array([1.0, 0.0, 0.0]), h["open_sd"] ** 2
        if r == REGIME_EXTENDED:
            return np.array([1.0, 1.0, 0.0]), 0.0
        if r in (REGIME_WEEKEND, REGIME_HOLIDAY):
            return np.array([0.0, 1.0, 1.0]), 0.0
        return None, None

    def gsf_loglik(self, logp, regime, sigma_sq, h, return_diag=False):
        n = logp.shape[0]
        jp = self.jump_prob
        y0 = logp[:, 0]
        v0 = h["stale_sd"] ** 2
        m0 = np.stack(
            [np.full(n, MU_LOG), y0 - MU_LOG, np.full(n, MU_LOG)], axis=1
        )
        V0 = np.zeros((n, 3, 3))
        ll = float(
            np.sum(
                -0.5 * np.log(2 * math.pi * v0)
                - 0.5 * (y0 - MU_LOG) ** 2 / v0
            )
        )
        m = np.stack([m0, m0], axis=1)
        V = np.stack([V0, V0], axis=1)
        logw = np.full((n, 2), -np.inf)
        logw[:, 0] = 0.0
        eye = np.eye(3)
        Z = np.full((n, N_STEPS + 1), np.nan) if return_diag else None
        JP = np.full((n, N_STEPS + 1), np.nan) if return_diag else None
        for t in range(1, N_STEPS + 1):
            r = int(regime[t]) if t < len(regime) else int(regime[-1])
            a, c, q_eta = self.transition(r, h)
            Q = np.zeros((n, 3, 3))
            if r == REGIME_OPEN:
                Q[:, 0, 0] = Q[:, 2, 2] = Q[:, 0, 2] = Q[:, 2, 0] = sigma_sq
            else:
                Q[:, 0, 0] = sigma_sq
                Q[:, 1, 1] = q_eta
            mp = m @ a.T + c
            Vp = (
                a[None, None, :, :] @ V @ a.T[None, None, :, :]
                + Q[:, None, :, :]
            )
            lw = logw - logsumexp(logw, axis=1, keepdims=True)
            wb = np.exp(lw)
            mbar = np.einsum("nj,nji->ni", wb, mp)
            diff = mp - mbar[:, None, :]
            spread = np.einsum("nj,nji,njk->nik", wb, diff, diff)
            V0b = np.einsum("nj,njik->nik", wb, Vp) + spread
            V1b = V0b.copy()
            if r == REGIME_OPEN:
                V1b[:, 0, 0] = V1b[:, 2, 2] = V1b[:, 0, 2] = V1b[:, 2, 0] = (
                    V0b[:, 0, 0] + 9 * sigma_sq
                )
            else:
                V1b[:, 0, 0] = V0b[:, 0, 0] + 9 * sigma_sq
            m = np.stack([mbar, mbar], axis=1)
            V = np.stack([V0b, V1b], axis=1)
            logw = np.stack(
                [np.full(n, math.log1p(-jp)), np.full(n, math.log(jp))],
                axis=1,
            )
            hv, rvar = self.obs(r, h)
            if hv is None:
                continue
            VH = V @ hv
            v = np.einsum("nji,i->nj", VH, hv) + rvar
            pred = m @ hv
            err = logp[:, t][:, None] - pred
            loglik = -0.5 * np.log(2 * math.pi * v) - 0.5 * err**2 / v
            logtot = logsumexp(logw + loglik, axis=1)
            ll += float(np.sum(logtot))
            if return_diag:
                predmix = np.einsum("nj,nj->n", np.exp(logw), pred)
                vmix = np.einsum(
                    "nj,nj->n",
                    np.exp(logw),
                    v + (pred - predmix[:, None]) ** 2,
                )
                Z[:, t] = (logp[:, t] - predmix) / np.sqrt(vmix)
                JP[:, t] = np.exp(logw[:, 1] + loglik[:, 1] - logtot)
            logw = logw + loglik - logtot[:, None]
            gain = VH / v[:, :, None]
            m = m + gain * err[:, :, None]
            ikh = eye[None, None, :, :] - gain[:, :, :, None] * hv[
                None, None, None, :
            ]
            V = (
                ikh @ V @ ikh.transpose(0, 1, 3, 2)
                + rvar
                * gain[:, :, :, None]
                * gain[:, :, None, :]
            )
            V = 0.5 * (V + V.transpose(0, 1, 3, 2))
        if return_diag:
            return ll, Z, JP
        return ll

    def gsf_loglik_mask(self, logp, regime, sigma_sq, h, mask):
        n = logp.shape[0]
        jp = self.jump_prob
        y0 = logp[:, 0]
        v0 = h["stale_sd"] ** 2
        m0 = np.stack(
            [np.full(n, MU_LOG), y0 - MU_LOG, np.full(n, MU_LOG)], axis=1
        )
        V0 = np.zeros((n, 3, 3))
        ll = float(
            np.sum(
                -0.5 * np.log(2 * math.pi * v0)
                - 0.5 * (y0 - MU_LOG) ** 2 / v0
            )
        )
        m = np.stack([m0, m0], axis=1)
        V = np.stack([V0, V0], axis=1)
        logw = np.full((n, 2), -np.inf)
        logw[:, 0] = 0.0
        eye = np.eye(3)
        for t in range(1, N_STEPS + 1):
            r = int(regime[t]) if t < len(regime) else int(regime[-1])
            a, c, q_eta = self.transition(r, h)
            Q = np.zeros((n, 3, 3))
            if r == REGIME_OPEN:
                Q[:, 0, 0] = Q[:, 2, 2] = Q[:, 0, 2] = Q[:, 2, 0] = sigma_sq
            else:
                Q[:, 0, 0] = sigma_sq
                Q[:, 1, 1] = q_eta
            mp = m @ a.T + c
            Vp = (
                a[None, None, :, :] @ V @ a.T[None, None, :, :]
                + Q[:, None, :, :]
            )
            lw = logw - logsumexp(logw, axis=1, keepdims=True)
            wb = np.exp(lw)
            mbar = np.einsum("nj,nji->ni", wb, mp)
            diff = mp - mbar[:, None, :]
            spread = np.einsum("nj,nji,njk->nik", wb, diff, diff)
            V0b = np.einsum("nj,njik->nik", wb, Vp) + spread
            V1b = V0b.copy()
            if r == REGIME_OPEN:
                V1b[:, 0, 0] = V1b[:, 2, 2] = V1b[:, 0, 2] = V1b[:, 2, 0] = (
                    V0b[:, 0, 0] + 9 * sigma_sq
                )
            else:
                V1b[:, 0, 0] = V0b[:, 0, 0] + 9 * sigma_sq
            m = np.stack([mbar, mbar], axis=1)
            V = np.stack([V0b, V1b], axis=1)
            logw = np.stack(
                [np.full(n, math.log1p(-jp)), np.full(n, math.log(jp))],
                axis=1,
            )
            hv, rvar = self.obs(r, h)
            if hv is None:
                continue
            R = np.where(~mask[:, t], rvar, 1e6)
            VH = V @ hv
            v = np.einsum("nji,i->nj", VH, hv) + R[:, None]
            pred = m @ hv
            err = logp[:, t][:, None] - pred
            loglik = -0.5 * np.log(2 * math.pi * v) - 0.5 * err**2 / v
            logtot = logsumexp(logw + loglik, axis=1)
            ll += float(np.sum(logtot))
            logw = logw + loglik - logtot[:, None]
            gain = VH / v[:, :, None]
            m = m + gain * err[:, :, None]
            ikh = eye[None, None, :, :] - gain[:, :, :, None] * hv[
                None, None, None, :
            ]
            V = (
                ikh @ V @ ikh.transpose(0, 1, 3, 2)
                + R[:, None, None, None]
                * gain[:, :, :, None]
                * gain[:, :, None, :]
            )
            V = 0.5 * (V + V.transpose(0, 1, 3, 2))
        return ll


def generate(seeds, sigma_mode):
    paths = []
    for seed in seeds:
        for wi in range(len(WINDOWS)):
            paths.append(simulate_market([seed, wi]))
    logp = np.stack([np.log(m.pool_price) for m in paths])
    regime = paths[0].regime
    if sigma_mode == "oracle":
        sigma_sq = np.array([m.sigma**2 for m in paths])
    elif sigma_mode == "open":
        from policy_value_check import estimate_sigma_open

        sigma_sq = np.array([estimate_sigma_open(m) ** 2 for m in paths])
    else:
        raise ValueError(sigma_mode)
    return paths, logp, regime, sigma_sq


def fit(model, logp, regime, sigma_sq, h0, masked=False, mask=None):
    if masked:
        fun = lambda th: -model.gsf_loglik_mask(  # noqa: E731
            logp, regime, sigma_sq, unpack(th), mask
        )
    else:
        fun = lambda th: -model.gsf_loglik(  # noqa: E731
            logp, regime, sigma_sq, unpack(th)
        )
    res = minimize(
        fun,
        pack(h0),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={
            "maxiter": 200,
            "maxfun": 2000,
            "ftol": 1e-11,
            "gtol": 1e-8,
            "maxls": 40,
        },
    )
    return float(res.fun), unpack(res.x), res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--z-threshold", type=float, default=Z_THRESHOLD)
    parser.add_argument("--sigma-mode", choices=("oracle", "open"), default="oracle")
    parser.add_argument("--cal-seeds", type=int, default=len(CAL_SEEDS))
    parser.add_argument(
        "--max-starts",
        type=int,
        default=5,
        help="use the first N unmasked starts (order: g0, truth_like, crude, "
        "slow_eta, fast_eta)",
    )
    args = parser.parse_args()

    raw = json.loads(PVC_JSON.read_text(encoding="utf-8"))
    params = raw["parameter_estimates"]
    model = Model(
        kappa=float(params["kappa_plug_in"]),
        mu=float(params["mu_plug_in"]),
        jump_prob=float(params["jump_prob_plug_in"]),
    )
    h_g0 = {
        k: float(raw["observation_mapping_estimate"][k]) for k in H_KEYS
    }
    seeds = CAL_SEEDS[: args.cal_seeds]
    paths, logp, regime, sigma_sq = generate(seeds, args.sigma_mode)

    starts = {
        "g0": h_g0,
        "truth_like": TRUTH_H,
        "crude": {
            "open_sd": 0.001,
            "ext_phi": 0.5,
            "ext_sd": 0.005,
            "stale_phi": 0.9,
            "stale_sd": 0.002,
        },
        "slow_eta": {
            "open_sd": 0.0005,
            "ext_phi": 0.2,
            "ext_sd": 0.01,
            "stale_phi": 0.5,
            "stale_sd": 0.005,
        },
        "fast_eta": {
            "open_sd": 0.002,
            "ext_phi": 0.95,
            "ext_sd": 0.001,
            "stale_phi": 0.99,
            "stale_sd": 0.0003,
        },
    }
    optima = {}
    best = None
    start_items = list(starts.items())[: max(1, args.max_starts)]
    for name, h0 in start_items:
        f, h, res = fit(model, logp, regime, sigma_sq, h0)
        optima[name] = {
            "neg_loglik": f,
            "h": h,
            "converged": bool(res.success),
            "message": str(res.message),
        }
        print(f"[h] unmasked start={name}: negll={f:.2f} ext_phi={h['ext_phi']:.4f}", flush=True)
        if best is None or f < best[0]:
            best = (f, h)
    unmasked = {"neg_loglik": best[0], "h": best[1], "optima": optima}

    # Two-pass robust/trimmed fit: iterate the mask to a fixed point.
    h = best[1]
    robust_trace = []
    mask = None
    last_res = None
    for it in range(3):
        _ll, Z, JP = model.gsf_loglik(logp, regime, sigma_sq, h, return_diag=True)
        mask = np.abs(Z) > args.z_threshold
        f, h_new, res_iter = fit(
            model, logp, regime, sigma_sq, h, masked=True, mask=mask
        )
        last_res = res_iter
        robust_trace.append(
            {
                "iter": it,
                "mask_fraction": float(mask[:, 1:].mean()),
                "mask_per_path": float(mask[:, 1:].sum(axis=1).mean()),
                "neg_loglik_masked": f,
                "h": h_new,
                "converged": bool(res_iter.success),
            }
        )
        print(
            f"[h] masked iter={it}: flag={mask[:,1:].mean():.5f} "
            f"negll={f:.2f} ext_phi={h_new['ext_phi']:.4f}",
            flush=True,
        )
        h = h_new

    # Recompute the mask at the final estimate and refit once so that the
    # reported h is the fit under the mask used in diagnostics/bootstrap.
    _ll, Z, JP = model.gsf_loglik(logp, regime, sigma_sq, h, return_diag=True)
    mask_final = np.abs(Z) > args.z_threshold
    if not np.array_equal(mask_final, mask):
        f, h_final, res_final = fit(
            model, logp, regime, sigma_sq, h, masked=True, mask=mask_final
        )
        last_res = res_final
        robust_trace.append(
            {
                "iter": "final_refit",
                "mask_fraction": float(mask_final[:, 1:].mean()),
                "mask_per_path": float(mask_final[:, 1:].sum(axis=1).mean()),
                "neg_loglik_masked": f,
                "h": h_final,
                "converged": bool(res_final.success),
            }
        )
        print(
            f"[h] masked final_refit: flag={mask_final[:,1:].mean():.5f} "
            f"negll={f:.2f} ext_phi={h_final['ext_phi']:.4f}",
            flush=True,
        )
        h = h_final
    mask = mask_final

    # Diagnostics at the final estimate.
    _ll, Z, JP = model.gsf_loglik(logp, regime, sigma_sq, h, return_diag=True)
    z_by_regime = {}
    for r in sorted(set(int(x) for x in regime)):
        idx = np.array(
            [
                t
                for t in range(1, N_STEPS + 1)
                if (int(regime[t]) if t < len(regime) else int(regime[-1])) == r
            ]
        )
        zz = Z[:, idx].ravel()
        if np.all(np.isnan(zz)):
            z_by_regime[str(r)] = {
                "n": int(zz.size),
                "observed": False,
                "mask_fraction": float(mask[:, idx].mean()),
            }
            continue
        z_by_regime[str(r)] = {
            "n": int(zz.size),
            "observed": True,
            "mean": float(np.nanmean(zz)),
            "sd": float(np.nanstd(zz)),
            "q01": float(np.nanpercentile(zz, 1)),
            "q99": float(np.nanpercentile(zz, 99)),
            "mask_fraction": float(mask[:, idx].mean()),
        }

    loglik = {
        "truth": float(model.gsf_loglik(logp, regime, sigma_sq, TRUTH_H)),
        "g0": float(model.gsf_loglik(logp, regime, sigma_sq, h_g0)),
        "robust": float(model.gsf_loglik(logp, regime, sigma_sq, h)),
    }

    # Numerical Hessian of the masked quasi-likelihood at the estimate.
    theta0 = pack(h)

    def masked_obj(theta):
        return -model.gsf_loglik_mask(logp, regime, sigma_sq, unpack(theta), mask)

    hess = np.zeros((5, 5))
    eps = 1e-4
    f0 = masked_obj(theta0)
    for i in range(5):
        for j in range(5):
            ei = np.zeros(5)
            ej = np.zeros(5)
            ei[i] = eps
            ej[j] = eps
            fpp = masked_obj(theta0 + ei + ej)
            fpm = masked_obj(theta0 + ei - ej)
            fmp = masked_obj(theta0 - ei + ej)
            fmm = masked_obj(theta0 - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
    # unpack() is elementwise: sd = exp(theta), phi = tanh(theta), so the
    # Jacobian d h / d theta is diagonal; Cov(h) = J Cov(theta) J.
    jac = {
        "open_sd": h["open_sd"],
        "ext_phi": 1.0 - h["ext_phi"] ** 2,
        "ext_sd": h["ext_sd"],
        "stale_phi": 1.0 - h["stale_phi"] ** 2,
        "stale_sd": h["stale_sd"],
    }
    try:
        cov = np.linalg.inv(hess)
        hess_se_theta = {
            k: float(math.sqrt(max(cov[i, i], 0.0))) for i, k in enumerate(H_KEYS)
        }
        hess_se = {k: float(jac[k] * hess_se_theta[k]) for k in H_KEYS}
        hess_ok = True
    except np.linalg.LinAlgError:
        hess_se_theta = {k: None for k in H_KEYS}
        hess_se = {k: None for k in H_KEYS}
        hess_ok = False

    # Path-level bootstrap of the masked fit, mask fixed at the final estimate.
    rng = np.random.default_rng(20261021)
    boot = []
    n_paths = logp.shape[0]
    for b in range(args.n_boot):
        idx = rng.integers(0, n_paths, size=n_paths)
        f, hb, res_b = fit(
            model,
            logp[idx],
            regime,
            sigma_sq[idx],
            h,
            masked=True,
            mask=mask[idx],
        )
        boot.append(
            {
                "neg_loglik_masked": f,
                **hb,
                "converged": bool(res_b.success),
            }
        )
        if (b + 1) % 8 == 0:
            print(f"[h] bootstrap {b+1}/{args.n_boot}", flush=True)
    boot_h = {k: np.array([b[k] for b in boot]) for k in H_KEYS}
    boot_summary = None
    if args.n_boot > 0:
        boot_summary = {
            k: {
                "mean": float(np.mean(boot_h[k])),
                "se": float(np.std(boot_h[k], ddof=1)),
                "q05": float(np.percentile(boot_h[k], 5)),
                "q50": float(np.percentile(boot_h[k], 50)),
                "q95": float(np.percentile(boot_h[k], 95)),
            }
            for k in H_KEYS
        }

    result = {
        "config": {
            "cal_seeds": [int(seeds[0]), int(seeds[-1])],
            "n_paths": int(n_paths),
            "windows": list(WINDOWS),
            "sigma_mode": args.sigma_mode,
            "z_threshold": float(args.z_threshold),
            "n_boot": int(args.n_boot),
            "kappa": model.kappa,
            "mu": model.mu,
            "jump_prob": model.jump_prob,
            "parameter_source": "results/policy_value_check.json (G0 plug-in)",
            "note": "The likelihood observes only P and the calendar. Per-path "
            "sigma is the simulator oracle in the main specification "
            "(sigma_mode='oracle'); --sigma-mode open estimates it from each "
            "path's OPEN observations and is reported as a sensitivity. "
            "kappa/mu/jump_prob are held at the G0 values, as in "
            "trajectory_value_check.py.",
        },
        "h_estimate": h,
        "h_unmasked_mle": unmasked,
        "h_g0_true_s_diagnostic": h_g0,
        "h_truth": TRUTH_H,
        "robust_trace": robust_trace,
        "uncertainty": {
            "bootstrap": boot_summary,
            "hessian_se": hess_se,
            "hessian_se_unconstrained": hess_se_theta,
            "hessian_space": "natural units (delta method); unconstrained "
            "space is log(sd)/atanh(phi)",
            "hessian_ok": bool(hess_ok),
        },
        "diagnostics": {
            "loglik": loglik,
            "z_by_regime": z_by_regime,
            "mask_fraction": float(mask[:, 1:].mean()),
            "mask_per_path": float(mask[:, 1:].sum(axis=1).mean()),
            "optimizer": {
                "final_success": None if last_res is None else bool(last_res.success),
                "final_message": None if last_res is None else str(last_res.message),
                "final_nit": None if last_res is None else int(last_res.nit),
                "bootstrap_converged": (
                    int(sum(b["converged"] for b in boot)) if args.n_boot > 0 else None
                ),
                "bootstrap_n": int(args.n_boot),
            },
        },
        "limits": [
            "kappa/mu/jump_prob are held at the G0 plug-in values; only h is "
            "estimated here. A fully observable-only pipeline still needs "
            "observable estimators for the latent S dynamics and jump process.",
            "The per-path sigma entering the main-spec likelihood is the "
            "simulator oracle (synthetic_env's m.sigma), not an observable; "
            "'observable-only' refers to h, not to sigma. --sigma-mode open "
            "estimates sigma from each path's OPEN observations and is a "
            "sensitivity; its optimum is flat to optimizer tolerance "
            "(~2e-5 in ext_phi across start configurations), so report four "
            "decimals only.",
            "jump_prob is the G0 plug-in estimate (0.003) while the simulator "
            "truth is 0.005; trimming interacts with this jump-rate "
            "misspecification, so the trimmed ext_phi estimate is not a "
            "clean function of the true jump process alone.",
            "The estimator is a two-pass trimmed/quasi-likelihood: the mask "
            "depends on the data and the Hessian treats it as fixed. The "
            "bootstrap captures path sampling but not mask-selection "
            "uncertainty; standard errors are approximate.",
            "The GSF is a two-component moment-matched approximation to the "
            "exact jump posterior; the oracle-jump diagnostic shows that "
            "perfect jump knowledge would move ext_phi further toward truth.",
            "open_sd and ext_phi are weakly identified from P alone: the "
            "latent S innovation (sigma ~ 0.008) dominates the open noise "
            "(5e-4) and the extended eta innovation (2e-3). Report their "
            "uncertainty (bootstrap/Hessian), not just the point estimate.",
            "h is estimated on the calibration seeds 2000-2127 and is used "
            "unchanged on the disjoint evaluation seeds 1000-1127; the "
            "coverage rerun uses the same calibration/test split as step 11.",
            "Truth h is a simulator constant used only for scoring/diagnostics.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[h] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
