#!/usr/bin/env python3
"""Real-path replay for P1 S2.5 (case sample, non-C1).

This script runs coverage, width, transfer, and sensitivity analyses for the S2.5 slice. It reads the
normalized Uniswap-v3 Swap tables and the per-event diagnostics produced by
``build_real_slice.py``, reconstructs an hourly pool-price path for every
A-layer event, replays the two target policies on the *actual* path (the
replay ground truth), and compares two model-based OPE pipelines:

* ``identified``: latent reference price ``S`` plus a regime-dependent
  observation map ``P = S * exp(eta)``; in the oracle-stale interval ``P`` is
  anchored to the latent price at the stale start with an AR(1) deviation.
* ``naive``: the observed pool price ``P`` is treated as the state and evolved
  as a random walk, with no observation map and no freeze anchor.

The truth is the deterministic replay of the target policy on the observed
path with a **P-marked reward** (fee income + change in pool-marked IL - cost).
This is not a deployed-policy counterfactual and not a causal logged OPE
estimate; see ``drafts/real-data-slice-plan.md`` S2.2/S5.3.

Outputs
-------
``results/real_data_replay.json``
    Parameters, per-event truth/estimates, E1 conformal coverage, E2 branch
    coverage, per-fold calibration support, transfer diagnostics, and E5 sensitivities.
``paper/tables/table12-real-data.md``
    Main table draft (generated with ``--table``).

Usage
-----
    ../../.venv-c1/bin/python real_data_replay.py \
        --case-events ../results/real_slice_case_events.csv \
        --event-json-dir ../results/real_slice_9m \
        --swaps-dir ../data/raw/ethereum \
        --draws 256 --table
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_env import (  # noqa: E402
    GAS_USD,
    LIQUIDITY_SHARE,
    NARROW,
    REGIME_EXTENDED,
    REGIME_HOLIDAY,
    REGIME_OPEN,
    REGIME_WEEKEND,
    SLIPPAGE,
    V0,
    WIDE,
    Position,
)
from policy_value_check import (  # noqa: E402
    apply_action,
    policy_narrow,
    policy_wide,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "results" / "real_slice_case_events.csv"
DEFAULT_EVENT_JSON = ROOT / "results" / "real_slice_9m"
DEFAULT_SWAPS_DIR = Path(os.environ.get("P1_DATA_ROOT", ROOT / "data" / "raw")) / "ethereum"
DEFAULT_OUT = ROOT / "results" / "real_data_replay.json"
DEFAULT_TABLE = ROOT / "paper" / "tables" / "table12-real-data.md"
DEFAULT_SUPPORT_TABLE = ROOT / "paper" / "tables" / "table15-real-calibration-support.md"

# Real regime labels.  CLOSED_LIVE keeps its own observation branch
# (sigma_closed) and maps to the synthetic EXTENDED policy regime; it is not
# merged into OPEN.  Its observation scale is derived from open_sd via the
# closed/open volatility ratio rather than estimated independently.
R_OPEN = "OPEN"
R_CLOSED = "CLOSED_LIVE"
R_FREEZE = "FREEZE"

POLICY_REGIME = {
    R_OPEN: REGIME_OPEN,
    R_CLOSED: REGIME_EXTENDED,
    R_FREEZE: REGIME_WEEKEND,
}

SYMBOL_FEE = {"TSLAon": 0.01, "NVDAon": 0.003, "SPYon": 0.003}
SYMBOL_QUOTE_DECIMALS = {"TSLAon": 6, "NVDAon": 6, "SPYon": 6}

POLICIES = {
    "Wide_Rebalance_2pct": policy_wide,
    "Narrow_Rebalance_1pct": policy_narrow,
}

TARGETS = ("total_main", "total_long", "event", "freeze", "open", "closed", "post2h", "post24h")
SCORES = ("raw", "studentized", "cqr")
Z90 = 1.6448536269514722
EPS = 1e-12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class EventData:
    event_id: str
    symbol: str
    type: str
    date: str
    freeze_start: pd.Timestamp
    freeze_end: pd.Timestamp
    stale_start: pd.Timestamp
    stale_open: pd.Timestamp
    stale_value: float
    fee_rate: float
    thin: bool
    n_swaps_window: int
    n_swaps_used: int
    times: np.ndarray
    p: np.ndarray
    volume: np.ndarray
    regime: np.ndarray
    policy_regime: np.ndarray
    masks: dict


_EASTERN = ZoneInfo("America/New_York")


def _is_market_open(t: pd.Timestamp) -> bool:
    """Regular US equity session in Eastern time, DST-aware.

    The liquid-regime case is entirely US EDT, where 09:30--16:00 ET is
    13:30--20:00 UTC.  The 9-month robustness sample also covers EST, so the
    conversion must follow America/New_York rather than a fixed UTC offset.
    """
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    et = t.tz_convert(_EASTERN)
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return 570 <= minutes < 960  # 09:30-16:00 ET


def _quote_amount(df: pd.DataFrame, quote_is_token0: bool,
                  quote_decimals: int) -> pd.Series:
    col = "amount0" if quote_is_token0 else "amount1"
    raw = pd.to_numeric(df[col], errors="coerce").abs()
    return raw / (10.0 ** quote_decimals)


def load_swaps(path: Path, quote_is_token0: bool, quote_decimals: int,
               price_col: str, ref_price: float
               ) -> tuple[pd.DataFrame, dict]:
    """Load one symbol's normalized swaps and apply the data-quality filter.

    A handful of swaps in the raw tables have zero/empty liquidity and an
    extreme ``sqrtPriceX96`` (tick at MIN/MAX).  They are not usable pool-price
    observations and would destroy a forward-filled path.  Keep swaps with
    positive liquidity and a price within [0.5, 2.0] x the table/event
    reference.  Same-timestamp swaps are collapsed to the last one because the
    replay uses an hourly step-function price.
    """
    df = pd.read_csv(path, parse_dates=["block_time_utc"])
    df["pool_price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["liq"] = pd.to_numeric(df["liquidity"], errors="coerce")
    df["quote_amount"] = _quote_amount(df, quote_is_token0, quote_decimals)
    med = float(df["pool_price"].median())
    ref = ref_price if math.isfinite(ref_price) and ref_price > 0 else med
    lo, hi = 0.5 * min(med, ref), 2.0 * max(med, ref)
    ok = (
        df["pool_price"].notna()
        & (df["pool_price"] > 0)
        & (df["liq"] > 0)
        & (df["pool_price"] >= lo)
        & (df["pool_price"] <= hi)
    )
    out = df.loc[ok, ["block_time_utc", "pool_price", "quote_amount"]].copy()
    out = out.sort_values("block_time_utc")
    stats = {
        "n_rows_total": int(len(df)),
        "n_rows_price_liquidity_ok": int(len(out)),
        "n_rows_bad": int(len(df) - len(out)),
        "n_rows_used": int(len(out)),
        "price_band": [float(lo), float(hi)],
    }
    return out, stats


def build_event(row: pd.Series, ev: dict, swaps: pd.DataFrame,
                pre_hours: float, post_hours: float) -> EventData | None:
    eid = str(row["event_id"])
    symbol = str(row["symbol"])
    etype = str(row["type"])
    fs = pd.Timestamp(ev["event"]["freeze_start_utc"])
    fe = pd.Timestamp(ev["event"]["freeze_end_utc"])
    gap = ev.get("oracle", {}).get("longest_gap") or {}
    if gap:
        stale_start = pd.Timestamp(gap["stale_start_utc"])
        stale_open = pd.Timestamp(gap["open_time_utc"])
        stale_value = float(gap["stale_value"])
    else:
        # Fallback: treat the calendar freeze as the stale interval.
        stale_start, stale_open = fs, fe
        anchor = ev.get("oracle", {}).get("last_before_freeze") or {}
        answer = anchor.get("answer")
        stale_value = float(answer) if answer not in (None, "") else float("nan")
    if not math.isfinite(stale_value) or stale_value <= 0:
        return None

    start = (fs - pd.Timedelta(hours=pre_hours)).floor("h")
    end = (fe + pd.Timedelta(hours=max(post_hours, 2.0))).ceil("h")
    grid = pd.date_range(start, end, freq="h", tz="UTC")
    if len(grid) < 4:
        return None

    sub = swaps[(swaps["block_time_utc"] >= start - pd.Timedelta(hours=2))
                & (swaps["block_time_utc"] <= end + pd.Timedelta(hours=2))]
    if len(sub) == 0:
        return None

    # Step-function price: last swap at or before each grid point.  Volume
    # uses all swaps in the hour; only the price series is deduplicated.
    price_rows = sub.sort_values("block_time_utc").drop_duplicates(
        "block_time_utc", keep="last"
    )
    idx = price_rows.set_index("block_time_utc")["pool_price"].sort_index()
    px = idx.reindex(grid, method="ffill")
    if px.isna().iloc[0]:
        before = swaps[swaps["block_time_utc"] < start]
        if len(before):
            fill = float(before["pool_price"].iloc[-1])
        else:
            pool_ref = ev["pool"].get("price_at_freeze_start")
            fill = float(pool_ref) if pool_ref not in (None, "") else stale_value
        px.iloc[0] = fill
    px = px.ffill().bfill()
    if px.isna().any():
        return None

    # Volume in the hour starting at each grid point.
    vol = (
        sub.groupby(sub["block_time_utc"].dt.floor("h"))["quote_amount"]
        .sum()
        .reindex(grid[:-1], fill_value=0.0)
        .to_numpy(dtype=float)
    )

    tarr = grid[:-1]
    regime = np.empty(len(tarr), dtype=object)
    for i, t in enumerate(tarr):
        if stale_start <= t < stale_open:
            regime[i] = R_FREEZE
        elif fs <= t < fe:
            regime[i] = R_CLOSED
        elif _is_market_open(t):
            regime[i] = R_OPEN
        else:
            regime[i] = R_CLOSED
    policy_regime = np.array(
        [POLICY_REGIME[str(r)] for r in regime], dtype=int
    )

    t = tarr
    masks = {
        "total_main": t < fe + pd.Timedelta(hours=2),
        "total_long": t < fe + pd.Timedelta(hours=24),
        "event": (t >= fs) & (t < fe + pd.Timedelta(hours=2)),
        "freeze": regime == R_FREEZE,
        "open": regime == R_OPEN,
        "closed": regime == R_CLOSED,
        "post2h": (t >= fe) & (t < fe + pd.Timedelta(hours=2)),
        "post24h": (t >= fe) & (t < fe + pd.Timedelta(hours=24)),
    }

    thin = bool(row.get("thin", False)) if "thin" in row else False
    return EventData(
        event_id=eid,
        symbol=symbol,
        type=etype,
        date=fs.date().isoformat(),
        freeze_start=fs,
        freeze_end=fe,
        stale_start=stale_start,
        stale_open=stale_open,
        stale_value=stale_value,
        fee_rate=float(SYMBOL_FEE.get(symbol, 0.003)),
        thin=thin,
        n_swaps_window=int(len(sub)),
        n_swaps_used=int(len(sub)),
        times=grid,
        p=px.to_numpy(dtype=float),
        volume=vol,
        regime=regime,
        policy_regime=policy_regime,
        masks=masks,
    )


# ---------------------------------------------------------------------------
# Replay (ground truth)
# ---------------------------------------------------------------------------


def rollout_rewards(p: np.ndarray, volume: np.ndarray,
                    policy_regime: np.ndarray, policy,
                    fee_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Replay one policy on a fixed price path; P-marked reward.

    ``p`` has length n+1, ``volume`` and ``policy_regime`` length n.  ``p[t]``
    is the observed pool price at the decision time; ``p[t+1]`` is the next
    grid point.  The reward is fee income + change in pool-marked IL - cost,
    normalized by V0, exactly mirroring ``synthetic_env.simulate_strategy``
    with ``il_true`` replaced by ``il`` at the observable P.
    """
    pos = Position()
    prev_il = 0.0
    n = len(volume)
    rewards = np.empty(n, dtype=float)
    actions = np.empty(n, dtype=int)
    for t in range(n):
        r = int(policy_regime[t])
        nr = int(policy_regime[t + 1]) if t + 1 < n else r
        action = int(policy(t, r, nr, float(p[t]), pos))
        cost, changed = apply_action(pos, action, float(p[t]))
        if changed:
            prev_il = 0.0
        in_range = bool(pos.in_pos and pos.lower <= p[t] <= pos.upper)
        fee = fee_rate * float(volume[t]) * LIQUIDITY_SHARE * float(in_range)
        il_now = pos.il(float(p[t + 1])) if pos.in_pos else 0.0
        il_step = il_now - prev_il
        prev_il = il_now
        rewards[t] = (fee + il_step - cost) / V0
        actions[t] = action
    return rewards, actions


def sum_targets(rewards: np.ndarray, masks: dict) -> dict:
    return {k: float(rewards[m].sum()) for k, m in masks.items()}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def simulate_path(rng: np.random.Generator, p0: float, regime: np.ndarray,
                  params: dict, model: str) -> np.ndarray:
    n = len(regime)
    p = np.empty(n + 1, dtype=float)
    p[0] = p0
    if model == "naive":
        lp = math.log(p0)
        sig = params["sigma_all"]
        for t in range(n):
            lp += sig * rng.normal()
            p[t + 1] = math.exp(lp)
        return p

    # identified: latent S + regime-dependent observation map
    ls = math.log(p0)
    anchor = p0
    eta = 0.0
    sig_open = params["sigma_open"]
    sig_closed = params["sigma_closed"]
    open_sd = params["open_sd"]
    closed_sd = params["closed_sd"]
    phi = params["stale_phi"]
    sd = params["stale_sd"]
    for t in range(n):
        r = str(regime[t])
        if r == R_OPEN:
            ls_new = ls + sig_open * rng.normal()
            eta = 0.0
            anchor = math.exp(ls_new)
            p[t + 1] = math.exp(ls_new + open_sd * rng.normal())
        elif r == R_CLOSED:
            # The oracle is live outside the freeze, so P is informative; the
            # closed market has its own (typically lower) volatility and a
            # correspondingly scaled observation noise.
            ls_new = ls + sig_closed * rng.normal()
            eta = 0.0
            p[t + 1] = math.exp(ls_new + closed_sd * rng.normal())
        elif r == R_FREEZE:
            ls_new = ls + sig_closed * rng.normal()
            if t == 0 or str(regime[t - 1]) != R_FREEZE:
                anchor = math.exp(ls_new)
                eta = 0.0
            eta = phi * eta + sd * rng.normal()
            p[t + 1] = anchor * math.exp(eta)
        else:
            raise ValueError(f"unknown regime: {r}")
        ls = ls_new
    return p


def model_draws(rng: np.random.Generator, p0: float, regime: np.ndarray,
                volume: np.ndarray, policy_regime: np.ndarray, policy,
                fee_rate: float, masks: dict, params: dict, model: str,
                n_draws: int) -> dict:
    out = {k: np.empty(n_draws, dtype=float) for k in TARGETS}
    for i in range(n_draws):
        p = simulate_path(rng, p0, regime, params, model)
        rew, _ = rollout_rewards(p, volume, policy_regime, policy, fee_rate)
        vals = sum_targets(rew, masks)
        for k in TARGETS:
            out[k][i] = vals[k]
    return out


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def estimate_params(events: list[EventData], case: pd.DataFrame) -> dict:
    """Estimate per-symbol observation-map and volatility parameters.

    ``open_sd`` uses the event-level boundary basis (stale start and market
    open), not a full oracle-round path; this is a first-pass estimate and is
    flagged in the output.  ``stale_phi``/``stale_sd`` use the hourly pool
    price inside the oracle stale interval relative to the stale value.
    """
    basis = case.copy()
    basis["log_basis_stale"] = np.log1p(pd.to_numeric(
        basis["basis_stale"], errors="coerce"
    ))
    basis["log_basis_open"] = np.log1p(pd.to_numeric(
        basis["basis_open"], errors="coerce"
    ))

    params: dict[str, dict] = {}
    for sym in sorted({e.symbol for e in events}):
        evs = [e for e in events if e.symbol == sym]
        open_rets: list[float] = []
        closed_rets: list[float] = []
        all_rets: list[float] = []
        stale_pairs: list[tuple[float, float]] = []
        for e in evs:
            lp = np.log(e.p)
            ret = np.diff(lp)
            for t in range(len(ret) - 1):
                if e.regime[t] == R_OPEN and e.regime[t + 1] == R_OPEN:
                    open_rets.append(float(ret[t]))
                if e.regime[t] == R_CLOSED and e.regime[t + 1] == R_CLOSED:
                    closed_rets.append(float(ret[t]))
                all_rets.append(float(ret[t]))
            # AR(1) pairs inside the oracle stale interval.
            mask = e.regime == R_FREEZE
            idx = np.flatnonzero(mask)
            if len(idx) >= 2:
                e_series = np.log(e.p[idx] / e.stale_value)
                stale_pairs.extend(
                    zip(e_series[:-1].tolist(), e_series[1:].tolist())
                )

        b = basis[basis["symbol"] == sym]
        bvals = np.concatenate([
            b["log_basis_stale"].dropna().to_numpy(dtype=float),
            b["log_basis_open"].dropna().to_numpy(dtype=float),
        ]) if len(b) else np.array([], dtype=float)
        open_sd = float(np.std(bvals, ddof=1)) if len(bvals) > 1 else 5e-4
        open_sd = max(open_sd, 1e-5)

        sigma_open = float(np.std(open_rets, ddof=1)) if len(open_rets) > 1 else (
            float(np.std(all_rets, ddof=1)) if len(all_rets) > 1 else 0.005
        )
        sigma_closed = float(np.std(closed_rets, ddof=1)) if len(closed_rets) > 1 else sigma_open
        sigma_all = float(np.std(all_rets, ddof=1)) if len(all_rets) > 1 else sigma_open
        sigma_open = max(sigma_open, 1e-5)
        sigma_closed = max(sigma_closed, 1e-5)
        sigma_all = max(sigma_all, 1e-5)
        # The CLOSED_LIVE observation noise is scaled from the open-window
        # basis, because the pool price is informative whenever the oracle is
        # live but the closed market has lower volatility.
        closed_sd = open_sd * float(np.clip(sigma_closed / sigma_open, 0.2, 1.0))
        closed_sd = max(closed_sd, 1e-5)

        if len(stale_pairs) >= 4:
            x = np.array([a for a, _ in stale_pairs], dtype=float)
            y = np.array([b_ for _, b_ in stale_pairs], dtype=float)
            denom = float(np.sum(x * x))
            phi = float(np.sum(x * y) / denom) if denom > 0 else 0.0
            phi = float(np.clip(phi, 0.0, 0.999))
            resid = y - phi * x
            stale_sd = float(np.std(resid, ddof=1))
        else:
            phi, stale_sd = 0.0, 1e-4
        stale_sd = max(stale_sd, 1e-5)

        params[sym] = {
            "sigma_open": sigma_open,
            "sigma_closed": sigma_closed,
            "sigma_all": sigma_all,
            "open_sd": open_sd,
            "closed_sd": closed_sd,
            "stale_phi": phi,
            "stale_sd": stale_sd,
            "n_open_returns": len(open_rets),
            "n_closed_returns": len(closed_rets),
            "n_stale_pairs": len(stale_pairs),
            "n_basis": int(len(bvals)),
        }
    return params


# ---------------------------------------------------------------------------
# Conformal
# ---------------------------------------------------------------------------


def conformal_radius(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def score_one(y: float, mu: float, sd: float, q05: float, q95: float,
              score: str) -> float:
    if score == "raw":
        return abs(y - mu)
    if score == "studentized":
        return abs(y - mu) / max(sd, EPS)
    if score == "cqr":
        return max(q05 - y, y - q95)
    raise ValueError(score)


def width_one(sd: float, q05: float, q95: float, radius: float,
              score: str) -> float:
    if score == "raw":
        return 2.0 * radius
    if score == "studentized":
        return 2.0 * radius * max(sd, EPS)
    if score == "cqr":
        return max(q95 - q05, 0.0) + 2.0 * radius
    raise ValueError(score)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _folds(events: list[dict], key) -> list[tuple[str, list[dict], list[dict]]]:
    groups = sorted({str(key(e)) for e in events})
    out = []
    for g in groups:
        test = [e for e in events if str(key(e)) == g]
        cal = [e for e in events if str(key(e)) != g]
        out.append((g, test, cal))
    return out


def _summary(records: list[dict], cluster_key: str, n_boot: int = 2000,
             seed: int = 0) -> dict:
    cov = np.array([r["covered"] for r in records], dtype=float)
    wid = np.array([r["width"] for r in records], dtype=float)
    bias = np.array([r["bias"] for r in records], dtype=float)
    clusters = np.array([r[cluster_key] for r in records])
    uniq = sorted(set(clusters.tolist()))
    rng = np.random.default_rng(seed)
    boot_cov, boot_wid = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.concatenate([np.flatnonzero(clusters == p) for p in pick])
        boot_cov.append(float(cov[mask].mean()))
        boot_wid.append(float(wid[mask].mean()))
    return {
        "n": int(len(records)),
        "coverage": float(cov.mean()),
        "coverage_ci95": [
            float(np.quantile(boot_cov, 0.025)),
            float(np.quantile(boot_cov, 0.975)),
        ],
        "width": float(wid.mean()),
        "width_ci95": [
            float(np.quantile(boot_wid, 0.025)),
            float(np.quantile(boot_wid, 0.975)),
        ],
        "bias": float(bias.mean()),
        "mae": float(np.abs(bias).mean()),
    }


def _eval_method(events: list[dict], model: str, score: str, target: str,
                 fold_key, alpha: float, cluster_key: str = "date") -> dict:
    records = []
    cal_scores_all, test_scores_all = [], []
    for _g, test, cal in _folds(events, fold_key):
        cal_scores = np.array([
            e["models"][model][target]["score"][score] for e in cal
        ], dtype=float)
        radius = conformal_radius(cal_scores, alpha)
        cal_scores_all.extend(cal_scores.tolist())
        for e in test:
            rec = e["models"][model][target]
            s = rec["score"][score]
            test_scores_all.append(s)
            records.append({
                "event_id": e["event_id"],
                "symbol": e["symbol"],
                "type": e["type"],
                "date": e["date"],
                "thin": e["thin"],
                "cluster": e[cluster_key],
                "covered": bool(s <= radius),
                "width": float(width_one(
                    rec["sd"], rec["q05"], rec["q95"], radius, score
                )),
                "radius": float(radius),
                "bias": float(rec["mu"] - rec["truth"]),
                "score": float(s),
            })
    out = _summary(records, cluster_key)
    out["kappa_conf_cal_test"] = ks_distance(
        np.array(cal_scores_all), np.array(test_scores_all)
    )
    out["coverage_lower_bound"] = max(
        0.0, 1.0 - alpha - out["kappa_conf_cal_test"]
    )

    def _max_pairwise_ks(groups: dict[str, list[float]]) -> float:
        keys = sorted(groups)
        best = 0.0
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                best = max(best, ks_distance(
                    np.array(groups[a], dtype=float),
                    np.array(groups[b], dtype=float),
                ))
        return best if keys else float("nan")

    out["kappa_type"] = _max_pairwise_ks({
        t: [r["score"] for r in records if r["type"] == t]
        for t in sorted({r["type"] for r in records})
    })
    out["kappa_symbol"] = _max_pairwise_ks({
        s: [r["score"] for r in records if r["symbol"] == s]
        for s in sorted({r["symbol"] for r in records})
    })
    out["coverage_lower_bound_type_shift"] = max(
        0.0, 1.0 - alpha - out["kappa_type"]
    )
    for sym in sorted({r["symbol"] for r in records}):
        sub = [r for r in records if r["symbol"] == sym]
        out.setdefault("per_symbol", {})[sym] = {
            "n": len(sub),
            "coverage": float(np.mean([r["covered"] for r in sub])),
        }
    for typ in sorted({r["type"] for r in records}):
        sub = [r for r in records if r["type"] == typ]
        out.setdefault("per_type", {})[typ] = {
            "n": len(sub),
            "coverage": float(np.mean([r["covered"] for r in sub])),
        }
    thin = [r for r in records if r["thin"]]
    nonthin = [r for r in records if not r["thin"]]
    out["thin_coverage"] = (
        float(np.mean([r["covered"] for r in thin])) if thin else None
    )
    out["nonthin_coverage"] = (
        float(np.mean([r["covered"] for r in nonthin])) if nonthin else None
    )
    return out


def _eval_plugin(events: list[dict], model: str, target: str,
                 cluster_key: str = "date") -> dict:
    records = []
    for e in events:
        rec = e["models"][model][target]
        half = Z90 * rec["sd"]
        records.append({
            "event_id": e["event_id"],
            "symbol": e["symbol"],
            "type": e["type"],
            "date": e["date"],
            "thin": e["thin"],
            "cluster": e[cluster_key],
            "covered": bool(abs(rec["truth"] - rec["mu"]) <= half),
            "width": float(2.0 * half),
            "radius": float(half),
            "bias": float(rec["mu"] - rec["truth"]),
            "score": float(abs(rec["truth"] - rec["mu"])),
        })
    return _summary(records, cluster_key)


# ---------------------------------------------------------------------------
# E2 / E3
# ---------------------------------------------------------------------------


def eval_e2(events: list[dict], model: str, alpha: float) -> dict:
    """Branch-conditional coverage of the post-freeze open decision.

    Branch B = 1 when the target policy rebalances at any step in the 2h
    post-open window; B is observed at evaluation but unavailable when the
    interval is formed.  A finite 90% split-conformal branch radius needs at
    least ``(1-alpha)/alpha = 9`` calibration events in the branch; with 7
    Wide rebalances the union interval is reported as not identifiable rather
    than as an infinite width.
    """
    min_cal = int(math.ceil((1.0 - alpha) / alpha))
    records = []
    for _g, test, cal in _folds(events, lambda e: e["date"]):
        cal0 = [e["models"][model]["total_main"]["score"]["raw"]
                for e in cal if e["branch_policy"] == 0]
        cal1 = [e["models"][model]["total_main"]["score"]["raw"]
                for e in cal if e["branch_policy"] == 1]
        pooled = [e["models"][model]["total_main"]["score"]["raw"] for e in cal]
        r_pool = conformal_radius(np.array(pooled), alpha)
        r0 = (conformal_radius(np.array(cal0), alpha)
              if len(cal0) >= min_cal else r_pool)
        r1 = (conformal_radius(np.array(cal1), alpha)
              if len(cal1) >= min_cal else r_pool)
        union_ok = math.isfinite(r0) and math.isfinite(r1)
        r_union = max(r0, r1) if union_ok else None
        for e in test:
            rec = e["models"][model]["total_main"]
            s = rec["score"]["raw"]
            branch = int(e["branch_policy"])
            rb = r1 if branch == 1 else r0
            records.append({
                "event_id": e["event_id"],
                "date": e["date"],
                "branch": branch,
                "covered_union": (bool(s <= r_union)
                                  if r_union is not None else None),
                "covered_marginal": bool(s <= r_pool),
                "covered_branch": bool(s <= rb) if math.isfinite(rb) else None,
                "width_union": (2.0 * r_union if r_union is not None else None),
                "width_marginal": 2.0 * r_pool,
            })
    n0 = sum(1 for e in events if e["branch_policy"] == 0)
    n1 = sum(1 for e in events if e["branch_policy"] == 1)
    n_first = sum(1 for e in events if e.get("branch_first_open_policy", 0) == 1)
    cov_m = float(np.mean([r["covered_marginal"] for r in records]))
    cov_b = [r["covered_branch"] for r in records if r["covered_branch"] is not None]
    cov_u = [r["covered_union"] for r in records if r["covered_union"] is not None]
    w_m = float(np.mean([r["width_marginal"] for r in records]))
    w_u = [r["width_union"] for r in records if r["width_union"] is not None]
    out = {
        "n_branch0": n0,
        "n_branch1": n1,
        "n_branch1_first_open": n_first,
        "min_branch_calibration": min_cal,
        "underpowered": bool(min(n0, n1) < min_cal),
        "marginal_coverage": cov_m,
        "branch_coverage": (float(np.mean(cov_b)) if cov_b else None),
        "union_coverage": (float(np.mean(cov_u)) if cov_u else None),
        "branch0_coverage": (
            float(np.mean([r["covered_branch"] for r in records
                           if r["branch"] == 0 and r["covered_branch"] is not None]))
            if any(r["branch"] == 0 and r["covered_branch"] is not None
                   for r in records) else None
        ),
        "branch1_coverage": (
            float(np.mean([r["covered_branch"] for r in records
                           if r["branch"] == 1 and r["covered_branch"] is not None]))
            if any(r["branch"] == 1 and r["covered_branch"] is not None
                   for r in records) else None
        ),
        "width_union": (float(np.mean(w_u)) if w_u else None),
        "width_marginal": w_m,
        "width_ratio": (
            float(np.mean(w_u) / w_m) if w_u and w_m > 0 else None
        ),
    }
    if out["underpowered"]:
        # The branch-specific radius fell back to the pooled radius in every
        # fold; a 90% branch-conditional interval is not identifiable.
        out["union_coverage"] = None
        out["width_union"] = None
        out["width_ratio"] = None
        out["union_note"] = (
            "branch-conditional 90% interval not identifiable; "
            "branch_coverage uses the pooled fallback for the small branch"
        )
    return out


def eval_e3(events: list[dict], model: str, score: str, alpha: float) -> dict:
    """LOSO, leave-one-type-out, and in-sample short->long target shift (raw)."""
    loso = {}
    for sym, test, cal in _folds(events, lambda e: e["symbol"]):
        cs = np.array([e["models"][model]["total_main"]["score"][score]
                       for e in cal], dtype=float)
        r = conformal_radius(cs, alpha)
        recs = []
        for e in test:
            rec = e["models"][model]["total_main"]
            s = rec["score"][score]
            recs.append({
                "covered": bool(s <= r),
                "width": width_one(rec["sd"], rec["q05"], rec["q95"], r, score),
            })
        loso[sym] = {
            "n": len(recs),
            "coverage": float(np.mean([x["covered"] for x in recs])),
            "width": float(np.mean([x["width"] for x in recs])),
        }
    loro = {}
    for typ, test, cal in _folds(events, lambda e: e["type"]):
        cs = np.array([e["models"][model]["total_main"]["score"][score]
                       for e in cal], dtype=float)
        r = conformal_radius(cs, alpha)
        recs = []
        for e in test:
            rec = e["models"][model]["total_main"]
            s = rec["score"][score]
            recs.append(bool(s <= r))
        loro[typ] = {"n": len(recs), "coverage": float(np.mean(recs))}

    # In-sample short (2h post-open) -> long (24h post-open) target shift.
    transfer = {}
    transfer_by_symbol = {}
    for src, dst in (("post2h", "post24h"), ("post24h", "post2h")):
        key = f"{src}_to_{dst}"
        cs = np.array([e["models"][model][src]["score"]["raw"]
                       for e in events], dtype=float)
        r = conformal_radius(cs, alpha)
        recs = []
        for e in events:
            s = e["models"][model][dst]["score"]["raw"]
            recs.append(bool(s <= r))
        transfer[key] = {
            "n": len(recs),
            "n_covered": sum(recs),
            "coverage": float(np.mean(recs)),
            "radius": float(r),
        }
        transfer_by_symbol[key] = {}
        for sym in sorted({e["symbol"] for e in events}):
            sym_events = [e for e in events if e["symbol"] == sym]
            covered = sum(
                e["models"][model][dst]["score"]["raw"] <= r
                for e in sym_events
            )
            transfer_by_symbol[key][sym] = {
                "n": len(sym_events),
                "n_covered": covered,
                "coverage": covered / len(sym_events),
            }
    return {
        "loso": loso,
        "loro": loro,
        "duration_transfer": transfer,
        "duration_transfer_per_symbol": transfer_by_symbol,
    }


def eval_symbol_conditional(events: list[dict], model: str, score: str,
                            alpha: float) -> dict:
    """Within-symbol leave-one-date-out conformal (practical LOSO fix).

    Pooling symbols and holding one symbol out fails when the symbol-level
    reward scale differs (TSLA's 1% fee tier is the clearest case).  This
    diagnostic calibrates each symbol on its own other dates; it is the
    symbol-Mondrian variant a practitioner would use.
    """
    per_symbol = {}
    for sym in sorted({e["symbol"] for e in events}):
        evs = [e for e in events if e["symbol"] == sym]
        recs = []
        for e in evs:
            cal = [x for x in evs if x["date"] != e["date"]]
            if len(cal) < 2:
                continue
            cs = np.array([x["models"][model]["total_main"]["score"][score]
                           for x in cal], dtype=float)
            r = conformal_radius(cs, alpha)
            if not math.isfinite(r):
                continue
            rec = e["models"][model]["total_main"]
            s = rec["score"][score]
            recs.append({
                "covered": bool(s <= r),
                "width": width_one(rec["sd"], rec["q05"], rec["q95"], r, score),
            })
        if recs:
            per_symbol[sym] = {
                "n": len(recs),
                "coverage": float(np.mean([x["covered"] for x in recs])),
                "width": float(np.mean([x["width"] for x in recs])),
            }
    all_recs = [x for v in per_symbol.values() for x in [v]]
    return {
        "per_symbol": per_symbol,
        "n_symbols": len(per_symbol),
        "macro_coverage": (
            float(np.mean([v["coverage"] for v in all_recs]))
            if all_recs else None
        ),
        "note": "Symbols are skipped when a leave-one-date-out fold is too small for a 90% radius.",
    }



# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def calibration_support(events: list[dict], alpha: float) -> dict:
    min_cal = int(math.ceil((1.0 - alpha) / alpha))
    rows = []
    for date, test, cal in _folds(events, lambda e: e["date"]):
        row = {
            "date": date,
            "type": test[0]["type"],
            "test_n": len(test),
            "test_symbols": dict(collections.Counter(e["symbol"] for e in test)),
            "test_freeze_hours": sorted({int(e["n_steps"]) for e in test}),
            "cal_n": len(cal),
            "cal_episodes": len({e["date"] for e in cal}),
            "cal_symbols": {
                sym: sum(e["symbol"] == sym for e in cal)
                for sym in ("SPYon", "NVDAon", "TSLAon")
            },
            "cal_freeze_hours_min": min(int(e["n_steps"]) for e in cal),
            "cal_freeze_hours_median": float(np.median([e["n_steps"] for e in cal])),
            "cal_freeze_hours_max": max(int(e["n_steps"]) for e in cal),
        }
        for label, pname in (("wide", "Wide_Rebalance_2pct"), ("narrow", "Narrow_Rebalance_1pct")):
            counts = collections.Counter(e["branch"][pname] for e in cal)
            row[f"{label}_branch0_cal"] = counts[0]
            row[f"{label}_branch1_cal"] = counts[1]
            row[f"{label}_finite_90"] = min(counts.values()) >= min_cal
        rows.append(row)
    return {
        "min_branch_calibration": min_cal,
        "count_definition": "Observed event counts, not an effective-sample-size estimate; symbol-events within a freeze episode remain clustered by date.",
        "folds": rows,
        "cal_events_range": [min(r["cal_n"] for r in rows), max(r["cal_n"] for r in rows)],
        "wide_branch1_cal_range": [min(r["wide_branch1_cal"] for r in rows), max(r["wide_branch1_cal"] for r in rows)],
        "narrow_branch0_cal_range": [min(r["narrow_branch0_cal"] for r in rows), max(r["narrow_branch0_cal"] for r in rows)],
        "narrow_branch1_cal_range": [min(r["narrow_branch1_cal"] for r in rows), max(r["narrow_branch1_cal"] for r in rows)],
        "nvda_cal_range": [min(r["cal_symbols"]["NVDAon"] for r in rows), max(r["cal_symbols"]["NVDAon"] for r in rows)],
        "wide_underpowered_folds": sum(not r["wide_finite_90"] for r in rows),
        "narrow_underpowered_folds": sum(not r["narrow_finite_90"] for r in rows),
    }


def write_table(result: dict, path: Path, heading: str) -> None:
    s = result["sample"]
    p = result["parameters"]
    e1 = result["e1"]
    lines = []
    lines.append(f"# {heading}\n")
    lines.append(
        f"Sample: {s['n_events']} A-layer symbol-events over {s['n_dates']} "
        f"independent freeze episodes ({s['n_weekend']} weekend + "
        f"{s['n_holiday']} holiday), {s['window']}. "
        f"{'Post-hoc regime-conditional case' if s.get('kind') == 'case' else 'Robustness check (not the main case)'}, "
        f"not a pre-registered sample.\n"
    )

    # Panel A
    lines.append("## A. Observation-map parameters\n")
    lines.append("| Symbol | Events | sigma_open | sigma_closed | open_sd | stale_phi |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for sym in ("SPYon", "NVDAon", "TSLAon"):
        v = p[sym]
        n_ev = s["by_symbol"].get(sym, 0)
        lines.append(
            f"| {sym} | {n_ev} | {v['sigma_open']:.5f} | "
            f"{v['sigma_closed']:.5f} | {v['open_sd']:.5f} | "
            f"{v['stale_phi']:.3f} |"
        )
    lines.append("")

    # Panel B
    lines.append(
        "## B. Event-level conformal coverage (alpha = 0.1, "
        "leave-one-episode-out, raw score)\n"
    )
    lines.append("| Target | Identified cov. | Identified width | Identified MAE | Naive cov. | Naive width | Naive MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    target_labels = {
        "freeze": "Freeze",
        "open": "Post-open OPEN",
        "post2h": "Post-open 2 h",
        "event": "Event window",
        "total_main": "Total main",
    }
    for target, label in target_labels.items():
        a = e1[f"identified.raw.{target}"]
        b = e1[f"naive.raw.{target}"]
        lines.append(
            f"| {label} | {a['coverage']:.3f} | {a['width']:.5f} | "
            f"{a['mae']:.5f} | {b['coverage']:.3f} | {b['width']:.5f} | "
            f"{b['mae']:.5f} |"
        )
    lines.append("")
    ref = e1["identified.raw.total_main"]
    counts = {
        t: int(round(e1[f"identified.raw.{t}"]["coverage"]
                     * e1[f"identified.raw.{t}"]["n"]))
        for t in target_labels
    }
    if len(set(counts.values())) == 1:
        cov_text = f"Coverage is {next(iter(counts.values()))}/{ref['n']} in every cell"
    else:
        cov_text = "Coverage counts are " + ", ".join(
            f"{label} {counts[t]}/{ref['n']}"
            for t, label in target_labels.items()
        )
    lo, hi = ref["coverage_ci95"]
    lines.append(
        f"{cov_text}; the cluster-bootstrap 95% CI is [{lo:.2f}, {hi:.2f}] "
        "for the total and event windows. Widths and MAE are in units of $V_0$."
    )
    lines.append("")

    # Panel C
    lines.append("## C. Modeled replay reward decomposition (Wide policy, mean truth, fraction of V0)\n")
    lines.append("| Window | Fee | IL | Cost | Net |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    comp = result.get("components", {}).get("Wide_Rebalance_2pct", {})
    for target, label in (("freeze", "Freeze"), ("post2h", "Post-open 2 h"),
                          ("event", "Event window"), ("total_main", "Total main")):
        c = comp.get(target)
        if not c:
            continue
        net = c["fee"] + c["il"] - c["cost"]
        lines.append(
            f"| {label} | {c['fee']:+.6f} | {c['il']:+.6f} | "
            f"{c['cost']:+.6f} | {net:+.6f} |"
        )
    lines.append("")
    lines.append("P-marked model decomposition; not feeGrowthGlobal or deployed-LP validation.")
    lines.append("")

    # Panel D: score sensitivity
    lines.append("## D. Score sensitivity (coverage / width)\n")
    lines.append("| Target | Score | Identified | Naive |")
    lines.append("| --- | --- | ---: | ---: |")
    for target in ("freeze", "total_main"):
        for score in ("raw", "studentized", "cqr"):
            a = e1[f"identified.{score}.{target}"]
            b = e1[f"naive.{score}.{target}"]
            lines.append(
                f"| {target} | {score} | {a['coverage']:.3f} / "
                f"{a['width']:.5f} | {b['coverage']:.3f} / "
                f"{b['width']:.5f} |"
            )
    lines.append("")

    # Branch and target-shift diagnostics
    lines.append("## E. Branch and target-shift diagnostics\n")
    lines.append("| Diagnostic | Identified | Naive |")
    lines.append("| --- | ---: | ---: |")
    for pname, label in (("Wide_Rebalance_2pct", "Wide branch"),
                         ("Narrow_Rebalance_1pct", "Narrow branch")):
        a = result["e2"][pname]["identified"]
        b = result["e2"][pname]["naive"]
        if a["underpowered"]:
            atxt = f"B=1 n={a['n_branch1']}; union n.i."
            btxt = f"B=1 n={b['n_branch1']}; union n.i."
        else:
            atxt = (f"B=1 n={a['n_branch1']}; union {a['union_coverage']:.3f}; "
                    f"width x{a['width_ratio']:.2f}")
            btxt = (f"B=1 n={b['n_branch1']}; union {b['union_coverage']:.3f}; "
                    f"width x{b['width_ratio']:.2f}")
        lines.append(f"| {label} | {atxt} | {btxt} |")

    def _e3(model: str, score: str, key: str, group: str = "TSLAon") -> float:
        block = result["e3"][model][score]
        if key in ("loso", "loro"):
            return block[key][group]["coverage"]
        if key == "sym":
            return block["symbol_conditional"]["macro_coverage"]
        return block["duration_transfer"]["post2h_to_post24h"]["coverage"]

    for score in ("raw", "cqr"):
        symbols = ("TSLAon", "SPYon") if s.get("kind") != "case" else ("TSLAon",)
        for sym in symbols:
            lines.append(
                f"| Leave-one-symbol-out {score.upper()} ({sym.removesuffix('on')}) | "
                f"{_e3('identified', score, 'loso', sym):.3f} | "
                f"{_e3('naive', score, 'loso', sym):.3f} |"
            )
    if s.get("kind") != "case":
        for typ in ("weekend", "holiday"):
            lines.append(
                f"| Leave-one-regime-out raw ({typ}) | {_e3('identified', 'raw', 'loro', typ):.3f} | "
                f"{_e3('naive', 'raw', 'loro', typ):.3f} |"
            )
    lines.append(
        f"| Within-symbol (SPY/TSLA macro) | "
        f"{_e3('identified', 'raw', 'sym'):.3f} | "
        f"{_e3('naive', 'raw', 'sym'):.3f} |"
    )
    lines.append(
        f"| In-sample 2h->24h target shift | "
        f"{_e3('identified', 'raw', 'transfer'):.3f} | "
        f"{_e3('naive', 'raw', 'transfer'):.3f} |"
    )
    if s.get("kind") != "case":
        for sym in ("SPYon", "TSLAon", "NVDAon"):
            def _duration_cell(model: str) -> str:
                v = result["e3"][model]["raw"]["duration_transfer_per_symbol"][
                    "post2h_to_post24h"][sym]
                return f"{v['n_covered']}/{v['n']} ({v['coverage']:.3f})"

            lines.append(
                f"| 2h->24h {sym.removesuffix('on')} | "
                f"{_duration_cell('identified')} | {_duration_cell('naive')} |"
            )
    lines.append("")
    lines.append(
        "n.i. = not identifiable at 90% (fewer than nine branch calibration "
        "events). The symbol-conditional row is a macro average over SPY and "
        f"TSLA; NVDA has {s['by_symbol'].get('NVDAon', 0)} events and is "
        "excluded from a 90% within-symbol radius. Duration rows are in-sample "
        "target shifts over the same events, not out-of-sample transfer; "
        "reverse 1.000 reflects the wider 24h radius, not a symmetric guarantee."
    )
    lines.append("")

    lines.append("## Notes\n")
    for c in result["caveats"]:
        lines.append(f"- {c}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_support_table(result: dict, path: Path) -> None:
    support = result["calibration_support"]
    lines = [
        "# Table 15 — Leave-one-episode-out calibration support\n",
        "| Episode | Type | Test n | Test symbols | Test h | Cal n (episodes) | Cal SPY/NVDA/TSLA | Cal h min/med/max | Wide B0/B1 | Narrow B0/B1 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    short = {"SPYon": "S", "NVDAon": "N", "TSLAon": "T"}
    for row in support["folds"]:
        test_symbols = "+".join(short[sym] for sym in sorted(row["test_symbols"]))
        test_h = "/".join(str(v) for v in row["test_freeze_hours"])
        cal_symbols = "/".join(str(row["cal_symbols"][sym]) for sym in ("SPYon", "NVDAon", "TSLAon"))
        cal_h = f"{row['cal_freeze_hours_min']}/{row['cal_freeze_hours_median']:g}/{row['cal_freeze_hours_max']}"
        lines.append(
            f"| {row['date']} | {row['type']} | {row['test_n']} | {test_symbols} | {test_h} | "
            f"{row['cal_n']} ({row['cal_episodes']}) | {cal_symbols} | {cal_h} | "
            f"{row['wide_branch0_cal']}/{row['wide_branch1_cal']} | "
            f"{row['narrow_branch0_cal']}/{row['narrow_branch1_cal']} |"
        )
    lines.extend([
        "",
        "Counts are observed symbol-events, not an effective-sample-size estimate; events within an episode remain clustered by date. "
        "S/N/T = SPYon/NVDAon/TSLAon. B0/B1 gives calibration counts by post-open branch. A finite 90% branch radius requires at least nine events in that branch. "
        f"Wide is underpowered in {support['wide_underpowered_folds']}/{len(support['folds'])} folds "
        f"(B1 {support['wide_branch1_cal_range'][0]}–{support['wide_branch1_cal_range'][1]}); "
        f"Narrow is underpowered in {support['narrow_underpowered_folds']}/{len(support['folds'])} folds. "
        f"NVDA contributes only {support['nvda_cal_range'][0]}–{support['nvda_cal_range'][1]} calibration events per fold. "
        "The 114/138 h freeze durations are represented in every calibration fold, but duration support alone does not identify a cross-duration radius.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def decompose_rewards(e: EventData, policy) -> dict:
    """Truth-replay reward components (fee, IL, cost), normalized by V0."""
    pos = Position()
    prev_il = 0.0
    n = len(e.volume)
    fee = np.empty(n, dtype=float)
    il = np.empty(n, dtype=float)
    cost = np.empty(n, dtype=float)
    for t in range(n):
        r = int(e.policy_regime[t])
        nr = int(e.policy_regime[t + 1]) if t + 1 < n else r
        action = int(policy(t, r, nr, float(e.p[t]), pos))
        c, changed = apply_action(pos, action, float(e.p[t]))
        if changed:
            prev_il = 0.0
        in_range = bool(pos.in_pos and pos.lower <= e.p[t] <= pos.upper)
        fee[t] = e.fee_rate * float(e.volume[t]) * LIQUIDITY_SHARE * float(in_range)
        il_now = pos.il(float(e.p[t + 1])) if pos.in_pos else 0.0
        il[t] = il_now - prev_il
        prev_il = il_now
        cost[t] = c
    out = {}
    for k, m in e.masks.items():
        out[k] = {
            "fee": float(fee[m].sum() / V0),
            "il": float(il[m].sum() / V0),
            "cost": float(cost[m].sum() / V0),
        }
    return out


def make_records(events: list[EventData], params: dict,
                 draws: int) -> list[dict]:
    records = []
    for idx, e in enumerate(events):
        rec = {
            "event_id": e.event_id,
            "symbol": e.symbol,
            "type": e.type,
            "date": e.date,
            "thin": e.thin,
            "freeze_start": e.freeze_start.isoformat(),
            "freeze_end": e.freeze_end.isoformat(),
            "stale_start": e.stale_start.isoformat(),
            "stale_open": e.stale_open.isoformat(),
            "n_steps": int(len(e.volume)),
            "n_swaps_window": e.n_swaps_window,
            "truth": {},
            "models": {},
            "components": {},
        }
        for pname, policy in POLICIES.items():
            rew, actions = rollout_rewards(
                e.p, e.volume, e.policy_regime, policy, e.fee_rate
            )
            truth = sum_targets(rew, e.masks)
            # Branch at the post-open decision: does the policy rebalance at
            # any step in the 2h post-open window?  The first-OPEN action is
            # kept as a secondary diagnostic; in this sample the price jump
            # often arrives one step after the first OPEN hour, so the
            # first-OPEN label undercounts rebalances.
            post = np.flatnonzero(e.masks["post2h"])
            nz = post[np.isin(actions[post], (1, 2))]
            b = int(len(nz) > 0)
            b_action = int(actions[nz[0]]) if len(nz) else -1
            first_open = np.flatnonzero(
                (e.times[:-1] >= e.freeze_end) & (e.regime == R_OPEN)
            )
            b_first = 0
            if len(first_open):
                b_first = int(actions[int(first_open[0])] in (1, 2))
            rec["truth"][pname] = truth
            rec["components"][pname] = decompose_rewards(e, policy)
            rec.setdefault("branch", {})[pname] = b
            rec.setdefault("branch_action", {})[pname] = b_action
            rec.setdefault("branch_first_open", {})[pname] = b_first
            rec["models"][pname] = {}
            for model in ("identified", "naive"):
                rng = np.random.default_rng(
                    [idx, list(POLICIES).index(pname),
                     0 if model == "identified" else 1, 20261022]
                )
                model_vals = model_draws(
                    rng, float(e.p[0]), e.regime, e.volume,
                    e.policy_regime, policy, e.fee_rate, e.masks,
                    params[e.symbol], model, draws,
                )
                rec["models"][pname][model] = {}
                for target in TARGETS:
                    arr = model_vals[target]
                    mu = float(arr.mean())
                    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                    q05, q95 = np.quantile(arr, [0.05, 0.95]).tolist()
                    y = truth[target]
                    rec["models"][pname][model][target] = {
                        "mu": mu,
                        "sd": sd,
                        "q05": float(q05),
                        "q95": float(q95),
                        "truth": float(y),
                        "score": {
                            sc: score_one(y, mu, sd, float(q05), float(q95), sc)
                            for sc in SCORES
                        },
                    }
        records.append(rec)
    return records


def eval_fee_sensitivity(events: list[EventData], params: dict, draws: int,
                         alpha: float, fee_rates: tuple[float, ...]) -> dict:
    out = {}
    for fee in fee_rates:
        evs = [replace(e, fee_rate=fee) for e in events]
        recs = make_records(evs, params, draws)
        main = []
        for rec in recs:
            r = dict(rec)
            r["models"] = rec["models"]["Wide_Rebalance_2pct"]
            r["branch_wide"] = rec["branch"]["Wide_Rebalance_2pct"]
            r["branch_first_open_wide"] = rec["branch_first_open"][
                "Wide_Rebalance_2pct"
            ]
            main.append(r)
        out[f"{fee:.4f}"] = {
            model: {
                target: _eval_method(
                    main, model, "raw", target, lambda e: e["date"], alpha,
                    "date",
                )
                for target in ("total_main", "freeze", "post2h")
            }
            for model in ("identified", "naive")
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-events", type=Path, default=DEFAULT_CASE)
    ap.add_argument("--event-json-dir", type=Path, default=DEFAULT_EVENT_JSON)
    ap.add_argument("--swaps-dir", type=Path, default=DEFAULT_SWAPS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--table-path", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--draws", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pre-hours", type=float, default=24.0)
    ap.add_argument("--post-hours", type=float, default=24.0)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--fee-sensitivity", action="store_true")
    ap.add_argument("--sample-kind", choices=("case", "robustness"),
                    default="case")
    ap.add_argument("--window-label",
                    default="2026-07-01T00:00Z/2026-10-20T00:00Z")
    ap.add_argument("--table-heading",
                    default="Table 12 — real-path usage check in the liquid-regime case")
    args = ap.parse_args()

    case = pd.read_csv(args.case_events, parse_dates=["freeze_start"])
    # Coverage and transfer analyses use the A layer; sensitivity checks retain
    # every event for the thin/non-thin split rather than dropping thin events.
    case = case[case["g1r_usable_2h"].astype(bool)].copy()
    if args.max_events:
        case = case.head(args.max_events)
    print(f"case A-layer events: {len(case)}")

    # Load per-event JSON once.
    ev_json = {}
    for eid in case["event_id"]:
        path = args.event_json_dir / f"{eid}.json"
        if path.exists():
            ev_json[eid] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"WARN missing event JSON: {eid}")

    # Load swaps per symbol using the first available event's pool metadata.
    swaps = {}
    meta = {}
    for sym in sorted(case["symbol"].unique()):
        row = case[case["symbol"] == sym].iloc[0]
        ev = ev_json.get(str(row["event_id"]))
        if ev is None:
            continue
        quote_is0 = bool(ev["pool"]["quote_is_token0"])
        dec = int(ev["pool"]["decimals0"] if quote_is0 else ev["pool"]["decimals1"])
        price_col = ev["pool"]["price_quote_per_ondo"]
        ref_raw = ev["pool"].get("price_at_freeze_start")
        ref = float(ref_raw) if ref_raw not in (None, "") else float("nan")
        path = args.swaps_dir / f"{sym}_swaps.csv"
        if not path.exists():
            print(f"WARN missing swaps: {path}")
            continue
        swaps[sym], stats = load_swaps(path, quote_is0, dec, price_col, ref)
        meta[sym] = {
            "path": str(path),
            "quote_is_token0": quote_is0,
            "quote_decimals": dec,
            "price_column": price_col,
            **stats,
        }
        print(f"{sym}: {len(swaps[sym])} usable swaps")

    events: list[EventData] = []
    for _, row in case.iterrows():
        ev = ev_json.get(str(row["event_id"]))
        if ev is None or row["symbol"] not in swaps:
            continue
        e = build_event(row, ev, swaps[row["symbol"]], args.pre_hours,
                        args.post_hours)
        if e is None:
            print(f"WARN could not build event: {row['event_id']}")
            continue
        events.append(e)
    print(f"built events: {len(events)}")
    if not events:
        raise SystemExit("no events built")

    params = estimate_params(events, case)

    # Truth + model draws
    event_records = make_records(events, params, args.draws)

    # E1/are evaluated for the main target policy only (Wide).  E2 is run
    # for both policies because Narrow has enough post-open rebalances for a
    # finite 90% branch radius; Wide is flagged as underpowered.
    main_model_records = []
    for rec in event_records:
        r = dict(rec)
        r["models"] = rec["models"]["Wide_Rebalance_2pct"]
        r["branch_wide"] = rec["branch"]["Wide_Rebalance_2pct"]
        r["branch_first_open_wide"] = rec["branch_first_open"]["Wide_Rebalance_2pct"]
        main_model_records.append(r)

    policy_records = {}
    for pname in POLICIES:
        prs = []
        for rec in event_records:
            r = dict(rec)
            r["models"] = rec["models"][pname]
            r["branch_policy"] = rec["branch"][pname]
            r["branch_first_open_policy"] = rec["branch_first_open"][pname]
            prs.append(r)
        policy_records[pname] = prs

    e1 = {}
    for model in ("identified", "naive"):
        for score in SCORES:
            for target in ("total_main", "event", "freeze", "open",
                           "closed", "post2h", "post24h"):
                e1[f"{model}.{score}.{target}"] = _eval_method(
                    main_model_records, model, score, target,
                    lambda e: e["date"], args.alpha, "date",
                )
        for target in ("total_main", "event", "freeze", "post2h", "post24h"):
            e1[f"{model}.plugin.{target}"] = _eval_plugin(
                main_model_records, model, target
            )

    e2 = {
        pname: {
            model: eval_e2(policy_records[pname], model, args.alpha)
            for model in ("identified", "naive")
        }
        for pname in POLICIES
    }
    e3 = {
        model: {score: eval_e3(main_model_records, model, score, args.alpha)
                for score in SCORES}
        for model in ("identified", "naive")
    }
    for model in ("identified", "naive"):
        for score in SCORES:
            e3[model][score]["symbol_conditional"] = eval_symbol_conditional(
                main_model_records, model, score, args.alpha
            )

    e5 = {
        "thin_nonthin": {
            k: {
                "thin": v.get("thin_coverage"),
                "nonthin": v.get("nonthin_coverage"),
            }
            for k, v in e1.items()
            if k.endswith("total_main")
        }
    }
    if args.fee_sensitivity:
        e5["fee"] = eval_fee_sensitivity(
            events, params, args.draws, args.alpha, (0.003, 0.01)
        )

    components = {}
    for pname in POLICIES:
        components[pname] = {}
        for target in ("total_main", "event", "freeze", "post2h", "post24h"):
            components[pname][target] = {
                k: float(np.mean([
                    rec["components"][pname][target][k] for rec in event_records
                ]))
                for k in ("fee", "il", "cost")
            }

    n_events = len(event_records)
    n_dates = len({r["date"] for r in event_records})
    n_nvda = sum(1 for r in event_records if r["symbol"] == "NVDAon")
    n_nonthin = sum(1 for r in event_records if not r["thin"])

    sample = {
        "kind": args.sample_kind,
        "window": args.window_label,
        "n_events": n_events,
        "n_dates": n_dates,
        "n_weekend": len({r["date"] for r in event_records if r["type"] == "weekend"}),
        "n_holiday": len({r["date"] for r in event_records if r["type"] == "holiday"}),
        "by_symbol": {
            sym: sum(1 for r in event_records if r["symbol"] == sym)
            for sym in sorted({r["symbol"] for r in event_records})
        },
        "thin": sum(1 for r in event_records if r["thin"]),
        "nonthin": sum(1 for r in event_records if not r["thin"]),
        "pre_hours": args.pre_hours,
        "post_hours": args.post_hours,
        "draws": args.draws,
        "alpha": args.alpha,
    }

    if args.sample_kind == "case":
        scope_caveat = (
            f"Post-hoc liquid-regime window ({args.window_label}); not a "
            "pre-registered sample. Results are regime-conditional and "
            "descriptive."
        )
    else:
        scope_caveat = (
            f"Robustness check on the 9-month A layer ({args.window_label}), "
            "not the main case. The main estimate remains the 33-event "
            "liquid-regime case; this sample mixes liquid and illiquid months "
            "and is not a pre-registered sample."
        )

    caveats = [
        scope_caveat,
        "Truth is deterministic replay of the target policy on the observed "
        "pool-price path with a P-marked reward; it is not a deployed-policy "
        "counterfactual and not a causal logged OPE estimate.",
        "Identified/naive are model-based OPE pipelines; the model is fit to "
        "observable P/oracle-basis data, not to latent S.",
        "open_sd is estimated from event-level boundary basis values, not a "
        "full Chainlink round path; treat it as a first-pass estimate.",
        "CLOSED_LIVE is a separate observation branch with its own "
        "sigma_closed; its noise scale is derived from open_sd through the "
        "closed/open volatility ratio, not estimated independently.",
        f"NVDAon uses the NVDA-USD (24/5) proxy feed and contributes only "
        f"{n_nvda} of the {n_events} A-layer events.",
        f"Bootstrap CIs resample the {n_dates} calendar dates, not "
        "symbol-events; they account for cross-symbol dependence but are wide.",
        "Observation-map parameters are estimated once on all events; "
        "leave-one-episode-out folds recalibrate only the conformal radius, "
        "so this is not nested cross-validation.",
        "Duration target-shift rows calibrate and evaluate on the same event "
        "set; they are not out-of-sample transfer results.",
        f"Thin events are retained for coverage, width, and transfer analyses; "
        f"the sensitivity check splits coverage over all {n_events} events "
        f"({n_events - n_nonthin} thin/{n_nonthin} non-thin). "
        "No economic-value estimate is reported.",
    ]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "inputs": {
            "case_events": str(args.case_events),
            "event_json_dir": str(args.event_json_dir),
            "swaps_dir": str(args.swaps_dir),
            "swaps": meta,
            "data_filter": "liquidity>0 and price in [0.5,2.0]x table/event reference",
        },
        "parameters": params,
        "events": event_records,
        "calibration_support": calibration_support(event_records, sample["alpha"]),
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e5": e5,
        "components": components,
        "caveats": caveats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out}")
    if args.table:
        write_table(result, args.table_path, args.table_heading)
        print(f"wrote {args.table_path}")
        write_calibration_support_table(result, DEFAULT_SUPPORT_TABLE)
        print(f"wrote {DEFAULT_SUPPORT_TABLE}")


if __name__ == "__main__":
    main()
