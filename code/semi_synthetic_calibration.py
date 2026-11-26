#!/usr/bin/env python3
"""P1 semi-synthetic calibration table (S2, 2026-10-22).

Non-C1 task.  The script reads:

* ``data/raw/semi_synthetic/{SYMBOL}_1h.csv`` — Yahoo Finance hourly bars
  fetched by ``fetch_semi_synthetic_prices.py``;
* ``../p2-market-structure/data/oracle_stale_intervals.csv`` — P2's measured
  Chainlink stale intervals;
* ``results/observable_h_check.json`` and ``results/policy_value_check.json`` —
  P1's observation-map estimates.

It writes:

* ``results/semi_synthetic_calibration.json`` — machine-readable table;
* ``paper/tables/table11-semi-synthetic.md`` — paper table.

Definitions
-----------
* sigma_h: std of log returns between consecutive regular-session bars
  (09:30--16:00 ET, gap <= 90 min), per hour.
* kappa_h: 1 - phi from an AR(1) regression of log close on its lag, using
  consecutive regular-session bars; per hour.  This is a local mean-reversion
  estimate, not a structural OU fit.
* jump prob / jump sd: regular-session hourly returns with |r| > 4 sigma_h;
  conditional sd is reported as a multiple of sigma_h.  This is a descriptive
  tail threshold, not a jump-diffusion MLE.
* freeze duration: P2 ``oracle_stale_intervals.csv`` rows for the matching
  24/5 feed.  Weekend rows are restricted to gaps >= 40 h; for holidays we
  take the longest gap per labelled holiday, then the median/p90 across
  holidays.  Short partial gaps inherit the calendar label and would
  otherwise bias the median down.
* gap benchmarks: gap sd divided by (i) the continuous-diffusion scale
  ``sigma_h * sqrt(tau)`` and (ii) the stress-variant jump scale
  ``gamma * sigma_h * sqrt(1 - exp(-tau/40))`` at gamma=1 and gamma=4
  (the paper's default).  The observed gap also contains freeze-period latent
  movement, so comparing it with the jump-only scale is conservative.
* h: the public bars provide return benchmarks and the feed logs provide stale
  schedules; these inputs are not a paired sample for independently fitting the
  full observation map.  The table carries P1's G0 true-S diagnostic and
  observable-only estimates and marks this as a limitation.

The script does not modify the frozen P1 dataset, the C1 loop, or any
``coverage_eval.py`` results.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("P1_DATA_ROOT", ROOT / "data" / "raw")) / "semi_synthetic"
P2_ROOT = ROOT.parents[0] / "p2-market-structure"
P2_STALE = P2_ROOT / "data" / "oracle_stale_intervals.csv"
OBS_H_JSON = ROOT / "results" / "observable_h_check.json"
G0_JSON = ROOT / "results" / "policy_value_check.json"
OUT_JSON = ROOT / "results" / "semi_synthetic_calibration.json"
OUT_TABLE = ROOT / "paper" / "tables" / "table11-semi-synthetic.md"

SYMBOLS = ("SPY", "QQQ", "TSLA", "NVDA")
P2_FEED = {
    "SPY": "SPY-USD (24/5)",
    "QQQ": "QQQ-USD (24/5)",
    "TSLA": "TSLA-USD (24/5)",
    "NVDA": "NVDA-USD (24/5)",
}
TAU0 = 40.0  # stress_variant.py default


def _session(minute: np.ndarray) -> np.ndarray:
    # Bar timestamps mark the start of the interval, so the 16:00 bar is the
    # first after-hours bar, not the 15:30--16:00 close.  Excluding it keeps
    # the regular-session return set to 09:30--16:00.
    return np.where(
        minute < 570, "pre", np.where(minute < 960, "regular", "post")
    )


def load_bars(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run code/fetch_semi_synthetic_prices.py first"
        )
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["timestamp_et"] = frame["timestamp_utc"].dt.tz_convert(
        "America/New_York"
    )
    frame = (
        frame.dropna(subset=["close"])
        .drop_duplicates(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    return frame


def return_stats(frame: pd.DataFrame) -> dict:
    et = frame["timestamp_et"]
    close = frame["close"].astype(float)
    logp = np.log(close)
    ret = logp.diff()
    dt = et.diff().dt.total_seconds()
    minute = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    session = _session(minute)
    same = session == np.roll(session, 1)
    same[0] = False
    consecutive = (dt.to_numpy() > 0) & (dt.to_numpy() <= 5400) & same

    out: dict = {
        "n_bars": int(len(frame)),
        "first_bar_et": et.iloc[0].isoformat(),
        "last_bar_et": et.iloc[-1].isoformat(),
    }
    for name in ("pre", "regular", "post"):
        mask = consecutive & (session == name)
        r = ret.to_numpy()[mask]
        out[f"n_returns_{name}"] = int(len(r))
        out[f"sigma_{name}"] = float(np.std(r, ddof=1)) if len(r) > 2 else None
        out[f"mean_{name}"] = float(np.mean(r)) if len(r) else None

    # AR(1) on consecutive regular-session bars.
    mask = consecutive & (session == "regular")
    x = logp.shift(1).to_numpy()[mask]
    y = logp.to_numpy()[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    phi = float(np.sum((x - xm) * (y - ym)) / sxx)
    intercept = float(ym - phi * xm)
    resid = y - intercept - phi * x
    sigma2 = float(np.sum(resid**2) / (n - 2))
    se_phi = math.sqrt(sigma2 / sxx)
    out["kappa_ar1_regular"] = float(1.0 - phi)
    out["kappa_ar1_regular_se"] = float(se_phi)
    out["ar1_phi_regular"] = phi
    out["ar1_n"] = int(n)

    # Jump tail descriptors on regular-session returns.
    r = ret.to_numpy()[consecutive & (session == "regular")]
    sigma_h = float(np.std(r, ddof=1))
    out["jump"] = {}
    for k in (4.0, 5.0):
        tail = r[np.abs(r) > k * sigma_h]
        out["jump"][f"threshold_{int(k)}sd"] = {
            "n": int(len(tail)),
            "prob": float(len(tail) / len(r)),
            "cond_sd": float(np.std(tail, ddof=1)) if len(tail) > 1 else None,
            "cond_sd_ratio": (
                float(np.std(tail, ddof=1) / sigma_h) if len(tail) > 1 else None
            ),
        }
    out["jump"]["sigma_h"] = sigma_h
    out["jump"]["n_regular_returns"] = int(len(r))

    # Gaps between consecutive bars: overnight, weekend, holiday.
    dates = et.dt.date.to_numpy()
    change = np.where(dates[1:] != dates[:-1])[0] + 1
    gap_ret = (logp.to_numpy()[change] - logp.to_numpy()[change - 1])
    gap_h = (
        (et.iloc[change].to_numpy() - et.iloc[change - 1].to_numpy())
        .astype("timedelta64[s]")
        .astype(float)
        / 3600.0
    )
    out["gaps"] = {}
    for name, lo, hi in (
        ("overnight", 6.0, 40.0),
        ("weekend", 40.0, 70.0),
        ("holiday", 70.0, 1e9),
    ):
        mask = (gap_h >= lo) & (gap_h < hi)
        vals = gap_ret[mask]
        out["gaps"][name] = {
            "n": int(len(vals)),
            "gap_hours_median": float(np.median(gap_h[mask])) if len(vals) else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "median_abs": float(np.median(np.abs(vals))) if len(vals) else None,
            "p99_abs": float(np.quantile(np.abs(vals), 0.99)) if len(vals) else None,
            "max_abs": float(np.max(np.abs(vals))) if len(vals) else None,
        }
    out["sigma_h_regular"] = sigma_h

    # 24/7-equivalent variance rate: realised variance over all elapsed hours,
    # including the overnight/weekend/holiday gaps.  P1's latent process moves
    # every hour, so this is the comparable per-hour scale, not the
    # regular-session-only sigma.
    all_ok = (dt.to_numpy() > 0) & np.isfinite(ret.to_numpy())
    rv = float(np.sum(ret.to_numpy()[all_ok] ** 2))
    hours = float(np.sum(dt.to_numpy()[all_ok]) / 3600.0)
    out["sigma_247"] = float(math.sqrt(rv / hours)) if hours > 0 else None
    out["annualized_vol_247"] = (
        float(math.sqrt(rv / hours) * math.sqrt(24.0 * 365.0))
        if hours > 0
        else None
    )
    return out


def freeze_stats(symbol: str) -> dict:
    """Freeze durations from P2's stale intervals.

    P2's rows include short partial gaps that inherit a holiday/weekend label
    from the calendar date (e.g. a 24 h weekday outage inside a holiday week).
    For weekends we keep only gaps of at least 40 h; for holidays we take the
    longest gap per labelled holiday and then summarise across holidays.  This
    recovers the actual closure duration instead of the label contamination.
    """
    if not P2_STALE.exists():
        raise FileNotFoundError(f"{P2_STALE} missing")
    frame = pd.read_csv(P2_STALE)
    feed = P2_FEED[symbol]
    rows = frame[frame["feed"] == feed]
    out: dict = {"p2_feed": feed, "n_rows": int(len(rows))}

    weekend = rows[rows["overlaps_freeze"].astype(str) == "weekend"][
        "gap_hours"
    ].astype(float)
    weekend = weekend[weekend >= 40.0]
    out["weekend"] = {
        "n": int(len(weekend)),
        "median_h": float(weekend.median()) if len(weekend) else None,
        "p90_h": float(weekend.quantile(0.90)) if len(weekend) else None,
        "max_h": float(weekend.max()) if len(weekend) else None,
    }

    holiday_rows = rows[
        rows["overlaps_freeze"].astype(str).str.startswith("holiday")
    ]
    per_holiday = holiday_rows.groupby("overlaps_freeze")["gap_hours"].max()
    out["holiday"] = {
        "n": int(len(per_holiday)),
        "median_h": float(per_holiday.median()) if len(per_holiday) else None,
        "p90_h": float(per_holiday.quantile(0.90)) if len(per_holiday) else None,
        "max_h": float(per_holiday.max()) if len(per_holiday) else None,
    }
    return out


def jump_scale(tau_h: float, sigma_h: float) -> float:
    if tau_h is None or sigma_h is None or tau_h <= 0:
        return float("nan")
    return sigma_h * math.sqrt(1.0 - math.exp(-tau_h / TAU0))


def gap_benchmarks(stats: dict, freeze: dict) -> dict:
    """Observed gap sd against two reference scales.

    * ``gap_ratio``: gap sd / (sigma_h * sqrt(tau)), the continuous-diffusion
      benchmark.  Values below 1 mean the freeze-period move is smaller than
      diffusing at the regular-session rate would imply.
    * ``gap_over_g1`` / ``gap_over_g4``: gap sd / (gamma * g_scale) for the
      stress variant's gamma=1 and gamma=4 (the paper's default) scales.
      Values below 1 mean the observed total gap is smaller than the jump
      component alone at that gamma.  The observed gap also contains any
      freeze-period latent movement, so this is a conservative comparison.
    """
    out: dict = {}
    sigma = stats["sigma_h_regular"]
    for name in ("weekend", "holiday"):
        gap_sd = stats["gaps"][name]["sd"]
        tau = freeze[name]["median_h"]
        unit = jump_scale(tau, sigma) if tau else float("nan")
        out[f"tau_{name}"] = tau
        out[f"gap_ratio_{name}"] = (
            float(gap_sd / (sigma * math.sqrt(tau)))
            if gap_sd is not None and tau and sigma > 0
            else None
        )
        for gamma in (1.0, 4.0):
            scale = gamma * unit
            out[f"gap_over_g{int(gamma)}_{name}"] = (
                float(gap_sd / scale)
                if gap_sd is not None and math.isfinite(scale) and scale > 0
                else None
            )
    return out


def load_h() -> dict:
    out: dict = {}
    if G0_JSON.exists():
        raw = json.loads(G0_JSON.read_text())["observation_mapping_estimate"]
        out["g0_true_s_diagnostic"] = {
            k: float(raw[k])
            for k in ("open_sd", "ext_phi", "ext_sd", "stale_phi", "stale_sd")
        }
    if OBS_H_JSON.exists():
        raw = json.loads(OBS_H_JSON.read_text())["h_estimate"]
        out["observable_only"] = {
            k: float(raw[k])
            for k in ("open_sd", "ext_phi", "ext_sd", "stale_phi", "stale_sd")
        }
    return out


def build_table(result: dict) -> str:
    lines = [
        "# Table 11 — Semi-synthetic calibration",
        "",
        "Real hourly bars (Yahoo Finance, 2025-10-22 to 2026-10-21, pre/post "
        "included) calibrate the return dynamics; freeze durations are measured "
        "Chainlink stale intervals. The full observation map is not independently "
        "calibrated by these inputs.",
        "",
        "## Panel A. Return dynamics (regular session)",
        "",
        "| Symbol | $n$ bars | $\\sigma_h$ ($10^{-3}$) | $\\sigma_h^{24/7}$ ($10^{-3}$) | $\\hat\\kappa_h$ ($10^{-3}$) | jump prob. ($\\lvert r\\rvert>4\\sigma_h$) | jump sd / $\\sigma_h$ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for symbol in SYMBOLS:
        s = result["symbols"][symbol]
        j = s["jump"]["threshold_4sd"]
        lines.append(
            "| {sym} | {n} | {sig:.2f} | {sig247:.2f} | {kap:.2f} | {jp:.4f} | {js:.2f} |".format(
                sym=symbol,
                n=s["n_bars"],
                sig=1000 * s["sigma_h_regular"],
                sig247=1000 * s["sigma_247"],
                kap=1000 * s["kappa_ar1_regular"],
                jp=j["prob"],
                js=j["cond_sd_ratio"] if j["cond_sd_ratio"] is not None else float("nan"),
            )
        )
    lines += [
        "",
        "## Panel B. Freeze duration and gap benchmarks",
        "",
        "| Symbol | weekend $\\tau$ median (h) | weekend $\\tau$ p90 (h) | holiday $\\tau$ median (h) | holiday $\\tau$ p90 (h) | weekend gap / $\\sigma_h\\sqrt{\\tau}$ | holiday gap / $\\sigma_h\\sqrt{\\tau}$ | weekend gap / $g(\\gamma=4)$ | holiday gap / $g(\\gamma=4)$ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for symbol in SYMBOLS:
        s = result["symbols"][symbol]
        f = s["freeze"]
        b = s["benchmarks"]
        def fmt(x):
            return f"{x:.2f}" if x is not None else "--"
        lines.append(
            "| {sym} | {wm:.1f} | {wp:.1f} | {hm:.1f} | {hp:.1f} | {wr} | {hr} | {wg} | {hg} |".format(
                sym=symbol,
                wm=f["weekend"]["median_h"] if f["weekend"]["median_h"] else float("nan"),
                wp=f["weekend"]["p90_h"] if f["weekend"]["p90_h"] else float("nan"),
                hm=f["holiday"]["median_h"] if f["holiday"]["median_h"] else float("nan"),
                hp=f["holiday"]["p90_h"] if f["holiday"]["p90_h"] else float("nan"),
                wr=fmt(b["gap_ratio_weekend"]),
                hr=fmt(b["gap_ratio_holiday"]),
                wg=fmt(b["gap_over_g4_weekend"]),
                hg=fmt(b["gap_over_g4_holiday"]),
            )
        )
    lines += [
        "",
        "## Panel C. Observation-map parameters (not independently calibrated here)",
        "",
        "| Source | $\\sigma_o$ | $\\phi_{ext}$ | $\\sigma_{ext}$ | $\\phi_{stale}$ | $\\sigma_{stale}$ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in (
        ("G0 true-$S$ diagnostic", "g0_true_s_diagnostic"),
        ("observable-only estimate", "observable_only"),
    ):
        h = result["h"].get(key)
        if not h:
            continue
        lines.append(
            "| {label} | {o:.5f} | {ep:.3f} | {es:.5f} | {sp:.4f} | {ss:.5f} |".format(
                label=label,
                o=h["open_sd"],
                ep=h["ext_phi"],
                es=h["ext_sd"],
                sp=h["stale_phi"],
                ss=h["stale_sd"],
            )
        )
    sig = [1000 * result["symbols"][s]["sigma_h_regular"] for s in SYMBOLS]
    sig247 = [1000 * result["symbols"][s]["sigma_247"] for s in SYMBOLS]
    ann = [
        100 * result["symbols"][s]["annualized_vol_247"] for s in SYMBOLS
    ]
    kap = [1000 * result["symbols"][s]["kappa_ar1_regular"] for s in SYMBOLS]
    jp = [
        result["symbols"][s]["jump"]["threshold_4sd"]["prob"] for s in SYMBOLS
    ]
    js = [
        result["symbols"][s]["jump"]["threshold_4sd"]["cond_sd_ratio"]
        for s in SYMBOLS
    ]
    wt = [result["symbols"][s]["freeze"]["weekend"]["median_h"] for s in SYMBOLS]
    ht = [result["symbols"][s]["freeze"]["holiday"]["median_h"] for s in SYMBOLS]

    lines += [
        "",
        "Notes. $\\sigma_h$ is the standard deviation of hourly log returns "
        "within the regular session (09:30--16:00 ET, gaps $\\le 90$ min); "
        "bars are stamped at the start of their interval, so the 16:00 bar is "
        "the first after-hours bar and is excluded. "
        "$\\sigma_h^{24/7}=\\sqrt{\\sum r^2/\\sum\\Delta t}$ over all consecutive "
        "bars including overnight/weekend/holiday gaps, i.e. a per-elapsed-hour "
        "realised variance rate; "
        "$\\hat\\kappa_h=1-\\hat\\phi$ from an AR(1) on consecutive regular-session "
        "bars. Jump probability and conditional sd use the $4\\sigma_h$ tail "
        "(5--9 events per symbol, so the conditional sd is noisy); "
        "the conditional sd is descriptive and not a jump-diffusion MLE. "
        "Overnight/weekend/holiday sd are log-close changes across duration "
        "buckets of 6--40 h, 40--70 h, and $>70$ h; the 40--70 h bucket can "
        "include midweek holiday closures, since it is not a calendar label. "
        "Freeze durations are measured Chainlink stale intervals for the matching "
        "24/5 feed: weekend rows with gap $\\ge 40$ h (28 gaps per feed), and "
        "the longest gap per labelled holiday (5 holidays per feed) before "
        "taking the median/p90 across holidays (short partial gaps inherit the "
        "calendar label and would otherwise bias the median down). The "
        "measured weekend stale interval (52--56 h) is shorter than the "
        "Friday-close-to-Monday-open market gap (about 65 h) because the 24/5 "
        "feed resumes on Sunday evening. "
        "In Panel B, gap/$\\sigma_h\\sqrt{\\tau}$ is the continuous-diffusion "
        "benchmark, and gap/$g(\\gamma=4)$ compares the observed total gap "
        "sd with the jump-only scale of the paper's stress variant "
        "($g=\\gamma\\sigma_h\\sqrt{1-e^{-\\tau/40}}$, $\\gamma=4$, "
        "$\\tau_0=40$). The gap sd is measured across the market closure "
        "(about 65 h on weekends), while $\\tau$ is the oracle stale interval "
        "(52--56 h); using the longer market duration in the denominator would "
        "lower the ratios, so the reported values are conservative. Values "
        "below 1 mean the observed gap is smaller than the benchmark. Because "
        "the observed gap also contains any freeze-period latent movement, the "
        "jump component itself is even smaller; the stress design is therefore "
        "conservative relative to these feeds. The observation map $h$ is not "
        "independently estimated by these calibration inputs: public bars provide "
        "return benchmarks and feed logs provide stale schedules, rather than a "
        "paired sample for fitting the full map; both synthetic observation-map "
        "estimates are reported, and the paper's coverage experiments use the "
        "observable-only one as their main specification.",
        "",
        "Calibration summary. The simulator's DGP uses $\\sigma_h=8.0\\times10^{-3}$, "
        "$\\kappa_h=3.0\\times10^{-3}$, jump probability 0.005 with jump sd "
        "$3\\sigma_h$, weekend freeze 65 h, and holiday freeze 81 h. The four "
        f"feeds span $\\sigma_h={min(sig):.2f}$--${max(sig):.2f}\\times10^{{-3}}$, "
        f"$\\sigma_h^{{24/7}}={min(sig247):.2f}$--${max(sig247):.2f}\\times10^{{-3}}$ "
        f"(annualised {min(ann):.0f}--{max(ann):.0f}%), "
        f"$\\kappa_h={min(kap):.2f}$--${max(kap):.2f}\\times10^{{-3}}$, "
        f"jump probability {min(jp):.4f}--{max(jp):.4f} (bracketing the simulator's 0.005) "
        "with conditional sd "
        f"{min(js):.1f}$\\sigma_h$--{max(js):.1f}$\\sigma_h$, weekend freeze "
        f"{min(wt):.1f}--{max(wt):.1f} h, and holiday freeze "
        f"{min(ht):.1f}--{max(ht):.1f} h. The simulator's $\\sigma_h=8.0\\times10^{{-3}}$ "
        "is above every session estimate and 1.6--5.4$\\times$ the 24/7-equivalent "
        "rate (its annualised latent vol is about 75%); the environment is "
        "therefore a high-volatility stress setting, not a broad-market "
        "calibration. Its mean reversion sits inside the measured range "
        f"(${min(kap):.2f}$--${max(kap):.2f}\\times10^{{-3}}$), close to "
        "NVDA's upper end; its weekend freeze is longer than the measured "
        "52--56 h; its 81 h holiday freeze is close to the measured 76--79 h; "
        "and its 3$\\sigma_h$ jump sd is thinner-tailed than the measured "
        f"{min(js):.1f}$\\sigma_h$--{max(js):.1f}$\\sigma_h$ conditional sd.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not P2_STALE.exists():
        raise FileNotFoundError(P2_STALE)
    result: dict = {
        "config": {
            "symbols": list(SYMBOLS),
            "price_source": "Yahoo Finance chart API, interval=1h, range=1y, "
            "includePrePost=true",
            "price_window": "2025-10-22 to 2026-10-21 (per fetched cache)",
            "freeze_source": str(P2_STALE.relative_to(ROOT.parents[0])),
            "freeze_filter": "weekend gaps >=40 h; holiday = longest gap per "
            "labelled holiday",
            "sigma_session": "regular 09:30--16:00 ET, consecutive bars <=90 min; "
            "bar timestamps are interval starts, so the 16:00 bar is excluded",
            "sigma_247": "sqrt(sum r^2 / sum dt) over all consecutive bars, "
            "per elapsed hour",
            "kappa": "1 - phi from AR(1) on consecutive regular-session log closes",
            "jump": "|r| > 4 sigma_h on regular-session hourly returns",
            "gap_benchmarks": "gap sd / (sigma_h*sqrt(tau)) and gap sd / "
            "(gamma*sigma_h*sqrt(1-exp(-tau/40))) for gamma=1,4",
            "tau0_hours": TAU0,
            "h_source": "P1 G0 diagnostic and observable-only estimates; "
            "not independently calibrated by the public-bar/feed-schedule inputs",
        },
        "symbols": {},
        "h": load_h(),
        "limits": [
            "Yahoo hourly closes are consolidated market data, not the latent "
            "reference price of the AMM; they proxy the 24/5 equity reference.",
            "The observation map h is not independently calibrated to real pool "
            "prices in this semi-synthetic check; the public bars and feed "
            "schedules used here calibrate return/freeze inputs, not a paired "
            "full-h estimation sample. The run therefore tests return/freeze "
            "calibration, not h external validity.",
            "kappa is a local AR(1) estimate on a nonstationary price level; "
            "it is not a structural mean-reversion parameter.",
            "Jump statistics use a fixed 4-sigma threshold and are descriptive; "
            "they do not identify a jump-diffusion mixture.",
            "Weekend/holiday gap sd mixes latent price movement with market "
            "microstructure at the reopen; it is not a pure de-freeze jump.",
            "The gap sd is measured across the market closure (~65 h on "
            "weekends), not the 52--56 h oracle stale interval used as tau; "
            "using the longer market duration in the denominator would lower "
            "the reported gap ratios, so they are conservative.",
            "Sigma and the AR(1) use bar timestamps as interval starts; the "
            "16:00 bar is after-hours and is excluded from the regular "
            "session. The 40--70 h gap bucket is duration-based and can "
            "contain midweek holiday closures.",
            "P2's holiday labels include short partial gaps (e.g. 24 h "
            "weekday outages inside a holiday week); the table takes the "
            "longest gap per labelled holiday to recover the closure.",
        ],
    }
    for symbol in SYMBOLS:
        stats = return_stats(load_bars(symbol))
        freeze = freeze_stats(symbol)
        stats["freeze"] = freeze
        stats["benchmarks"] = gap_benchmarks(stats, freeze)
        result["symbols"][symbol] = stats
        print(
            f"{symbol}: sigma_h={stats['sigma_h_regular']:.5f} "
            f"kappa_h={stats['kappa_ar1_regular']:.5f} "
            f"weekend_tau={freeze['weekend']['median_h']:.1f}h "
            f"holiday_tau={freeze['holiday']['median_h']:.1f}h"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1) + "\n")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(build_table(result))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
