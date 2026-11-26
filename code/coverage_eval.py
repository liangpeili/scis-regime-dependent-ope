"""C1 OPE: split-conformal coverage of per-step rewards under regime shift.

H3 negative-result machine: standard split conformal prediction on per-step
reward_env should hold ~90% coverage under an iid random split (positive
control) but break (under-cover) on a held-out calendar regime (distribution
shift).

No-lookahead guarantee: every feature for predicting y_t = reward_env at step t
is either the action taken at t (action_id_t, chosen before the step's outcome
is observed) or a value recorded at step t-1 (lagged within its own
strategy/window/seed trajectory). The regime label of step t is used ONLY as
the holdout split key and (in the Mondrian variant) as the calibration group
key, never as a feature. Regimes are calendar-derived and known at time t
(no HMM smoothing, no lookahead).

METHOD (kept): Mondrian / regime-conditional conformal. The calibration
residuals are grouped by regime and a separate radius is computed per group.
For a test row whose regime was observed during calibration we use that
regime's radius (the random-split control). For a completely held-out regime
no calibration residual of that regime exists, so we fall back to the largest
per-regime radius observed in calibration, restricted to the test regime's
observation-mapping class ({OPEN, EXTENDED} = informative;
{WEEKEND, HOLIDAY, HALT} = stale/frozen). The unrestricted global-maximum
variant is kept as a baseline via `--fallback max`; the original global
split-conformal baseline is commit c610494.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

C1_DIR = Path(__file__).resolve().parent
OUT_DIR = C1_DIR.parent / "results"
OUT_JSON = OUT_DIR / "coverage_gap.json"

ALPHA = 0.1  # nominal coverage 1 - ALPHA = 0.9
NOMINAL = 1.0 - ALPHA
SPARSE_THRESHOLD = 50
GROUP_COLS = ["strategy", "window", "seed"]

# Observation-mapping classes, used by the "similarity" fallback:
#   {OPEN, EXTENDED}          informative (anchored or lagged stream)
#   {WEEKEND, HOLIDAY, HALT}  stale anchor / frozen pool
# For a held-out regime we take the max calibration radius among the other
# regimes in its class. The classes follow from the regime definitions, not
# from the test residuals. The alternative partitions are sensitivity checks:
#   three_class             HALT split off from the stale class
#   singletons              no class knowledge (every held-out class is empty
#                           -> falls back to the global max, i.e. experiment 1)
#   weekend_informative     WEEKEND wrongly treated as informative (stress)
#   holiday_informative     HOLIDAY wrongly treated as informative (stress)
#   halt_informative        HALT wrongly treated as informative (stress)
CLASS_PARTITIONS = {
    "two_class": {
        0: (0, 1),
        1: (0, 1),
        2: (2, 3, 4),
        3: (2, 3, 4),
        4: (2, 3, 4),
    },
    "three_class": {
        0: (0, 1),
        1: (0, 1),
        2: (2, 3),
        3: (2, 3),
        4: (4,),
    },
    "singletons": {0: (0,), 1: (1,), 2: (2,), 3: (3,), 4: (4,)},
    "weekend_informative": {
        0: (0, 1, 2),
        1: (0, 1, 2),
        2: (0, 1, 2),
        3: (3, 4),
        4: (3, 4),
    },
    "holiday_informative": {
        0: (0, 1, 3),
        1: (0, 1, 3),
        2: (2, 4),
        3: (0, 1, 3),
        4: (2, 4),
    },
    "halt_informative": {
        0: (0, 1, 4),
        1: (0, 1, 4),
        2: (2, 3),
        3: (2, 3),
        4: (0, 1, 4),
    },
}
REGIME_CLASSES = CLASS_PARTITIONS["two_class"]

# action_id_t is the t-step action; everything else is shifted by one step
# inside its own trajectory, so the feature row for y_t contains only
# information available before the t-step outcome is revealed.
FEATURE_COLS = [
    "action_id",
    "in_position_lag1",
    "portfolio_value_lag1",
    "fee_income_step_usd_lag1",
    "impermanent_loss_usd_lag1",
    "operation_cost_usd_lag1",
    "eth_ret_lag1",
    "usdc_ret_lag1",
]

LAG_SOURCE_COLS = [
    "in_position",
    "portfolio_value",
    "fee_income_step_usd",
    "impermanent_loss_usd",
    "operation_cost_usd",
]


def load_dataset() -> pd.DataFrame:
    parquet = OUT_DIR / "dataset.parquet"
    csvgz = OUT_DIR / "dataset.csv.gz"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csvgz.exists():
        return pd.read_csv(csvgz)
    raise FileNotFoundError("run generate_dataset.py first (no dataset.parquet/csv.gz)")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["episode_step"]).copy()
    g = df.groupby(GROUP_COLS, sort=False)

    for col in LAG_SOURCE_COLS:
        df[f"{col}_lag1"] = g[col].shift(1)
    # per-step simple return of each leg at step t-1, computed from that
    # step's own prices and then lagged: ret_{t-1} = price_to / price_from - 1
    df["eth_ret"] = df["eth_price_to"] / df["eth_price_from"] - 1.0
    df["usdc_ret"] = df["usdc_price_to"] / df["usdc_price_from"] - 1.0
    df["eth_ret_lag1"] = g["eth_ret"].shift(1)
    df["usdc_ret_lag1"] = g["usdc_ret"].shift(1)

    df["in_position_lag1"] = df["in_position_lag1"].astype("float")

    # First row of each trajectory has no t-1 -> NaN lags -> dropped.
    # `regime` is the calendar regime at step t (generated by synthetic_env.py)
    # and is the holdout split key only; it never enters FEATURE_COLS.
    out = df.dropna(subset=FEATURE_COLS + ["reward_env", "regime"]).copy()
    out = out[out["regime"] >= 0].copy()
    out["regime"] = out["regime"].astype(int)
    return out


def conformal_radius(abs_resid: np.ndarray, alpha: float) -> float:
    # Split-conformal radius: the k-th order statistic of the calibration
    # absolute residuals, k = ceil((n+1)(1-alpha)) (no division by n; this is
    # exactly the k-th smallest residual). When k > n the finite calibration
    # set cannot certify the requested level, and the conformally valid radius
    # is +inf (the interval covers all of R) rather than the largest observed
    # residual — truncating k to n would silently tighten the interval below
    # its guaranteed level.
    n = len(abs_resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(abs_resid)[k - 1])


def mondrian_radii(abs_resid: np.ndarray, regimes: np.ndarray, alpha: float) -> dict:
    """Per-regime split-conformal radii (Mondrian calibration)."""
    return {
        int(r): conformal_radius(abs_resid[regimes == r], alpha)
        for r in np.unique(regimes)
    }


def fit_intervals(X_train, y_train, X_cal, y_cal, X_test,
                  cal_regimes=None, test_regimes=None, exclude_regimes=(),
                  fallback_candidates=None):
    """Fit the model and return (y_hat, radius, radii_by_regime).

    With `cal_regimes`, calibration is Mondrian: separate radii per regime.
    If `test_regimes` is given, each test row gets its own regime's radius
    (unseen regime -> fallback radius); otherwise a single scalar radius is
    returned. `fallback_candidates` restricts the fallback to a set of
    regimes (similarity classes); `exclude_regimes` removes regimes from the
    fallback candidates only (ablation). If a restriction empties the
    candidate set, all calibration regimes are used.
    """
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    cal_resid = np.abs(y_cal - model.predict(X_cal))
    y_hat = model.predict(X_test)

    if cal_regimes is None:
        radius = conformal_radius(cal_resid, ALPHA)
        return y_hat, radius, {}

    radii = mondrian_radii(cal_resid, cal_regimes, ALPHA)
    eligible = dict(radii)
    if fallback_candidates is not None:
        eligible = {
            k: v for k, v in eligible.items() if k in set(fallback_candidates)
        }
    eligible = {
        k: v for k, v in eligible.items() if k not in set(exclude_regimes)
    }
    max_radius = max((eligible or radii).values()) if radii else float("inf")
    if test_regimes is None:
        radius = max_radius
    else:
        radius = np.array(
            [radii.get(int(r), max_radius) for r in test_regimes], dtype=float
        )
    return y_hat, radius, radii


def strategy_mondrian_radii(abs_resid: np.ndarray, regimes: np.ndarray,
                            strategies: np.ndarray, alpha: float) -> dict:
    """Per-(regime, strategy) split-conformal radii."""
    return {
        (int(r), str(s)): conformal_radius(
            abs_resid[(regimes == r) & (strategies == s)], alpha
        )
        for r, s in set(zip(regimes.tolist(), strategies.tolist()))
    }


def fit_intervals_strategy(X_train, y_train, X_cal, y_cal, X_test,
                           cal_regimes, cal_strategies,
                           test_regimes, test_strategies,
                           unseen_regime, exclude_regimes=(),
                           fallback_candidates=None):
    """Fit with (regime, strategy)-conditional Mondrian radii.

    For a seen test regime (random-split control), each test row uses its own
    (regime, strategy) cell. For an unseen test regime, the fallback for a
    test row (r, s) is the max radius over calibration regimes in the same
    observation-mapping class with the same strategy s; if that is empty,
    widen to all calibration regimes for s, then to the global max.
    """
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    cal_resid = np.abs(y_cal - model.predict(X_cal))
    y_hat = model.predict(X_test)
    radii = strategy_mondrian_radii(
        cal_resid, cal_regimes, cal_strategies, ALPHA
    )
    max_radius = max(radii.values()) if radii else float("inf")
    strategy_max = {}
    for (_, s), v in radii.items():
        strategy_max[s] = max(strategy_max.get(s, -np.inf), v)
    exclude = set(exclude_regimes)
    class_set = None if fallback_candidates is None else set(fallback_candidates)

    radius = np.empty(len(test_regimes), dtype=float)
    for i, (r, s) in enumerate(zip(test_regimes, test_strategies)):
        r, s = int(r), str(s)
        if not unseen_regime:
            radius[i] = radii.get((r, s), strategy_max.get(s, max_radius))
            continue
        candidates = [
            v for (rr, ss), v in radii.items()
            if ss == s and rr != r and rr not in exclude
            and (class_set is None or rr in class_set)
        ]
        if not candidates:
            candidates = [
                v for (rr, ss), v in radii.items()
                if ss == s and rr != r and rr not in exclude
            ]
        if not candidates:
            candidates = [v for (_, ss), v in radii.items() if ss == s]
        radius[i] = max(candidates) if candidates else max_radius
    return y_hat, radius, radii


def coverage_and_width(y_true, y_hat, radius):
    """Coverage and mean interval width; radius may be scalar or per-row."""
    covered = np.abs(y_true - y_hat) <= radius
    return float(np.mean(covered)), float(2.0 * np.mean(radius))


def per_strategy_stats(y_true, y_hat, radius, strategies):
    """Split coverage/width by logging policy (strategy-conditional check).

    Returns (coverage, width, n_test) dicts keyed by strategy name. The same
    radius object used for the pooled metric is reused here, so the per-cell
    numbers are not a second, independently tuned method.
    """
    cov, width, n_test = {}, {}, {}
    for s in sorted(set(strategies.tolist())):
        mask = strategies == s
        r_s = radius[mask] if np.ndim(radius) else radius
        c_s, w_s = coverage_and_width(y_true[mask], y_hat[mask], r_s)
        cov[str(s)] = c_s
        width[str(s)] = w_s
        n_test[str(s)] = int(mask.sum())
    return cov, width, n_test


def chrono_train_mask(pool: pd.DataFrame, train_frac: float = 0.7) -> np.ndarray:
    """Boolean train mask for a chronological train/calibration split.

    The four test windows are calendar-contiguous market intervals, so window
    order IS calendar order, and episode_step is intra-window time: the true
    past -> future key is (window, episode_step). The candidate cut is taken
    at the train_frac quantile position of the sorted UNIQUE keys and then
    aligned to a key boundary: every row whose key equals the cut key falls on
    the calibration side. Ties therefore never straddle the split, making the
    result reproducible regardless of the input row order (many rows share a
    key — one per strategy).
    """
    keys = sorted(set(zip(pool["window"], pool["episode_step"])))
    cut_win, cut_step = keys[min(int(len(keys) * train_frac), len(keys) - 1)]
    is_train = (pool["window"] < cut_win) | (
        (pool["window"] == cut_win) & (pool["episode_step"] < cut_step)
    )
    return is_train.to_numpy()


def heldout_regime_eval(df: pd.DataFrame, exclude_regimes=(), fallback_map=None,
                        conditional=False):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["reward_env"].to_numpy(dtype=float)
    regimes = df["regime"].to_numpy()
    strategies = df["strategy"].to_numpy()

    results = {}
    for r in sorted(df["regime"].unique()):
        pool_mask = regimes != r
        pool_idx = np.flatnonzero(pool_mask)
        # chronological 70% train / 30% calibration on (window, episode_step),
        # no shuffling, split aligned to a key boundary (see chrono_train_mask)
        is_train = chrono_train_mask(df.iloc[pool_idx])
        train_idx, cal_idx = pool_idx[is_train], pool_idx[~is_train]
        test_sel = ~pool_mask
        if conditional:
            y_hat, radius, cal_radii = fit_intervals_strategy(
                X[train_idx], y[train_idx], X[cal_idx], y[cal_idx], X[test_sel],
                cal_regimes=regimes[cal_idx],
                cal_strategies=strategies[cal_idx],
                test_regimes=regimes[test_sel],
                test_strategies=strategies[test_sel],
                # every test row is regime r, which has NO calibration
                # residuals: fall back within the similarity class, per
                # strategy, to calibration regimes seen for that strategy
                unseen_regime=True,
                exclude_regimes=exclude_regimes,
                fallback_candidates=(
                    fallback_map.get(int(r)) if fallback_map is not None else None
                ),
            )
        else:
            y_hat, radius, cal_radii = fit_intervals(
                X[train_idx], y[train_idx], X[cal_idx], y[cal_idx], X[test_sel],
                cal_regimes=regimes[cal_idx],
                # all test rows are regime r, which has NO calibration residuals:
                # fall back to the max per-regime calibration radius
                test_regimes=None,
                exclude_regimes=exclude_regimes,
                fallback_candidates=(
                    fallback_map.get(int(r)) if fallback_map is not None else None
                ),
            )
        cov, width = coverage_and_width(y[test_sel], y_hat, radius)
        s_cov, s_width, s_n = per_strategy_stats(
            y[test_sel], y_hat, radius, strategies[test_sel]
        )
        results[int(r)] = {
            "coverage": cov,
            "width": width,
            # scalar radius for the max-fallback method; mean per-row radius
            # when the strategy-conditional method gives row-specific radii
            "radius": float(np.mean(radius)),
            "cal_radii": cal_radii,
            "strategy_coverage": s_cov,
            "strategy_width": s_width,
            "strategy_n_test": s_n,
            "n_test": int(test_sel.sum()),
            "n_train": int(len(train_idx)),
            "n_cal": int(len(cal_idx)),
        }
    return results


def random_split_eval(df: pd.DataFrame, conditional=False):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["reward_env"].to_numpy(dtype=float)
    regimes = df["regime"].to_numpy()
    strategies = df["strategy"].to_numpy()
    idx = np.arange(len(y))
    # 70 / 15 / 15, ignoring regime.
    # NOTE: per-row random splitting measures "intra-trajectory interpolation"
    # (train/cal/test rows come from the same trajectories), so the coverage
    # guarantee is valid but optimistic relative to deployment on genuinely
    # new trajectories. See blocked_eval for the trajectory- and path-blocked
    # controls that quantify this optimism.
    train_idx, rest_idx, _, _ = train_test_split(
        idx, y, test_size=0.30, random_state=0
    )
    cal_idx, test_idx, _, _ = train_test_split(
        rest_idx, y[rest_idx], test_size=0.50, random_state=0
    )
    if conditional:
        y_hat, radius, cal_radii = fit_intervals_strategy(
            X[train_idx], y[train_idx], X[cal_idx], y[cal_idx], X[test_idx],
            cal_regimes=regimes[cal_idx],
            cal_strategies=strategies[cal_idx],
            test_regimes=regimes[test_idx],
            test_strategies=strategies[test_idx],
            unseen_regime=False,
        )
    else:
        y_hat, radius, cal_radii = fit_intervals(
            X[train_idx], y[train_idx], X[cal_idx], y[cal_idx], X[test_idx],
            cal_regimes=regimes[cal_idx],
            test_regimes=regimes[test_idx],
        )
    cov, width = coverage_and_width(y[test_idx], y_hat, radius)
    s_cov, s_width, s_n = per_strategy_stats(
        y[test_idx], y_hat, radius, strategies[test_idx]
    )
    return {
        "coverage": cov,
        "width": width,
        "cal_radii": cal_radii,
        "strategy_coverage": s_cov,
        "strategy_width": s_width,
        "strategy_n_test": s_n,
        "n_test": int(len(test_idx)),
    }


def blocked_eval(df: pd.DataFrame, block_cols):
    """Trajectory-blocked positive control.

    Splits whole groups (no row from a group appears in two folds) instead of
    splitting rows at random. Two block levels are used:
      * block_cols = GROUP_COLS (strategy, window, seed): unseen trajectories,
        but the same market path can still appear in train via another policy;
      * block_cols = (window, seed): unseen market paths entirely.
    The per-regime Mondrian radii are calibrated on the calibration groups and
    applied per test row by its regime. Exchangeability holds at the group
    level, not the row level, so this is a diagnostic of how optimistic the
    row-random control is — not a replacement for it.
    """
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["reward_env"].to_numpy(dtype=float)
    regimes = df["regime"].to_numpy()
    group_ids = df.groupby(block_cols, sort=False).ngroup().to_numpy()
    uniq = np.unique(group_ids)
    g_train, g_rest = train_test_split(uniq, test_size=0.30, random_state=0)
    g_cal, g_test = train_test_split(g_rest, test_size=0.50, random_state=0)
    train_mask = np.isin(group_ids, g_train)
    cal_mask = np.isin(group_ids, g_cal)
    test_mask = np.isin(group_ids, g_test)
    y_hat, radius, cal_radii = fit_intervals(
        X[train_mask], y[train_mask], X[cal_mask], y[cal_mask], X[test_mask],
        cal_regimes=regimes[cal_mask],
        test_regimes=regimes[test_mask],
    )
    cov, width = coverage_and_width(y[test_mask], y_hat, radius)
    covered = (np.abs(y[test_mask] - y_hat) <= radius).astype(float)
    group_cov = pd.Series(covered).groupby(group_ids[test_mask]).mean()
    return {
        "coverage": cov,
        "width": width,
        "n_test": int(test_mask.sum()),
        "n_train_groups": int(len(g_train)),
        "n_cal_groups": int(len(g_cal)),
        "n_test_groups": int(len(g_test)),
        "group_coverage_mean": float(group_cov.mean()),
        "group_coverage_std": float(group_cov.std(ddof=1)),
        "group_coverage_se": float(
            group_cov.std(ddof=1) / np.sqrt(len(group_cov))
        ),
        "group_coverage_min": float(group_cov.min()),
        "group_coverage_max": float(group_cov.max()),
        "cal_radii": {int(k): float(v) for k, v in cal_radii.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-halt-fallback",
        action="store_true",
        help="ablation: exclude regime 4 (HALT) from the unseen-regime radius fallback",
    )
    parser.add_argument(
        "--fallback",
        choices=["max", "similarity"],
        default="similarity",
        help="unseen-regime radius: max over all calibration regimes, or max "
        "within the test regime's observation-mapping class (default: kept method)",
    )
    parser.add_argument(
        "--classes",
        choices=sorted(CLASS_PARTITIONS),
        default="two_class",
        help="observation-mapping partition for --fallback similarity; "
        "non-default partitions are sensitivity checks",
    )
    parser.add_argument(
        "--conditional",
        choices=["none", "strategy"],
        default="none",
        help="calibration granularity: regime-only (none) or per "
        "(regime, strategy) cell (strategy); the latter is a conditional-"
        "coverage check, not the kept default unless it is kept explicitly",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default: results/coverage_gap.json for the kept "
        "method; required explicitly for baselines/ablations so they cannot "
        "overwrite the canonical result)",
    )
    args = parser.parse_args()
    exclude_regimes = (4,) if args.exclude_halt_fallback else ()
    fallback_map = (
        CLASS_PARTITIONS[args.classes] if args.fallback == "similarity" else None
    )
    conditional = args.conditional == "strategy"
    if args.out is None:
        if (
            args.fallback != "similarity"
            or exclude_regimes
            or args.classes != "two_class"
            or conditional
        ):
            parser.error(
                "--out is required when running a baseline/sensitivity check; "
                "refusing to overwrite results/coverage_gap.json with a "
                "non-default run"
            )
        args.out = OUT_JSON

    df = build_features(load_dataset())
    print(f"feature rows after lag drop: {len(df)}")
    print(f"fallback exclude regimes: {sorted(exclude_regimes) or 'none'}")
    print(f"fallback mode: {args.fallback}")
    print(f"conditional calibration: {args.conditional}")

    heldout = heldout_regime_eval(
        df, exclude_regimes=exclude_regimes, fallback_map=fallback_map,
        conditional=conditional,
    )
    random_split = random_split_eval(df, conditional=conditional)
    traj_blocked = blocked_eval(df, GROUP_COLS)
    path_blocked = blocked_eval(df, ["window", "seed"])

    sparse_regimes = [
        r for r in heldout if heldout[r]["n_test"] < SPARSE_THRESHOLD
    ]
    dense_regimes = [r for r in heldout if r not in sparse_regimes]
    # Aggregate gap: n_test-weighted mean of max(0, nominal - coverage) over
    # NON-SPARSE held-out regimes only. Sparse regimes (n_test < 50) are
    # excluded from the aggregate — their coverage estimates have huge
    # binomial variance — and reported separately via sparse_regimes and the
    # per-regime fields.
    if dense_regimes:
        gap = float(
            sum(
                heldout[r]["n_test"] * max(0.0, NOMINAL - heldout[r]["coverage"])
                for r in dense_regimes
            )
            / sum(heldout[r]["n_test"] for r in dense_regimes)
        )
    else:
        gap = None

    # Strategy-conditional gap: same max(0, nominal - coverage) definition,
    # but over (regime, strategy) cells with n_test >= SPARSE_THRESHOLD. A
    # cell-sparse strategy is left in the per-strategy fields but not in the
    # aggregate.
    strategy_gap = None
    dense_cells = []
    for r in sorted(heldout):
        for s, n in heldout[r]["strategy_n_test"].items():
            if n >= SPARSE_THRESHOLD:
                dense_cells.append((r, s, n, heldout[r]["strategy_coverage"][s]))
    if dense_cells:
        strategy_gap = float(
            sum(n * max(0.0, NOMINAL - c) for _, _, n, c in dense_cells)
            / sum(n for _, _, n, _ in dense_cells)
        )

    report = {
        "nominal": NOMINAL,
        "heldout_regime_coverage": {
            str(r): heldout[r]["coverage"] for r in sorted(heldout)
        },
        "heldout_regime_width": {str(r): heldout[r]["width"] for r in sorted(heldout)},
        "heldout_regime_radius": {
            str(r): heldout[r]["radius"] for r in sorted(heldout)
        },
        "heldout_strategy_coverage": {
            str(r): heldout[r]["strategy_coverage"] for r in sorted(heldout)
        },
        "heldout_strategy_width": {
            str(r): heldout[r]["strategy_width"] for r in sorted(heldout)
        },
        "heldout_strategy_n_test": {
            str(r): heldout[r]["strategy_n_test"] for r in sorted(heldout)
        },
        "calibration_radii_per_regime": {
            str(r): {str(k): v for k, v in heldout[r]["cal_radii"].items()}
            for r in sorted(heldout)
        },
        "random_split_coverage": random_split["coverage"],
        "random_split_width": random_split["width"],
        "random_split_strategy_coverage": random_split["strategy_coverage"],
        "random_split_strategy_width": random_split["strategy_width"],
        "random_split_cal_radii": {
            str(k): v for k, v in random_split["cal_radii"].items()
        },
        "trajectory_blocked_coverage": traj_blocked["coverage"],
        "trajectory_blocked_width": traj_blocked["width"],
        "trajectory_blocked_n_test": traj_blocked["n_test"],
        "trajectory_blocked_n_groups": {
            "train": traj_blocked["n_train_groups"],
            "cal": traj_blocked["n_cal_groups"],
            "test": traj_blocked["n_test_groups"],
        },
        "trajectory_blocked_group_coverage": {
            "mean": traj_blocked["group_coverage_mean"],
            "std": traj_blocked["group_coverage_std"],
            "se": traj_blocked["group_coverage_se"],
            "min": traj_blocked["group_coverage_min"],
            "max": traj_blocked["group_coverage_max"],
        },
        "path_blocked_coverage": path_blocked["coverage"],
        "path_blocked_width": path_blocked["width"],
        "path_blocked_n_test": path_blocked["n_test"],
        "path_blocked_n_groups": {
            "train": path_blocked["n_train_groups"],
            "cal": path_blocked["n_cal_groups"],
            "test": path_blocked["n_test_groups"],
        },
        "path_blocked_group_coverage": {
            "mean": path_blocked["group_coverage_mean"],
            "std": path_blocked["group_coverage_std"],
            "se": path_blocked["group_coverage_se"],
            "min": path_blocked["group_coverage_min"],
            "max": path_blocked["group_coverage_max"],
        },
        "fallback_exclude_regimes": [str(r) for r in sorted(exclude_regimes)],
        "fallback_mode": args.fallback,
        "fallback_partition": args.classes,
        "conditional_mode": args.conditional,
        "coverage_gap": gap,
        "strategy_coverage_gap": strategy_gap,
        "strategy_coverage_gap_definition": (
            "n_test-weighted mean of max(0, nominal - coverage) over "
            f"(regime, strategy) cells with n_test >= {SPARSE_THRESHOLD}; "
            "complements coverage_gap, which pools strategies"
        ),
        "coverage_gap_definition": (
            "n_test-weighted mean of max(0, nominal - coverage) over "
            f"non-sparse held-out regimes (n_test >= {SPARSE_THRESHOLD}); "
            "sparse regimes excluded from the aggregate and reported "
            "separately"
        ),
        "metric_direction": "lower_is_better",
        "n_test_per_regime": {str(r): heldout[r]["n_test"] for r in sorted(heldout)},
        "n_test_random_split": random_split["n_test"],
        "sparse_regimes": [str(r) for r in sorted(sparse_regimes)],
        "sparse": bool(sparse_regimes),
        "regimes_evaluated": [str(r) for r in sorted(heldout)],
        "n_train_per_regime": {str(r): heldout[r]["n_train"] for r in sorted(heldout)},
        "n_cal_per_regime": {str(r): heldout[r]["n_cal"] for r in sorted(heldout)},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten: {args.out}")
    print(json.dumps(report, indent=2))

    print("\n--- interpretation ---")
    for r in sorted(heldout):
        cov = heldout[r]["coverage"]
        tag = " [SPARSE n<50]" if r in sparse_regimes else ""
        print(f"heldout regime {r}: coverage={cov:.3f} "
              f"width={heldout[r]['width']:.4f} n={heldout[r]['n_test']}{tag}")
    print(f"random split:   coverage={random_split['coverage']:.3f} "
          f"width={random_split['width']:.4f} n={random_split['n_test']}")
    print(f"traj-blocked:   coverage={traj_blocked['coverage']:.3f} "
          f"width={traj_blocked['width']:.4f} n={traj_blocked['n_test']} "
          f"groups={traj_blocked['n_train_groups']}/"
          f"{traj_blocked['n_cal_groups']}/{traj_blocked['n_test_groups']} "
          f"group-SE={traj_blocked['group_coverage_se']:.4f}")
    print(f"path-blocked:   coverage={path_blocked['coverage']:.3f} "
          f"width={path_blocked['width']:.4f} n={path_blocked['n_test']} "
          f"groups={path_blocked['n_train_groups']}/"
          f"{path_blocked['n_cal_groups']}/{path_blocked['n_test_groups']} "
          f"group-SE={path_blocked['group_coverage_se']:.4f}")
    print("strategy-conditional held-out coverage (per regime):")
    for r in sorted(heldout):
        parts = [
            f"{s}={c:.3f}/{w:.4f}" for s, c in
            sorted(heldout[r]["strategy_coverage"].items())
            for w in [heldout[r]["strategy_width"][s]]
        ]
        n_sparse = sum(
            1 for n in heldout[r]["strategy_n_test"].values()
            if n < SPARSE_THRESHOLD
        )
        tag = f" [{n_sparse} sparse cell(s) excluded from strategy gap]" if n_sparse else ""
        print(f"  regime {r} (n={heldout[r]['n_test']}): " + ", ".join(parts) + tag)
    # under-coverage verdict is based on non-sparse regimes only (sparse ones
    # have too few test points to support a coverage claim)
    under = [r for r in dense_regimes if heldout[r]["coverage"] < NOMINAL - 0.03]
    ctrl_ok = abs(random_split["coverage"] - NOMINAL) <= 0.03
    gap_str = f"{gap:.4f}" if gap is not None else "n/a (all regimes sparse)"
    sg_str = (
        f"{strategy_gap:.4f}" if strategy_gap is not None
        else "n/a (all cells sparse)"
    )
    print(
        f"verdict (calendar regime labels): held-out regimes "
        f"{'UNDER-COVER (expected negative result: regime shift breaks standard conformal)' if under else 'do NOT under-cover'}"
        f" (under-covered: {sorted(under) if under else 'none'}); "
        f"iid control {'holds ~90% as required' if ctrl_ok else 'DEVIATES from 90% — check pipeline'}. "
        f"coverage_gap={gap_str} (lower_is_better, weighted over non-sparse regimes); "
        f"strategy_coverage_gap={sg_str}."
    )


if __name__ == "__main__":
    main()
