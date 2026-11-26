#!/usr/bin/env python3
"""Assemble one real-data slice event from pool swaps and Chainlink rounds.

S2.5 real-data slice (2026-10-22), non-C1.  This script reads the normalized
swap CSV produced by ``fetch_pool_swaps.py`` and the P2 event table, fetches
Chainlink ``AnswerUpdated`` rounds for the matching window, and writes a
single-event diagnostic JSON.  It is a data-availability and boundary-check
tool, not the final real-path replay.

The event is classified in three layers (``drafts/real-data-slice-plan.md``
S4.4):

* ``usable``: a swap in the calendar freeze and a swap in the 2h after market
  open (G1-R layer A);
* ``main_replay_ok``: additionally pool burn-in and a <=2% archive boundary
  basis at both stale start and market open (layer B);
* ``economic_ok``: additionally non-thin post-open activity (layer C).

Usage (from projects/p1-c1-ope/code):
    ../../../.venv-c1/bin/python build_real_slice.py \
        --event-id TSLAon-eth-059 \
        --swaps-csv ../data/silver/tslaon_swaps.csv

The Alchemy key is read from ``$ALCHEMY_API_KEY`` or
``~/.config/alchemy/api_key``; it is never written into the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_pool_swaps import (  # noqa: E402
    DEFAULT_API_KEY_FILE,
    DEFAULT_RAW_DIR,
    Rpc,
    call_address,
    call_uint,
    fetch_logs,
    find_block_by_ts,
    pool_prices,
    resolve_rpc_url,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVENT_TABLE = (
    ROOT / "projects/p2-market-structure/data/event_table_skeleton.csv"
)
DEFAULT_POOL_REGISTRY = (
    ROOT / "projects/p2-market-structure/data/pool_registry.csv"
)

# keccak256("AnswerUpdated(int256,uint256,uint256)")
ANSWER_UPDATED_TOPIC = (
    "0x0559884fd3a460db3073b7fc896cc77986f16e378210ded43186175bf646fc5f"
)
SEL_DECIMALS = "0x313ce567"
SEL_TOKEN0 = "0x0dfe1681"
SEL_TOKEN1 = "0xd21220a7"
SEL_SLOT0 = "0x3850c7bd"
SEL_LATEST_ROUND_DATA = "0xfeaf968c"

# Quote assets used by the P1 slice.  The script still reads decimals on-chain;
# these addresses only decide which side of the pool is the quote.
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
QUOTE_TOKENS = {USDC, USDT}

# Chainlink RDD: TSLAon-USD (Calculated), Ethereum mainnet.
DEFAULT_AGGREGATOR = "0x9F6B06e826d3DF391285c695749F8f921F6972D9"
DEFAULT_FEED = "TSLAon-USD (Calculated)"

# Event-layer defaults (plan S4.4; overridable for sensitivity runs).
DEFAULT_BASIS_MAIN = 0.02
DEFAULT_BASIS_ROBUSTNESS = 0.10
DEFAULT_THIN_MIN_SWAPS = 3
DEFAULT_THIN_MIN_NOTIONAL_USDC = 1000.0
DEFAULT_BURN_IN_DAYS = 7.0


def decode_answer_updated(log: dict) -> dict:
    """Decode Chainlink ``AnswerUpdated`` (current, roundId indexed)."""
    topics = log["topics"]
    if len(topics) < 3:
        raise ValueError(f"unexpected AnswerUpdated topics: {topics}")
    current = int(topics[1], 16)
    if current >= 1 << 255:
        current -= 1 << 256
    updated_at = int(log["data"][2:66], 16)
    return {
        "block_number": int(log["blockNumber"], 16),
        "tx_hash": log["transactionHash"],
        "round_id": int(topics[2], 16),
        "answer": current,
        "updated_at": updated_at,
    }


def last_at_or_before(frame: pd.DataFrame, column: str, ts):
    subset = frame[frame[column] <= ts]
    return subset.iloc[-1] if len(subset) else None


def first_at_or_after(frame: pd.DataFrame, column: str, ts):
    subset = frame[frame[column] >= ts]
    return subset.iloc[0] if len(subset) else None


def basis(pool_price: float | None, oracle_price: float | None):
    if (
        pool_price is None
        or pd.isna(pool_price)
        or oracle_price is None
        or pd.isna(oracle_price)
        or oracle_price == 0
    ):
        return None
    return (pool_price - oracle_price) / oracle_price


def signed_int(word: str) -> int:
    value = int(word, 16)
    if value >= 1 << 255:
        value -= 1 << 256
    return value


def read_slot0_price(
    rpc: Rpc,
    pool: str,
    timestamp,
    decimals0: int,
    decimals1: int,
    quote_is_token0: bool,
) -> tuple[float | None, int | None]:
    """Archive ``slot0()`` price at the first block at/after ``timestamp``.

    Returns ``(quote per Ondo token, block number)``.  ``pool_prices`` returns
    ``(token1 per token0, token0 per token1)``, so the quote side decides the
    index.
    """
    block = find_block_by_ts(rpc, int(timestamp.timestamp()))
    raw = rpc.call(
        "eth_call",
        [{"to": pool, "data": SEL_SLOT0}, hex(block)],
    )
    sqrt_price = int(raw[2:66], 16)
    prices = pool_prices(sqrt_price, decimals0, decimals1)
    index = 1 if quote_is_token0 else 0
    value = float(prices[index])
    return value, block


def read_pool_created_at(registry_path: str, pool: str):
    path = Path(registry_path)
    if not path.exists():
        return None
    registry = pd.read_csv(path)
    if "pair_address" not in registry or "pair_created_at" not in registry:
        return None
    row = registry[
        registry["pair_address"].str.lower() == pool.lower()
    ]
    if row.empty:
        return None
    created_ms = row.iloc[0]["pair_created_at"]
    if pd.isna(created_ms):
        return None
    return pd.to_datetime(int(created_ms), unit="ms", utc=True)


def quote_notional(
    frame: pd.DataFrame,
    mask: pd.Series,
    quote_is_token0: bool,
    quote_decimals: int,
) -> float:
    if not mask.any():
        return 0.0
    column = "amount0" if quote_is_token0 else "amount1"
    return float(frame.loc[mask, column].abs().sum() / (10**quote_decimals))


def max_gap_hours(times) -> float | None:
    if len(times) < 2:
        return None
    ordered = sorted(times)
    return max(
        (b - a).total_seconds() / 3600.0
        for a, b in zip(ordered, ordered[1:])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", default="TSLAon-eth-059")
    parser.add_argument("--event-table", default=str(DEFAULT_EVENT_TABLE))
    parser.add_argument("--swaps-csv", required=True)
    parser.add_argument(
        "--pool", default="0x31227b50eCCDC9C589826AA2D9E7C5619B1895Da"
    )
    parser.add_argument("--aggregator", default=DEFAULT_AGGREGATOR)
    parser.add_argument("--feed-name", default=DEFAULT_FEED)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument(
        "--pool-registry", default=str(DEFAULT_POOL_REGISTRY)
    )
    parser.add_argument(
        "--pool-created-at",
        help="ISO-8601 pool creation time; overrides the registry lookup",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "projects/p1-c1-ope/results/spike_real_slice.json"),
    )
    parser.add_argument("--rpc-url")
    parser.add_argument(
        "--api-key-file", default=str(DEFAULT_API_KEY_FILE)
    )
    parser.add_argument(
        "--basis-main-threshold", type=float, default=DEFAULT_BASIS_MAIN
    )
    parser.add_argument(
        "--basis-robustness-threshold",
        type=float,
        default=DEFAULT_BASIS_ROBUSTNESS,
    )
    parser.add_argument(
        "--thin-min-swaps", type=int, default=DEFAULT_THIN_MIN_SWAPS
    )
    parser.add_argument(
        "--thin-min-notional-usdc",
        type=float,
        default=DEFAULT_THIN_MIN_NOTIONAL_USDC,
    )
    parser.add_argument(
        "--burn-in-days", type=float, default=DEFAULT_BURN_IN_DAYS
    )
    parser.add_argument(
        "--swaps-from-ts",
        help=(
            "Fetch-window start (ISO-8601 UTC) of the swaps CSV; used for "
            "coverage checks. Defaults to the first Swap timestamp."
        ),
    )
    parser.add_argument(
        "--swaps-to-ts",
        help=(
            "Fetch-window end (ISO-8601 UTC) of the swaps CSV; used for "
            "coverage checks. Defaults to the last Swap timestamp."
        ),
    )
    args = parser.parse_args()

    events = pd.read_csv(args.event_table)
    match = events[events["event_id"] == args.event_id]
    if match.empty:
        raise SystemExit(f"event not found: {args.event_id}")
    event = match.iloc[0]
    freeze_start = pd.to_datetime(event["freeze_start_utc"], utc=True)
    freeze_end = pd.to_datetime(event["freeze_end_utc"], utc=True)

    swaps = pd.read_csv(args.swaps_csv, parse_dates=["block_time_utc"])
    swaps["block_time_utc"] = pd.to_datetime(
        swaps["block_time_utc"], utc=True
    )
    swaps = swaps.sort_values(
        ["block_number", "log_index"]
    ).reset_index(drop=True)
    if swaps.empty:
        raise SystemExit(f"no swaps in {args.swaps_csv}")

    rpc = Rpc(resolve_rpc_url(args))
    aggregator = args.aggregator
    oracle_decimals = call_uint(rpc, aggregator, SEL_DECIMALS)

    # Token order and quote side decide both the swap-price column and the
    # archive slot0 direction.  NVDAon is token0=Ondo, token1=USDC, so the
    # old hard-coded ``price_token0_per_token1`` was wrong for that pool.
    token0 = call_address(rpc, args.pool, SEL_TOKEN0)
    token1 = call_address(rpc, args.pool, SEL_TOKEN1)
    decimals0 = call_uint(rpc, token0, SEL_DECIMALS)
    decimals1 = call_uint(rpc, token1, SEL_DECIMALS)
    token0_lower = token0.lower()
    token1_lower = token1.lower()
    quote_is_token0 = token0_lower in QUOTE_TOKENS
    quote_is_token1 = token1_lower in QUOTE_TOKENS
    if not (quote_is_token0 or quote_is_token1):
        raise SystemExit(
            f"pool {args.pool} is not quoted in USDC/USDT: "
            f"token0={token0}, token1={token1}"
        )
    if quote_is_token0:
        price_col = "price_token0_per_token1"  # quote per Ondo token
        quote_token = token0
        quote_decimals = decimals0
    else:
        price_col = "price_token1_per_token0"  # quote per Ondo token
        quote_token = token1
        quote_decimals = decimals1

    fetch_start = freeze_start - timedelta(days=2)
    fetch_end = freeze_end + timedelta(days=1)
    from_block = find_block_by_ts(rpc, int(fetch_start.timestamp()))
    to_block = (
        find_block_by_ts(rpc, int(fetch_end.timestamp())) - 1
    )
    oracle_dir = Path(args.raw_dir) / "oracle" / aggregator.lower()
    logs = fetch_logs(
        rpc,
        aggregator,
        ANSWER_UPDATED_TOPIC,
        from_block,
        to_block,
        oracle_dir,
    )
    rounds = pd.DataFrame.from_records(
        [decode_answer_updated(log) for log in logs]
    )
    if rounds.empty:
        raise SystemExit("no Chainlink rounds in the requested window")
    rounds["updated_at_utc"] = pd.to_datetime(
        rounds["updated_at"], unit="s", utc=True
    )
    rounds = rounds.sort_values("updated_at").reset_index(drop=True)
    rounds["answer_scaled"] = rounds["answer"] / (10**oracle_decimals)
    rounds["gap_hours"] = (
        rounds["updated_at"].diff().fillna(0) / 3600.0
    )

    # Longest oracle gap that overlaps the freeze event: this is the actual
    # stale interval, which can start after the 20:00 ET market close.
    gap_mask = (
        (rounds["updated_at"].shift(1) < int(freeze_end.timestamp()))
        & (rounds["updated_at"] > int(freeze_start.timestamp()))
        & (rounds["gap_hours"] > 1.0)
    )
    gap_rows = rounds[gap_mask]
    longest = (
        gap_rows.sort_values("gap_hours").iloc[-1]
        if len(gap_rows)
        else None
    )

    before_freeze = last_at_or_before(
        rounds, "updated_at_utc", freeze_start
    )
    after_freeze = first_at_or_after(
        rounds, "updated_at_utc", freeze_end
    )
    oracle_at_market_open = last_at_or_before(
        rounds, "updated_at_utc", freeze_end
    )

    oracle_stale = None
    oracle_open = None
    stale_start = None
    open_time = None
    gap_hours = None
    jump = None
    jump_pct = None
    if longest is not None:
        gap_index = longest.name
        current_index = rounds.index.get_loc(gap_index)
        if current_index > 0:
            stale_row = rounds.iloc[current_index - 1]
            stale_start = stale_row["updated_at_utc"]
            open_time = longest["updated_at_utc"]
            gap_hours = float(longest["gap_hours"])
            oracle_stale = float(stale_row["answer_scaled"])
            oracle_open = float(longest["answer_scaled"])
            jump = oracle_open - oracle_stale
            if oracle_stale:
                jump_pct = jump / oracle_stale

    pool_at_freeze = last_at_or_before(
        swaps, "block_time_utc", freeze_start
    )
    pool_at_open = first_at_or_after(swaps, "block_time_utc", freeze_end)
    pool_at_stale = (
        last_at_or_before(swaps, "block_time_utc", stale_start)
        if stale_start is not None
        else None
    )
    pool_at_open_round = (
        first_at_or_after(swaps, "block_time_utc", open_time)
        if open_time is not None
        else None
    )

    # Archive eth_call cross-checks.  These are the boundary-basis checks used
    # by the main-analysis rule, so each timestamp gets its own slot0 call.
    freeze_block = find_block_by_ts(rpc, int(freeze_start.timestamp()))
    slot0_at_freeze, _ = read_slot0_price(
        rpc,
        args.pool,
        freeze_start,
        decimals0,
        decimals1,
        quote_is_token0,
    )
    slot0_at_stale = None
    stale_block = None
    if stale_start is not None:
        slot0_at_stale, stale_block = read_slot0_price(
            rpc,
            args.pool,
            stale_start,
            decimals0,
            decimals1,
            quote_is_token0,
        )
    slot0_at_market_open, market_open_block = read_slot0_price(
        rpc,
        args.pool,
        freeze_end,
        decimals0,
        decimals1,
        quote_is_token0,
    )
    slot0_at_oracle_open = None
    oracle_open_block = None
    if open_time is not None:
        slot0_at_oracle_open, oracle_open_block = read_slot0_price(
            rpc,
            args.pool,
            open_time,
            decimals0,
            decimals1,
            quote_is_token0,
        )

    round_raw = rpc.call(
        "eth_call",
        [
            {"to": aggregator, "data": SEL_LATEST_ROUND_DATA},
            hex(freeze_block),
        ],
    )
    round_words = [
        round_raw[2 + i * 64 : 2 + (i + 1) * 64] for i in range(5)
    ]
    round_answer = signed_int(round_words[1]) / (10**oracle_decimals)
    round_updated = int(round_words[3], 16)

    freeze_mask = (
        (swaps["block_time_utc"] > freeze_start)
        & (swaps["block_time_utc"] <= freeze_end)
    )
    post_mask = (
        (swaps["block_time_utc"] > freeze_end)
        & (swaps["block_time_utc"] <= freeze_end + timedelta(hours=2))
    )
    pre_2h_mask = (
        (swaps["block_time_utc"] >= freeze_start - timedelta(hours=2))
        & (swaps["block_time_utc"] < freeze_start)
    )
    pre_24h_mask = (
        (swaps["block_time_utc"] >= freeze_start - timedelta(hours=24))
        & (swaps["block_time_utc"] < freeze_start)
    )
    pre_72h_mask = (
        (swaps["block_time_utc"] >= freeze_start - timedelta(hours=72))
        & (swaps["block_time_utc"] < freeze_start)
    )
    pre_7d_mask = (
        (swaps["block_time_utc"] >= freeze_start - timedelta(days=7))
        & (swaps["block_time_utc"] < freeze_start)
    )
    oracle_gap_mask = (
        (swaps["block_time_utc"] > stale_start)
        & (swaps["block_time_utc"] <= open_time)
        if stale_start is not None and open_time is not None
        else pd.Series(False, index=swaps.index)
    )
    post_oracle_2h_mask = (
        (swaps["block_time_utc"] > open_time)
        & (swaps["block_time_utc"] <= open_time + timedelta(hours=2))
        if open_time is not None
        else pd.Series(False, index=swaps.index)
    )

    def price_of(row):
        return None if row is None else float(row[price_col])

    pool_move = None
    if pool_at_stale is not None and pool_at_open_round is not None:
        start_price = float(pool_at_stale[price_col])
        if start_price:
            pool_move = (
                float(pool_at_open_round[price_col]) - start_price
            ) / start_price

    post_open_notional = quote_notional(
        swaps, post_mask, quote_is_token0, quote_decimals
    )
    freeze_notional = quote_notional(
        swaps, freeze_mask, quote_is_token0, quote_decimals
    )
    thin = (
        int(post_mask.sum()) < args.thin_min_swaps
        or post_open_notional < args.thin_min_notional_usdc
    )

    pool_created_at = None
    if args.pool_created_at:
        pool_created_at = pd.to_datetime(
            args.pool_created_at, utc=True
        )
    else:
        pool_created_at = read_pool_created_at(
            args.pool_registry, args.pool
        )
    pool_age_days = (
        None
        if pool_created_at is None
        else (freeze_start - pool_created_at).total_seconds() / 86400.0
    )

    csv_start = swaps["block_time_utc"].iloc[0]
    csv_end = swaps["block_time_utc"].iloc[-1]
    # Coverage is about the *fetch window*, not the first/last observed Swap:
    # a pool with no trades before an event still has a covered (empty) path.
    window_start = (
        pd.to_datetime(args.swaps_from_ts, utc=True)
        if args.swaps_from_ts
        else csv_start
    )
    window_end = (
        pd.to_datetime(args.swaps_to_ts, utc=True)
        if args.swaps_to_ts
        else csv_end
    )
    pre_7d_window_covered = (
        window_start <= freeze_start - timedelta(days=7)
    )
    n_pre_7d = int(pre_7d_mask.sum())
    if pool_age_days is None:
        burn_in_ok = None
        burn_in_reason = "unknown_pool_age"
    elif pool_age_days < args.burn_in_days:
        burn_in_ok = False
        burn_in_reason = f"pool_age_{pool_age_days:.2f}d_lt_{args.burn_in_days:g}d"
    elif n_pre_7d >= 1:
        burn_in_ok = True
        burn_in_reason = "ok"
    elif csv_start > freeze_start:
        burn_in_ok = False
        burn_in_reason = "no_pre_freeze_swap_in_csv"
    elif not pre_7d_window_covered:
        burn_in_ok = None
        burn_in_reason = "pre_7d_window_truncated"
    else:
        burn_in_ok = False
        burn_in_reason = "no_swap_in_pre_7d"

    basis_stale_archive = basis(slot0_at_stale, oracle_stale)
    oracle_market_open = (
        None
        if oracle_at_market_open is None
        else float(oracle_at_market_open["answer_scaled"])
    )
    basis_market_open = basis(
        slot0_at_market_open, oracle_market_open
    )
    basis_checks = [
        value
        for value in (basis_stale_archive, basis_market_open)
        if value is not None
    ]
    if len(basis_checks) < 2:
        basis_tier = "unknown"
    else:
        max_abs_basis = max(abs(value) for value in basis_checks)
        if max_abs_basis <= args.basis_main_threshold:
            basis_tier = "main"
        elif max_abs_basis <= args.basis_robustness_threshold:
            basis_tier = "robustness"
        else:
            basis_tier = "excluded"

    usable = bool(freeze_mask.any() and post_mask.any())
    event_window_covered = bool(
        window_start <= freeze_start
        and window_end >= freeze_end + timedelta(hours=2)
    )
    has_oracle_anchor = bool(
        oracle_stale is not None
        and oracle_open is not None
        and before_freeze is not None
        and after_freeze is not None
    )
    g1r_usable = bool(
        usable
        and event_window_covered
        and has_oracle_anchor
        and basis_tier != "excluded"
    )
    main_replay_ok = bool(
        g1r_usable and burn_in_ok is True and basis_tier == "main"
    )
    economic_ok = bool(main_replay_ok and not thin)

    freeze_times = [freeze_start]
    freeze_times.extend(swaps.loc[freeze_mask, "block_time_utc"].tolist())
    freeze_times.append(freeze_end)
    max_no_trade_gap = max_gap_hours(freeze_times)

    result = {
        "event": {
            "event_id": event["event_id"],
            "symbol": event["ondo_symbol"],
            "type": event["type"],
            "freeze_start_utc": freeze_start.isoformat(),
            "freeze_end_utc": freeze_end.isoformat(),
            "tau_hours": float(event["tau_hours"]),
        },
        "pool": {
            "address": args.pool,
            "token0": token0,
            "token1": token1,
            "decimals0": int(decimals0),
            "decimals1": int(decimals1),
            "quote_token": quote_token,
            "quote_is_token0": bool(quote_is_token0),
            "price_quote_per_ondo": price_col,
            "pool_created_at_utc": (
                None
                if pool_created_at is None
                else pool_created_at.isoformat()
            ),
            "pool_age_days": pool_age_days,
            "n_swaps": int(len(swaps)),
            "csv_window_utc": [
                csv_start.isoformat(),
                csv_end.isoformat(),
            ],
            "fetch_window_utc": [
                window_start.isoformat(),
                window_end.isoformat(),
            ],
            "first_swap_utc": csv_start.isoformat(),
            "last_swap_utc": csv_end.isoformat(),
            "n_swaps_pre_2h": int(pre_2h_mask.sum()),
            "n_swaps_pre_24h": int(pre_24h_mask.sum()),
            "n_swaps_pre_72h": int(pre_72h_mask.sum()),
            "n_swaps_pre_7d": n_pre_7d,
            "pre_7d_window_covered": bool(pre_7d_window_covered),
            "last_swap_before_freeze_utc": (
                None
                if pool_at_freeze is None
                else pool_at_freeze["block_time_utc"].isoformat()
            ),
            "hours_since_last_pre_freeze_swap": (
                None
                if pool_at_freeze is None
                else (
                    freeze_start - pool_at_freeze["block_time_utc"]
                ).total_seconds()
                / 3600.0
            ),
            "days_first_window_swap_to_freeze_start": (
                (freeze_start - csv_start).total_seconds() / 86400.0
            ),
            "n_swaps_freeze": int(freeze_mask.sum()),
            "n_swaps_post_2h": int(post_mask.sum()),
            "n_swaps_oracle_gap": int(oracle_gap_mask.sum()),
            "n_swaps_post_oracle_2h": int(post_oracle_2h_mask.sum()),
            "freeze_notional_usdc": freeze_notional,
            "post_open_notional_usdc": post_open_notional,
            "first_post_open_swap_utc": (
                None
                if pool_at_open is None
                else pool_at_open["block_time_utc"].isoformat()
            ),
            "max_no_trade_gap_hours_in_freeze": max_no_trade_gap,
            "price_at_freeze_start": price_of(pool_at_freeze),
            "price_at_freeze_end": price_of(pool_at_open),
            "price_at_stale_start": price_of(pool_at_stale),
            "price_at_oracle_open": price_of(pool_at_open_round),
            "pool_move_stale_to_open": pool_move,
        },
        "oracle": {
            "feed": args.feed_name,
            "aggregator": aggregator,
            "decimals": int(oracle_decimals),
            "n_rounds": int(len(rounds)),
            "window_utc": [
                rounds["updated_at_utc"].iloc[0].isoformat(),
                rounds["updated_at_utc"].iloc[-1].isoformat(),
            ],
            "last_before_freeze": (
                None
                if before_freeze is None
                else {
                    "updated_at_utc": before_freeze[
                        "updated_at_utc"
                    ].isoformat(),
                    "answer": float(before_freeze["answer_scaled"]),
                }
            ),
            "first_after_freeze": (
                None
                if after_freeze is None
                else {
                    "updated_at_utc": after_freeze[
                        "updated_at_utc"
                    ].isoformat(),
                    "answer": float(after_freeze["answer_scaled"]),
                }
            ),
            "last_at_market_open": (
                None
                if oracle_at_market_open is None
                else {
                    "updated_at_utc": oracle_at_market_open[
                        "updated_at_utc"
                    ].isoformat(),
                    "answer": oracle_market_open,
                }
            ),
            "longest_gap": (
                None
                if longest is None
                else {
                    "stale_start_utc": stale_start.isoformat(),
                    "open_time_utc": open_time.isoformat(),
                    "gap_hours": gap_hours,
                    "stale_value": oracle_stale,
                    "open_value": oracle_open,
                    "jump": jump,
                    "jump_pct": jump_pct,
                }
            ),
        },
        "comparison": {
            "basis_at_stale_start": basis(
                price_of(pool_at_stale)
                if pool_at_stale is not None
                else slot0_at_stale,
                oracle_stale,
            ),
            "basis_at_stale_start_archive": basis_stale_archive,
            "basis_at_oracle_gap_open": basis(
                slot0_at_oracle_open, oracle_open
            ),
            "basis_at_oracle_gap_open_swap": basis(
                price_of(pool_at_open_round), oracle_open
            ),
            "basis_at_market_open": basis_market_open,
        },
        "archive_check": {
            "freeze_start_block": freeze_block,
            "slot0_at_freeze_start": slot0_at_freeze,
            "stale_start_block": stale_block,
            "slot0_at_stale_start": slot0_at_stale,
            "market_open_block": market_open_block,
            "slot0_at_freeze_end": slot0_at_market_open,
            "oracle_open_block": oracle_open_block,
            "slot0_at_oracle_open": slot0_at_oracle_open,
            "slot0_vs_swap_abs_diff": (
                None
                if price_of(pool_at_freeze) is None
                or slot0_at_freeze is None
                else abs(slot0_at_freeze - price_of(pool_at_freeze))
            ),
            "latest_round_data_answer": round_answer,
            "latest_round_data_updated_at_utc": pd.to_datetime(
                round_updated, unit="s", utc=True
            ).isoformat(),
            "last_before_freeze_answer": (
                None
                if before_freeze is None
                else float(before_freeze["answer_scaled"])
            ),
            "oracle_at_market_open": oracle_market_open,
        },
        "coverage": {
            "has_swap_in_freeze": bool(freeze_mask.any()),
            "has_swap_post_2h": bool(post_mask.any()),
            "event_window_covered": event_window_covered,
            "has_oracle_anchor": has_oracle_anchor,
            "usable": usable,
            "g1r_usable": g1r_usable,
            "burn_in_ok": burn_in_ok,
            "burn_in_reason": burn_in_reason,
            "basis_tier": basis_tier,
            "thin": bool(thin),
            "main_replay_ok": main_replay_ok,
            "economic_ok": economic_ok,
        },
        "rules": {
            "burn_in_days": args.burn_in_days,
            "basis_main_threshold": args.basis_main_threshold,
            "basis_robustness_threshold": (
                args.basis_robustness_threshold
            ),
            "thin_min_swaps": args.thin_min_swaps,
            "thin_min_notional_usdc": args.thin_min_notional_usdc,
        },
        "limits": [
            "Single-event diagnostic; the 9-month production slice is "
            "assembled separately.",
            "Pool price is the post-swap sqrtPriceX96 of the selected v3 "
            "pool; no price impact or fee accrual is modeled here.",
            "Oracle value is the last AnswerUpdated before the longest gap; "
            "the gap may start after the 20:00 ET market close.",
            "Archive boundary basis uses slot0 at the block at/after the "
            "timestamp, not the first post-open swap.",
            "Real-path replay comes later; this JSON only checks data "
            "availability and boundary rules.",
        ],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
