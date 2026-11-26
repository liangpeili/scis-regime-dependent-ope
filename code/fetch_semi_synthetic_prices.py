#!/usr/bin/env python3
"""Fetch hourly reference prices for the P1 semi-synthetic calibration.

Non-C1 task (S2, 2026-10-22).  The script only downloads public Yahoo Finance
chart data and writes a local cache under ``data/raw/semi_synthetic/``; it does
not touch the frozen P1 dataset, ``coverage_eval.py``, or any result used by
the C1 loop.

Source
------
Yahoo Finance chart API, 1-hour bars, 1-year range, pre/post market included:
    https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
        ?range=1y&interval=1h&includePrePost=true

The API is undocumented and may rate-limit; the cache is the reproducible
artifact for the calibration run.  Timestamps are stored in UTC and in
America/New_York.  ``close`` is the unadjusted hourly close; no split occurred
for the four symbols inside the fetch window (checked by the calibration
script's date range and a manual split check).

Usage (from projects/p1-c1-ope/code):
    ../../../.venv-c1/bin/python fetch_semi_synthetic_prices.py
    ../../../.venv-c1/bin/python fetch_semi_synthetic_prices.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("P1_DATA_ROOT", ROOT / "data" / "raw")) / "semi_synthetic"
DEFAULT_SYMBOLS = ("SPY", "QQQ", "TSLA", "NVDA")
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-hub/p1"}


def fetch_symbol(symbol: str, range_: str = "1y") -> tuple[pd.DataFrame, dict]:
    params = {
        "range": range_,
        "interval": "1h",
        "includePrePost": "true",
        "events": "div,splits",
    }
    response = requests.get(
        URL.format(symbol=symbol), params=params, headers=HEADERS, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    result = payload["chart"]["result"][0]
    meta = result["meta"]
    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True),
            "close": quote["close"],
            "volume": quote.get("volume"),
        }
    )
    frame["timestamp_et"] = frame["timestamp_utc"].dt.tz_convert(
        "America/New_York"
    )
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["close"]).drop_duplicates(
        subset=["timestamp_utc"]
    )
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    info = {
        "symbol": symbol,
        "exchange_timezone": meta.get("exchangeTimezoneName"),
        "currency": meta.get("currency"),
        "regular_market_time_utc": datetime.fromtimestamp(
            int(meta["regularMarketTime"]), tz=timezone.utc
        ).isoformat(),
        "n_bars": int(len(frame)),
        "first_bar_utc": frame["timestamp_utc"].iloc[0].isoformat(),
        "last_bar_utc": frame["timestamp_utc"].iloc[-1].isoformat(),
    }
    return frame, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--range", default="1y")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": "Yahoo Finance chart API",
        "url_template": URL + "?range={range}&interval=1h&includePrePost=true",
        "range": args.range,
        "interval": "1h",
        "include_pre_post": True,
        "fetched_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "symbols": {},
    }
    for symbol in args.symbols:
        out = RAW_DIR / f"{symbol}_1h.csv"
        if out.exists() and not args.force:
            frame = pd.read_csv(out, parse_dates=["timestamp_utc", "timestamp_et"])
            print(f"skip {symbol}: {out} ({len(frame)} rows)")
            manifest["symbols"][symbol] = {
                "path": str(out.relative_to(ROOT)),
                "n_bars": int(len(frame)),
                "cached": True,
            }
            continue
        frame, info = fetch_symbol(symbol, args.range)
        frame.to_csv(out, index=False)
        manifest["symbols"][symbol] = {
            **info,
            "path": str(out.relative_to(ROOT)),
            "cached": False,
        }
        print(f"wrote {out} ({len(frame)} rows)")
        time.sleep(0.5)

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
