#!/usr/bin/env python3
"""Fetch Uniswap v3 Swap logs for one pool and a block/time window.

S2.5 real-data spike (2026-10-22), non-C1: this script only reads an Ethereum
archive RPC and writes raw JSON plus a normalized CSV.  It does not touch the
frozen C1 dataset, ``coverage_eval.py``, or the synthetic environment.

The raw cache is the reproducible artifact: every ``eth_getLogs`` response is
stored under ``<raw-dir>/<pool>/`` before parsing.  Re-running the same window
reads the cache and does not spend RPC credits.

Usage (from projects/p1-c1-ope/code):
    ../../../.venv-c1/bin/python fetch_pool_swaps.py \
        --pool 0x31227b50eCCDC9C589826AA2D9E7C5619B1895Da \
        --from-ts 2026-10-13T00:00:00Z --to-ts 2026-10-20T00:00:00Z \
        --out-csv ../data/silver/tslaon_swaps.csv

The Alchemy key is read from ``$ALCHEMY_API_KEY`` or
``~/.config/alchemy/api_key``; it is never written into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

import pandas as pd
import requests

getcontext().prec = 50

# keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
SWAP_TOPIC = (
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
)

SEL_TOKEN0 = "0x0dfe1681"
SEL_TOKEN1 = "0xd21220a7"
SEL_FEE = "0xddca3f43"
SEL_DECIMALS = "0x313ce567"

DEFAULT_RAW_DIR = Path(os.environ.get(
    "P1_DATA_ROOT", Path(__file__).resolve().parents[1] / "data" / "raw"
)) / "p1_spike"
DEFAULT_API_KEY_FILE = Path("~/.config/alchemy/api_key")


class RpcError(RuntimeError):
    """JSON-RPC error returned by the provider."""

    def __init__(self, error: dict):
        self.code = error.get("code")
        self.message = str(error.get("message", error))
        super().__init__(f"RPC error {self.code}: {self.message}")


class Rpc:
    """Minimal JSON-RPC client with retry and batch support."""

    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()

    def _post(self, payload, timeout: float | None = None):
        try:
            response = self.session.post(
                self.url, json=payload, timeout=timeout or self.timeout
            )
        except requests.RequestException as exc:
            # Never include the request URL: it contains the API key.
            raise RuntimeError(
                f"request failed: {type(exc).__name__}"
            ) from None
        if response.status_code == 429 or response.status_code >= 500:
            raise RuntimeError(f"HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                f"HTTP {response.status_code} (non-JSON body)"
            ) from None
        if response.status_code >= 400 and not (
            isinstance(data, dict) and "error" in data
        ):
            raise RuntimeError(f"HTTP {response.status_code}")
        return data

    def call(
        self,
        method: str,
        params: list,
        timeout: float | None = None,
        tries: int = 6,
    ):
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        last: Exception | None = None
        for attempt in range(tries):
            try:
                data = self._post(payload, timeout=timeout)
                if "error" in data:
                    raise RpcError(data["error"])
                return data["result"]
            except RpcError:
                # Range/limit errors must reach the caller so it can shrink
                # the block window; they are not transient.
                raise
            except Exception as exc:  # noqa: BLE001 - retry provider noise
                last = exc
                if attempt == tries - 1:
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"{method} failed after {tries} tries: {last}")

    def batch(self, calls: list[tuple[str, list]], timeout: float | None = None):
        payload = [
            {"id": i, "jsonrpc": "2.0", "method": method, "params": params}
            for i, (method, params) in enumerate(calls)
        ]
        data = self._post(payload, timeout=timeout)
        if isinstance(data, dict):
            raise RpcError(data.get("error", data))
        by_id = {item["id"]: item for item in data}
        results = []
        for i in range(len(calls)):
            item = by_id.get(i)
            if item is None:
                raise RuntimeError(f"batch response missing id {i}")
            if "error" in item:
                raise RpcError(item["error"])
            results.append(item["result"])
        return results


def resolve_rpc_url(args) -> str:
    """Resolve the Alchemy endpoint without logging the key."""
    if args.rpc_url:
        return args.rpc_url
    key = os.environ.get("ALCHEMY_API_KEY")
    if not key:
        key_file = Path(args.api_key_file).expanduser()
        if key_file.exists():
            key = key_file.read_text().strip()
    if not key:
        raise SystemExit(
            "Set ALCHEMY_API_KEY or pass --api-key-file / --rpc-url"
        )
    return f"https://eth-mainnet.g.alchemy.com/v2/{key}"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def block_timestamp(rpc: Rpc, block: int) -> int:
    data = rpc.call("eth_getBlockByNumber", [hex(block), False])
    return int(data["timestamp"], 16)


def find_block_by_ts(
    rpc: Rpc, timestamp: int, lo: int = 1, hi: int | None = None
) -> int:
    """First block whose timestamp is >= ``timestamp``."""
    if hi is None:
        hi = int(rpc.call("eth_blockNumber", []), 16)
    while lo < hi:
        mid = (lo + hi) // 2
        if block_timestamp(rpc, mid) < timestamp:
            lo = mid + 1
        else:
            hi = mid
    return lo


def fetch_logs(
    rpc: Rpc,
    address: str,
    topic0: str | None,
    from_block: int,
    to_block: int,
    cache_dir: Path,
    chunk: int = 50_000,
) -> list[dict]:
    """Fetch logs with an adaptive block window and on-disk response cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict] = []
    current = from_block
    while current <= to_block:
        end = min(current + chunk - 1, to_block)
        cache = cache_dir / f"logs_{current}_{end}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
            logs.extend(payload["result"])
            current = end + 1
            continue

        params = {
            "address": address,
            "fromBlock": hex(current),
            "toBlock": hex(end),
        }
        if topic0:
            params["topics"] = [topic0]
        try:
            result = rpc.call(
                "eth_getLogs", [params], timeout=120
            )
        except RpcError as exc:
            text = str(exc).lower()
            range_error = any(
                key in text
                for key in (
                    "range",
                    "limit",
                    "too many",
                    "exceed",
                    "10000",
                    "10,000",
                    "response size",
                )
            )
            if not range_error:
                raise
            smaller = max(chunk // 2, 500)
            if smaller == chunk:
                raise
            print(
                f"  range error at {current}-{end}; "
                f"chunk {chunk} -> {smaller}",
                flush=True,
            )
            chunk = smaller
            continue

        cache.write_text(
            json.dumps(
                {
                    "address": address,
                    "topic0": topic0,
                    "from_block": current,
                    "to_block": end,
                    "result": result,
                },
                separators=(",", ":"),
            )
        )
        logs.extend(result)
        current = end + 1
        print(
            f"  fetched through {end} ({len(logs)} logs)",
            flush=True,
        )
    return logs


def fetch_block_times(
    rpc: Rpc, block_numbers: list[int], cache_path: Path
) -> dict[int, int]:
    """Resolve block timestamps in JSON-RPC batches, caching the result."""
    cache: dict[str, int] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    wanted = {int(block) for block in block_numbers}
    missing = sorted(wanted - {int(key) for key in cache})
    for start in range(0, len(missing), 100):
        chunk = missing[start : start + 100]
        calls = [
            ("eth_getBlockByNumber", [hex(block), False]) for block in chunk
        ]
        results = rpc.batch(calls, timeout=120)
        for block, result in zip(chunk, results):
            if result is not None:
                cache[str(block)] = int(result["timestamp"], 16)
        cache_path.write_text(json.dumps(cache))
        print(
            f"  block times {min(start + 100, len(missing))}/{len(missing)}",
            flush=True,
        )
    return {block: cache[str(block)] for block in wanted}


def call_address(rpc: Rpc, to: str, selector: str) -> str:
    raw = rpc.call("eth_call", [{"to": to, "data": selector}, "latest"])
    return "0x" + raw[-40:].lower()


def call_uint(rpc: Rpc, to: str, selector: str) -> int:
    raw = rpc.call("eth_call", [{"to": to, "data": selector}, "latest"])
    return int(raw, 16)


def decode_swap(log: dict) -> dict:
    """Decode a Uniswap v3 Swap event into a flat record."""
    topics = log["topics"]
    if len(topics) < 3:
        raise ValueError(f"unexpected Swap topics: {topics}")
    data = log["data"][2:]
    words = [data[i : i + 64] for i in range(0, len(data), 64)]
    if len(words) != 5:
        raise ValueError(f"unexpected Swap data words: {len(words)}")

    def signed(word: str) -> int:
        value = int(word, 16)
        if value >= 1 << 255:
            value -= 1 << 256
        return value

    return {
        "block_number": int(log["blockNumber"], 16),
        "tx_hash": log["transactionHash"],
        "log_index": int(log["logIndex"], 16),
        "sender": "0x" + topics[1][-40:],
        "recipient": "0x" + topics[2][-40:],
        "amount0": signed(words[0]),
        "amount1": signed(words[1]),
        "sqrt_price_x96": int(words[2], 16),
        "liquidity": int(words[3], 16),
        "tick": signed(words[4]),
    }


def pool_prices(
    sqrt_price_x96: int, decimals0: int, decimals1: int
) -> tuple[Decimal, Decimal]:
    """Human-unit prices implied by a v3 sqrtPriceX96.

    Returns ``(token1 per token0, token0 per token1)``.
    """
    ratio = (Decimal(sqrt_price_x96) / Decimal(2**96)) ** 2
    token1_per_token0 = ratio * (Decimal(10) ** (decimals0 - decimals1))
    if token1_per_token0 == 0:
        return token1_per_token0, Decimal("NaN")
    return token1_per_token0, Decimal(1) / token1_per_token0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--from-ts", help="inclusive ISO-8601 UTC start")
    parser.add_argument("--to-ts", help="exclusive ISO-8601 UTC end")
    parser.add_argument("--from-block", type=int)
    parser.add_argument("--to-block", type=int)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--summary-json", help="optional path for a small run summary"
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--chunk-blocks", type=int, default=50_000)
    parser.add_argument("--rpc-url")
    parser.add_argument(
        "--api-key-file", default=str(DEFAULT_API_KEY_FILE)
    )
    args = parser.parse_args()

    if args.from_block is not None or args.to_block is not None:
        if args.from_block is None or args.to_block is None:
            parser.error("--from-block and --to-block must be used together")
        from_block, to_block = args.from_block, args.to_block
    else:
        if not args.from_ts or not args.to_ts:
            parser.error("pass --from-ts/--to-ts or --from-block/--to-block")
        rpc = Rpc(resolve_rpc_url(args))
        from_block = find_block_by_ts(
            rpc, int(parse_utc(args.from_ts).timestamp())
        )
        to_block = (
            find_block_by_ts(rpc, int(parse_utc(args.to_ts).timestamp())) - 1
        )
    if to_block < from_block:
        parser.error("empty block range")

    rpc = Rpc(resolve_rpc_url(args))
    pool = args.pool
    token0 = call_address(rpc, pool, SEL_TOKEN0)
    token1 = call_address(rpc, pool, SEL_TOKEN1)
    decimals0 = call_uint(rpc, token0, SEL_DECIMALS)
    decimals1 = call_uint(rpc, token1, SEL_DECIMALS)
    fee = call_uint(rpc, pool, SEL_FEE)

    print(
        f"pool {pool} token0={token0} ({decimals0}d) "
        f"token1={token1} ({decimals1}d) fee={fee}",
        flush=True,
    )
    print(f"blocks {from_block}..{to_block}", flush=True)

    raw_dir = Path(args.raw_dir) / pool.lower()
    logs = fetch_logs(
        rpc,
        pool,
        SWAP_TOPIC,
        from_block,
        to_block,
        raw_dir,
        chunk=args.chunk_blocks,
    )
    records = [decode_swap(log) for log in logs]
    if not records:
        raise SystemExit("no Swap logs in the requested window")

    frame = pd.DataFrame.from_records(records)
    block_times = fetch_block_times(
        rpc,
        frame["block_number"].tolist(),
        raw_dir / "block_times.json",
    )
    frame["block_time_utc"] = pd.to_datetime(
        [block_times[b] for b in frame["block_number"]],
        unit="s",
        utc=True,
    )
    prices = [
        pool_prices(value, decimals0, decimals1)
        for value in frame["sqrt_price_x96"]
    ]
    frame["price_token1_per_token0"] = [float(p[0]) for p in prices]
    frame["price_token0_per_token1"] = [float(p[1]) for p in prices]
    frame = frame.sort_values(
        ["block_number", "log_index"]
    ).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    price = frame["price_token0_per_token1"]
    summary = {
        "source": "Alchemy eth-mainnet",
        "pool": pool,
        "token0": token0,
        "token0_decimals": decimals0,
        "token1": token1,
        "token1_decimals": decimals1,
        "fee": fee,
        "from_block": from_block,
        "to_block": to_block,
        "n_swaps": int(len(frame)),
        "n_blocks": int(frame["block_number"].nunique()),
        "first_swap_utc": frame["block_time_utc"].iloc[0].isoformat(),
        "last_swap_utc": frame["block_time_utc"].iloc[-1].isoformat(),
        "price_token0_per_token1": {
            "first": float(price.iloc[0]),
            "last": float(price.iloc[-1]),
            "min": float(price.min()),
            "max": float(price.max()),
        },
        "raw_cache": str(raw_dir),
        "csv": str(out_csv),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
