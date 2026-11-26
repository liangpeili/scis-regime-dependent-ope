"""Compact synthetic environment for P1 (observation-mapping drift).

Design goals
------------
P1 asks whether off-policy evaluation / conformal coverage survives when the
mapping from the latent reference price S to the observable pool price P
changes across market-calendar regimes.  This module builds a deliberately
small simulator in which:

* the latent log-price follows the SAME mean-reverting random walk in every
  regime (identification assumption I1': regimes do not change the dynamics of
  S; they change only what the agent can observe);
* the observation mapping h_r differs by regime:
    OPEN       P = S * exp(iid noise)                 (anchored, informative)
    EXTENDED   P = S * exp(AR(1) deviation)           (24/5 stream, slightly lagged)
    WEEKEND    P = S_last_open * exp(AR(1) deviation) (stale anchor, uninformative)
    HOLIDAY    same as WEEKEND, longer duration
    HALT       P = P_prev                             (no trades)
* the LP position is valued with the exact Uniswap-v3 range formula; fees
  accrue only while in range; enter/rebalance/exit pay gas + slippage.
  `portfolio_value` is the *observable* pool-marked value; `reward_env` is the
  LP excess return over HODL: fees + change in fundamental IL (marked at the
  latent S) - costs, divided by initial notional.  In OPEN the pool mark and
  the fundamental mark nearly coincide; in freeze the observed mark is stale
  while the fundamental IL keeps moving.

The six logging policies are heuristic (no RL training in v1).  Their purpose
is to generate diverse, regime-dependent action data for method validation;
they are not a claim about optimal LP behaviour.

Calendar (hourly steps, 14-day window)
--------------------------------------
  day 4           : HOLIDAY (a Friday, so it merges into a 3-day freeze)
  day 7, 11:00-12:00 : HALT (a short trading halt)
  Sat/Sun         : WEEKEND
  Mon-Fri 09-16   : OPEN
  other weekday   : EXTENDED

Regime labels are calendar-derived and known in advance (no HMM smoothing, no
lookahead).  True prices are written only to a separate diagnostics file, not
to the evaluation dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V0 = 100_000.0          # initial notional (USD)
FEE_RATE = 0.003        # 30 bps fee tier
LIQUIDITY_SHARE = 0.01  # share of in-range liquidity owned by our LP
GAS_USD = 2.0           # per enter/exit leg
SLIPPAGE = 0.0005       # 5 bps on notional
BASE_VOLUME = 500_000.0 # hourly pool volume (USD), lognormal noise
SIGMA_BASE = 0.008      # hourly latent vol (same in every regime)
KAPPA = 0.003           # mean reversion of log price
MU_LOG = math.log(100.0)

NARROW = 0.05
WIDE = 0.30

REGIME_OPEN = 0
REGIME_EXTENDED = 1
REGIME_WEEKEND = 2
REGIME_HOLIDAY = 3
REGIME_HALT = 4

REGIME_NAMES = {
    REGIME_OPEN: "OPEN",
    REGIME_EXTENDED: "EXTENDED",
    REGIME_WEEKEND: "WEEKEND",
    REGIME_HOLIDAY: "HOLIDAY",
    REGIME_HALT: "HALT",
}

STRATEGIES = (
    "Passive_Wide",
    "Passive_Narrow",
    "Periodic_24h",
    "Threshold_2pct",
    "Exit_Freeze_50",
    "Stochastic_5pct",
)

WINDOW_DAYS = 14
HOLIDAY_DAY = 4
HALT_DAY = 7
HALT_HOURS = (11, 12)


# ---------------------------------------------------------------------------
# Calendar and market path
# ---------------------------------------------------------------------------


def calendar_regime(day: int, hour: int) -> int:
    """Calendar regime for a given day/hour inside a 14-day window.

    The weekend freeze runs from Friday 16:00 to Monday 09:00 (65 hours in a
    normal week), matching the 24/5 oracle schedule: weekday overnight hours
    are covered by the stream, weekends are not.
    """
    if day == HALT_DAY and hour in HALT_HOURS:
        return REGIME_HALT
    if day == HOLIDAY_DAY:
        return REGIME_HOLIDAY
    dow = day % 7
    if dow in (5, 6):
        return REGIME_WEEKEND
    if dow == 4 and hour >= 16:
        return REGIME_WEEKEND
    if dow == 0 and hour < 9:
        return REGIME_WEEKEND
    if 9 <= hour < 16:
        return REGIME_OPEN
    return REGIME_EXTENDED


@dataclass
class MarketPath:
    """One latent price path shared by all strategies in a (window, seed)."""

    true_price: np.ndarray   # S, length n_steps + 1
    pool_price: np.ndarray   # P, length n_steps + 1
    regime: np.ndarray       # length n_steps
    volume: np.ndarray       # length n_steps
    sigma: float


def simulate_market(seed_seq, n_steps: int = 336) -> MarketPath:
    """Simulate latent price and the regime-dependent observation mapping."""
    rng = np.random.default_rng(seed_seq)
    sigma = SIGMA_BASE * math.exp(rng.normal(0.0, 0.35))

    true_price = np.empty(n_steps + 1, dtype=float)
    true_price[0] = math.exp(MU_LOG)
    for t in range(n_steps):
        log_s = math.log(true_price[t])
        eps = rng.normal()
        jump = rng.normal(0.0, 3.0 * sigma) if rng.random() < 0.005 else 0.0
        true_price[t + 1] = math.exp(
            log_s + KAPPA * (MU_LOG - log_s) + sigma * eps + jump
        )

    regime = np.array(
        [calendar_regime((t // 24) % WINDOW_DAYS, t % 24) for t in range(n_steps)],
        dtype=int,
    )
    volume = rng.lognormal(math.log(BASE_VOLUME), 0.5, size=n_steps)
    volume[regime == REGIME_HALT] = 0.0

    # Observation mapping.  eta is the AR(1) log-deviation of the pool price
    # from its anchor; it is carried across steps so that extended/freeze
    # periods have serially correlated observation noise.
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
# LP position
# ---------------------------------------------------------------------------


def _lp_value_per_liquidity(price: float, lower: float, upper: float) -> float:
    """Uniswap-v3 position value per unit of liquidity at `price`.

    Uses sqrt-price tick math; the linear (non-sqrt) approximation grossly
    overstates losses when the price leaves a narrow range.
    """
    sqrt_p = math.sqrt(price)
    sqrt_l = math.sqrt(lower)
    sqrt_u = math.sqrt(upper)
    if price <= lower:
        return (1.0 / sqrt_l - 1.0 / sqrt_u) * price
    if price >= upper:
        return sqrt_u - sqrt_l
    return (1.0 / sqrt_p - 1.0 / sqrt_u) * price + (sqrt_p - sqrt_l)


@dataclass
class Position:
    cash: float = V0
    in_pos: bool = False
    liquidity: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    anchor: float = 0.0
    amount0: float = 0.0
    amount1: float = 0.0

    def value(self, price: float) -> float:
        if not self.in_pos:
            return 0.0
        return self.liquidity * _lp_value_per_liquidity(price, self.lower, self.upper)

    def hodl_value(self, price: float) -> float:
        if not self.in_pos:
            return 0.0
        return self.amount0 * price + self.amount1

    def il(self, price: float) -> float:
        return self.value(price) - self.hodl_value(price)

    def open(self, price: float, width: float) -> float:
        """Open a position with all available cash; return the cost paid."""
        if self.in_pos or self.cash <= GAS_USD:
            return 0.0
        notional = (self.cash - GAS_USD) / (1.0 + SLIPPAGE)
        if notional <= 0.0:
            return 0.0
        cost = GAS_USD + SLIPPAGE * notional
        lower, upper = price * (1.0 - width), price * (1.0 + width)
        liquidity = notional / _lp_value_per_liquidity(price, lower, upper)
        self.liquidity = liquidity
        self.lower = lower
        self.upper = upper
        self.anchor = price
        sqrt_p = math.sqrt(price)
        self.amount0 = liquidity * (1.0 / sqrt_p - 1.0 / math.sqrt(upper))
        self.amount1 = liquidity * (sqrt_p - math.sqrt(lower))
        self.cash = 0.0
        self.in_pos = True
        return cost

    def close(self, price: float) -> float:
        """Close the position to cash; return the cost paid."""
        if not self.in_pos:
            return 0.0
        value = self.value(price)
        cost = GAS_USD + SLIPPAGE * value
        self.cash += value - cost
        self.in_pos = False
        self.liquidity = 0.0
        self.lower = self.upper = self.anchor = 0.0
        self.amount0 = self.amount1 = 0.0
        return cost


# ---------------------------------------------------------------------------
# Logging policies
# ---------------------------------------------------------------------------


def choose_action(strategy: str, t: int, regime: int, price: float,
                  pos: Position, rng: np.random.Generator,
                  state: dict) -> int:
    """Return an action id.

    0 hold | 1 rebalance narrow | 2 rebalance wide | 3 exit
    4 enter narrow | 5 enter wide
    """
    if strategy == "Passive_Wide":
        return 5 if not pos.in_pos else 0
    if strategy == "Passive_Narrow":
        return 4 if not pos.in_pos else 0
    if strategy == "Periodic_24h":
        if t % 24 == 0:
            return 2 if pos.in_pos else 5
        return 0
    if strategy == "Threshold_2pct":
        if not pos.in_pos:
            return 4
        if abs(price / pos.anchor - 1.0) > 0.02:
            return 1
        return 0
    if strategy == "Exit_Freeze_50":
        if regime in (REGIME_WEEKEND, REGIME_HOLIDAY, REGIME_HALT):
            if pos.in_pos and not state["exited"] and rng.random() < 0.5:
                state["exited"] = True
                return 3
            return 0
        state["exited"] = False
        return 5 if not pos.in_pos else 0
    if strategy == "Stochastic_5pct":
        if not pos.in_pos:
            return 4 if rng.random() < 0.2 else 0
        if rng.random() < 0.05:
            return int(rng.choice([0, 1, 2, 3]))
        return 0
    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Strategy rollout
# ---------------------------------------------------------------------------


def simulate_strategy(market: MarketPath, strategy: str, seed_seq,
                      window: str, seed: int, n_steps: int = 336) -> list[dict]:
    """Run one logging policy on a shared market path and return log rows."""
    rng = np.random.default_rng(seed_seq)
    pos = Position()
    fee_cash = 0.0
    prev_il = 0.0          # pool-marked IL (observable feature)
    prev_il_true = 0.0     # fundamental IL vs HODL (hidden, drives reward)
    state = {"exited": False}
    rows: list[dict] = []

    for t in range(n_steps):
        price_from = float(market.pool_price[t])
        price_to = float(market.pool_price[t + 1])
        true_to = float(market.true_price[t + 1])
        regime = int(market.regime[t])

        # The agent observes the pool-marked value; the economic reward is the
        # LP excess return over HODL: fees + change in fundamental IL - costs.
        # Fundamental IL is marked at the latent price S, so in freeze it keeps
        # moving while the observed pool mark is stale.  This is the
        # observation-mapping asymmetry P1 studies.
        action = choose_action(strategy, t, regime, price_from, pos, rng, state)

        cost = 0.0
        changed = False
        if action == 0:
            pass
        elif action == 1:  # rebalance narrow
            if pos.in_pos:
                cost += pos.close(price_from)
            cost += pos.open(price_from, NARROW)
            changed = True
        elif action == 2:  # rebalance wide
            if pos.in_pos:
                cost += pos.close(price_from)
            cost += pos.open(price_from, WIDE)
            changed = True
        elif action == 3:  # exit
            if pos.in_pos:
                cost += pos.close(price_from)
                changed = True
        elif action == 4:  # enter narrow
            if not pos.in_pos:
                cost += pos.open(price_from, NARROW)
                changed = True
        elif action == 5:  # enter wide
            if not pos.in_pos:
                cost += pos.open(price_from, WIDE)
                changed = True
        else:
            raise ValueError(f"unknown action id: {action}")

        if changed:
            prev_il = 0.0
            prev_il_true = 0.0

        in_range = bool(pos.in_pos and pos.lower <= price_from <= pos.upper)
        fee_income = (
            FEE_RATE * float(market.volume[t]) * LIQUIDITY_SHARE * float(in_range)
        )
        fee_cash += fee_income

        value_end_obs = pos.cash + fee_cash + pos.value(price_to)
        il_now = pos.il(price_to) if pos.in_pos else 0.0
        il_step = il_now - prev_il
        prev_il = il_now

        if pos.in_pos:
            il_true_now = pos.il(true_to)
            il_true_step = il_true_now - prev_il_true
            prev_il_true = il_true_now
        else:
            il_true_step = 0.0
            prev_il_true = 0.0

        reward = (fee_income + il_true_step - cost) / V0

        rows.append(
            {
                "strategy": strategy,
                "window": window,
                "seed": int(seed),
                "episode": 0,
                "episode_step": t + 1,
                "current_regime": regime,
                "regime": regime,
                "action_id": float(action),
                "in_position": int(pos.in_pos),
                "portfolio_value": value_end_obs,
                "cash_usd": pos.cash,
                "position_value": pos.value(price_to) if pos.in_pos else 0.0,
                "fee_income_step_usd": fee_income,
                "impermanent_loss_usd": il_step,
                "operation_cost_usd": cost,
                "eth_price_from": price_from,
                "eth_price_to": price_to,
                "usdc_price_from": 1.0,
                "usdc_price_to": 1.0,
                "reward_env": reward,
                "anchor_price": pos.anchor if pos.in_pos else 0.0,
                "lower_price": pos.lower if pos.in_pos else 0.0,
                "upper_price": pos.upper if pos.in_pos else 0.0,
            }
        )

    return rows


def true_path_rows(market: MarketPath, window: str, seed: int) -> list[dict]:
    """Diagnostics only: latent price and the observation mapping."""
    rows = []
    for t in range(len(market.regime)):
        rows.append(
            {
                "window": window,
                "seed": int(seed),
                "episode_step": t + 1,
                "regime": int(market.regime[t]),
                "true_price": float(market.true_price[t]),
                "pool_price": float(market.pool_price[t]),
                "sigma": float(market.sigma),
            }
        )
    return rows
