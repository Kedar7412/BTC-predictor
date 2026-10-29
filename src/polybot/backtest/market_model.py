"""ASSUMED market books for the backtest.  READ THIS BEFORE TRUSTING ANY RESULT.

!! Polymarket 5 minute BTC markets have no 5 year history, so the UP/DOWN bid and ask prices !!
!! used here are a MODEL, not recorded data.  A backtest on modelled books can NOT prove    !!
!! that the strategy has an edge.  It only shows how the strategy behaves under stated     !!
!! assumptions about how mispriced the market is.                                          !!

Scenarios (all produce a "market mid" for UP; DOWN is the complement):

  efficient    mid = bot's own probability.  OPTIMISTIC upper bound on how well the bot agrees
               with the market (zero edge except spread); not realistic evidence.
  noisy        mid = bot probability + gaussian noise in logit space (noise_sd).  The market is
               centred on the bot's own model, which is OPTIMISTIC: random mispricing is exactly
               what the bot is built to harvest, so this is an upper bound, not a prediction.
  lagged       mid = Phi(z) computed from the BTC price `lag_minutes` ago (real information the
               bot also has, just stale).  Not centred on the bot's model.  More honest, but the
               staleness size is a guess.
  stale_noisy  lagged plus logit noise.  The most conservative built-in scenario.

Common: half_spread around the mid, mid clipped to [0.02, 0.98], ask_size_usd of liquidity at the
ask, adverse_selection added to the ask (models latency: you fill after the price moved).
Fees and slippage are NOT applied here; edge.evaluate_side applies Settings.fee_rate and
Settings.slippage exactly as the live bot does.  Noise is a pure function of (seed, window, tick),
so results do not depend on which ticks the simulator skips.

Real recorded books (FEAT-002) can replace this by implementing `BookSource`.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .. import signal as signal_mod
from ..models import Book

SCENARIOS = ("efficient", "noisy", "lagged", "stale_noisy")
MID_MIN, MID_MAX = 0.02, 0.98

ASSUMPTIONS_BANNER = (
    "=" * 78 + "\n"
    "ASSUMPTIONS: Polymarket bid/ask prices are NOT historical data. They come from the\n"
    "modelled scenario below. Results show behaviour under that assumption, they do NOT\n"
    "prove a real edge. 'efficient' and 'noisy' are centred on the bot's own probability\n"
    "and are optimistic upper bounds.\n" + "=" * 78
)


@dataclass
class TickContext:
    row: int            # row index in the SignalSet (stable, used for noise lookup)
    ws: int             # window start (epoch seconds)
    tick: int           # 0..3 (elapsed 60,120,180,240 s)
    now: int            # decision time
    end_ts: int
    open_px: float      # window open price K
    price: float        # last 1m close (known at `now`)
    lag_price: float    # close `lag_minutes` earlier (nan if unavailable or lag is 0)
    prob_up: float      # the bot's own estimate
    sigma: float
    seconds_left: float


class BookSource(ABC):
    """Anything that can supply (UP book, DOWN book) at a decision tick."""
    lag_minutes: int = 0

    def prepare(self, ws: np.ndarray) -> None:
        """Called once with all window starts (rows) before the simulation."""

    @abstractmethod
    def books(self, ctx: TickContext) -> tuple[Book, Book] | None:
        """Return (up, down) or None when no book is available."""

    def outcome(self, ctx: TickContext, actual_up_won: bool) -> bool:
        """Settlement outcome. Real data uses the actual result; only the no-information control overrides."""
        return actual_up_won

    def describe(self) -> dict:
        return {"source": type(self).__name__}


_M = (1 << 64) - 1


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hashed_normals(seed: int, ws: np.ndarray, ticks: int = 4) -> np.ndarray:
    """N(0,1) draws of shape (len(ws), ticks), a pure function of (seed, ws, tick)."""
    key = (ws.astype(np.uint64)[:, None] * np.uint64(ticks) + np.arange(ticks, dtype=np.uint64)[None, :])
    with np.errstate(over="ignore"):
        key = key ^ (np.uint64(seed & _M) * np.uint64(0xD6E8FEB86659FD93))
    h1 = _splitmix(key)
    h2 = _splitmix(h1 ^ np.uint64(0xA5A5A5A55A5A5A5A))
    u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53
    u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


class AssumedMarket(BookSource):
    def __init__(self, scenario: str = "stale_noisy", noise_sd: float = 0.15, lag_minutes: int = 2,
                 half_spread: float = 0.01, ask_size_usd: float = 200.0,
                 adverse_selection: float = 0.0, seed: int = 0):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if lag_minutes < 0 or half_spread < 0 or noise_sd < 0 or ask_size_usd <= 0:
            raise ValueError("lag_minutes, half_spread, noise_sd must be >= 0 and ask_size_usd > 0")
        self.scenario, self.noise_sd = scenario, noise_sd
        self.half_spread, self.ask_size_usd = half_spread, ask_size_usd
        self.adverse_selection, self.seed = adverse_selection, seed
        self.lag_minutes = lag_minutes if scenario in ("lagged", "stale_noisy") else 0
        self._noise: np.ndarray | None = None

    def prepare(self, ws: np.ndarray) -> None:
        self._noise = hashed_normals(self.seed, np.asarray(ws, dtype=np.int64))

    def describe(self) -> dict:
        return {"source": "AssumedMarket (MODELLED, not recorded)", "scenario": self.scenario,
                "noise_sd": self.noise_sd, "lag_minutes": self.lag_minutes,
                "half_spread": self.half_spread, "ask_size_usd": self.ask_size_usd,
                "adverse_selection": self.adverse_selection, "seed": self.seed}

    def mid_up(self, ctx: TickContext) -> float | None:
        sc = self.scenario
        if sc in ("efficient", "noisy"):
            base = ctx.prob_up
        else:
            if not (ctx.lag_price > 0):
                return None
            z = math.log(ctx.lag_price / ctx.open_px) / (ctx.sigma * math.sqrt(max(ctx.seconds_left, 1.0)))
            base = signal_mod.norm_cdf(z)
        if sc in ("noisy", "stale_noisy") and self.noise_sd > 0:
            if self._noise is None:
                raise RuntimeError("AssumedMarket.prepare() was not called")
            b = min(1 - 1e-4, max(1e-4, base))
            base = 1.0 / (1.0 + math.exp(-(math.log(b / (1 - b)) + self.noise_sd * self._noise[ctx.row, ctx.tick])))
        return min(MID_MAX, max(MID_MIN, base))

    def _book(self, mid: float) -> Book:
        bid = max(0.01, mid - self.half_spread)
        ask = min(0.99, mid + self.half_spread + self.adverse_selection)
        return Book(bid, ask, self.ask_size_usd / ask)

    def books(self, ctx: TickContext) -> tuple[Book, Book] | None:
        mid = self.mid_up(ctx)
        if mid is None:
            return None
        return self._book(mid), self._book(1.0 - mid)


def hashed_uniform(seed: int, ws: np.ndarray, ticks: int = 4) -> np.ndarray:
    """U(0,1) draws of shape (len(ws), ticks), a pure function of (seed, ws, tick)."""
    key = (ws.astype(np.uint64)[:, None] * np.uint64(ticks) + np.arange(ticks, dtype=np.uint64)[None, :])
    with np.errstate(over="ignore"):
        key = key ^ (np.uint64(seed & _M) * np.uint64(0x9E3779B97F4A7C15))
    h = _splitmix(_splitmix(key))
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53


class NoInfoControl(AssumedMarket):
    """CONTROL, not a scenario. The market mid is the TRUE probability: outcomes are drawn as
    Bernoulli(mid) instead of using the real result. The bot's signal then carries no information
    beyond what the price already says, so any trade it takes can only lose spread, fee and slippage.
    This validates COST ACCOUNTING only (spread, fee, slippage are charged and sizing cannot create
    profit). Real outcomes never enter, so it cannot detect look-ahead leaks; those are covered by
    the no-look-ahead tests."""

    def prepare(self, ws: np.ndarray) -> None:
        super().prepare(ws)
        self._u = hashed_uniform(self.seed + 104729, np.asarray(ws, dtype=np.int64))

    def outcome(self, ctx: TickContext, actual_up_won: bool) -> bool:
        mid = self.mid_up(ctx)
        return bool(self._u[ctx.row, ctx.tick] < (mid if mid is not None else 0.5))

    def describe(self) -> dict:
        return {**super().describe(), "control": "outcomes drawn from the market mid (no information)"}
