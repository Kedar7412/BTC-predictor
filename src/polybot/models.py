from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Book:
    """Top of book for one outcome token."""
    best_bid: float
    best_ask: float
    ask_size: float = 0.0  # shares available at best ask

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


@dataclass
class MarketInfo:
    slug: str
    start_ts: int
    end_ts: int
    up_token: str
    down_token: str


@dataclass
class Signal:
    prob_up: float
    sigma: float          # per sqrt(second) volatility used
    z: float
    conviction_up: float  # P(up)
    conviction_down: float


@dataclass
class Opportunity:
    side: str             # "UP" or "DOWN"
    prob: float
    ask: float
    eff_price: float      # ask + slippage + fee per share
    edge: float
    ev: float
    roi: float


@dataclass
class Decision:
    trade: bool
    reason: str
    opp: Opportunity | None = None
    shares: float = 0.0
    stake: float = 0.0


@dataclass
class Position:
    window_start: int
    window_end: int
    side: str
    token: str
    shares: float
    price: float
    cost: float
    prob: float
    edge: float
    live: bool = False
    meta: dict = field(default_factory=dict)
