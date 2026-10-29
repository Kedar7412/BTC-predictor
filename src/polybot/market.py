from __future__ import annotations

import json
import math
import random
from typing import Protocol

import httpx

from .feeds import Clock, PriceFeed
from .models import Book, MarketInfo
from .signal import norm_cdf

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def window_start(ts: float, window: int = 300) -> int:
    return int(ts // window * window)


class MarketProvider(Protocol):
    async def get_market(self, start_ts: int) -> MarketInfo | None: ...
    async def get_books(self, market: MarketInfo) -> tuple[Book, Book] | None: ...


class PolymarketProvider:
    """Discovers `btc-updown-5m-<start_ts>` via Gamma and reads CLOB books."""

    def __init__(self, client: httpx.AsyncClient | None = None, window: int = 300):
        self.http = client or httpx.AsyncClient(timeout=5.0)
        self.window = window

    async def get_market(self, start_ts: int) -> MarketInfo | None:
        slug = f"btc-updown-5m-{start_ts}"
        r = await self.http.get(f"{GAMMA}/markets", params={"slug": slug})
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        m = rows[0]
        tokens = m.get("clobTokenIds")
        outcomes = m.get("outcomes")
        tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        if not tokens or len(tokens) != 2:
            return None
        names = [str(o).lower() for o in (outcomes or ["up", "down"])]
        up_i = names.index("up") if "up" in names else 0
        return MarketInfo(slug, start_ts, start_ts + self.window, tokens[up_i], tokens[1 - up_i])

    async def _book(self, token: str) -> Book | None:
        r = await self.http.get(f"{CLOB}/book", params={"token_id": token})
        r.raise_for_status()
        j = r.json()
        asks = sorted(((float(a["price"]), float(a["size"])) for a in j.get("asks", [])))
        bids = sorted(((float(b["price"]), float(b["size"])) for b in j.get("bids", [])), reverse=True)
        if not asks or not bids:
            return None
        return Book(bids[0][0], asks[0][0], asks[0][1])

    async def get_books(self, market: MarketInfo) -> tuple[Book, Book] | None:
        up = await self._book(market.up_token)
        down = await self._book(market.down_token)
        return (up, down) if up and down else None


class SimulatedMarket:
    """Offline market whose quotes are a noisy version of the true fair probability."""

    def __init__(self, feed: PriceFeed, clock: Clock, window: int = 300, noise: float = 0.08,
                 half_spread: float = 0.01, sigma: float = 0.00009, seed: int = 11):
        self.feed, self.clock, self.window = feed, clock, window
        self.noise, self.half_spread, self.sigma = noise, half_spread, sigma
        self.rng = random.Random(seed)
        self.opens: dict[int, float] = {}

    async def get_market(self, start_ts: int) -> MarketInfo | None:
        if start_ts not in self.opens:
            px = await self.feed.price_at(start_ts)
            self.opens[start_ts] = px if px is not None else await self.feed.price()
        return MarketInfo(f"sim-{start_ts}", start_ts, start_ts + self.window, f"up-{start_ts}", f"dn-{start_ts}")

    async def get_books(self, market: MarketInfo) -> tuple[Book, Book] | None:
        px = await self.feed.price()
        left = max(market.end_ts - self.clock.now(), 1.0)
        z = math.log(px / self.opens[market.start_ts]) / (self.sigma * math.sqrt(left))
        p = min(0.97, max(0.03, norm_cdf(z) + self.rng.gauss(0, self.noise)))
        h = self.half_spread
        up = Book(max(0.01, p - h), min(0.99, p + h), 500.0)
        q = 1 - p
        down = Book(max(0.01, q - h), min(0.99, q + h), 500.0)
        return up, down
