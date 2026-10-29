from __future__ import annotations

import bisect
import math
import random
import time
from typing import Protocol

import httpx


class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, s: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    async def sleep(self, s: float) -> None:
        import asyncio
        await asyncio.sleep(s)


class SimClock:
    """Virtual clock; sleep advances time instantly."""

    def __init__(self, start: float):
        self.t = float(start)

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += s


class PriceFeed(Protocol):
    async def price(self) -> float: ...
    async def price_at(self, ts: float) -> float | None: ...


class SimulatedFeed:
    """Seeded geometric random walk driven by a clock. Offline."""

    def __init__(self, clock: Clock, start_price: float = 65000.0, sigma: float = 0.00009, seed: int = 7):
        self.clock = clock
        self.sigma = sigma
        self.rng = random.Random(seed)
        self._p = start_price
        self._last = clock.now()
        self._ts: list[float] = [self._last]
        self._px: list[float] = [start_price]

    async def price(self) -> float:
        now = self.clock.now()
        dt = now - self._last
        if dt > 0:
            self._p *= math.exp(self.sigma * math.sqrt(dt) * self.rng.gauss(0, 1))
            self._last = now
            self._ts.append(now)
            self._px.append(self._p)
        return self._p

    async def price_at(self, ts: float) -> float | None:
        i = bisect.bisect_right(self._ts, ts) - 1
        return self._px[i] if i >= 0 else None


class LiveFeed:
    """Binance spot REST with Coinbase fallback."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.http = client or httpx.AsyncClient(timeout=5.0)

    async def price(self) -> float:
        try:
            r = await self.http.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            r = await self.http.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
            r.raise_for_status()
            return float(r.json()["data"]["amount"])

    async def price_at(self, ts: float) -> float | None:
        try:
            r = await self.http.get("https://api.binance.com/api/v3/klines",
                                    params={"symbol": "BTCUSDT", "interval": "1m",
                                            "startTime": int(ts // 60 * 60) * 1000, "limit": 1})
            r.raise_for_status()
            rows = r.json()
            return float(rows[0][1]) if rows else None
        except Exception:
            return None
