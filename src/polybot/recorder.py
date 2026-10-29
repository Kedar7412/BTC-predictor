"""Book recorder: collect REAL Polymarket top-of-book snapshots for later backtesting.

Run it for weeks in paper/observe mode: `python -m polybot.record --out books.jsonl`.
It only READS (Gamma + CLOB book endpoints plus a BTC price). It never places orders and needs no keys.
One JSON object per line:
  {"ts", "window_start", "up_bid", "up_ask", "up_size", "down_bid", "down_ask", "down_size", "btc"}
Use the file with `python -m polybot.backtest --books books.jsonl`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Awaitable, Callable

from .feeds import Clock, SystemClock
from .market import MarketProvider, window_start

log = logging.getLogger("polybot.recorder")


class Recorder:
    def __init__(self, provider: MarketProvider, price_fn: Callable[[], Awaitable[float]] | None,
                 path: str | Path, clock: Clock | None = None, tick_seconds: float = 5.0, window: int = 300):
        self.provider, self.price_fn = provider, price_fn
        self.path = Path(path)
        self.clock = clock or SystemClock()
        self.tick, self.window = tick_seconds, window
        self.written = 0

    async def snapshot(self, market, ws: int) -> dict | None:
        """Fetch one snapshot for `market` and append it. Returns the row (None if no book)."""
        books = await self.provider.get_books(market)
        if not books:
            return None
        up, down = books
        btc = None
        if self.price_fn is not None:
            try:
                btc = await self.price_fn()
            except Exception as e:  # price is informational, never fatal
                log.warning("price fetch failed: %s", e)
        row = {"ts": round(self.clock.now(), 3), "window_start": int(ws),
               "up_bid": up.best_bid, "up_ask": up.best_ask, "up_size": up.ask_size,
               "down_bid": down.best_bid, "down_ask": down.best_ask, "down_size": down.ask_size, "btc": btc}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.written += 1
        return row

    async def record_window(self, ws: int) -> int:
        """Poll one window from now until it ends. Returns snapshots written."""
        market = None
        for _ in range(5):
            try:
                market = await self.provider.get_market(ws)
            except Exception as e:
                log.warning("market lookup failed: %s", e)
            if market:
                break
            await self.clock.sleep(2.0)
        if not market:
            return 0
        n = 0
        while self.clock.now() < ws + self.window:
            try:
                if await self.snapshot(market, ws):
                    n += 1
            except Exception as e:
                log.warning("book fetch failed: %s", e)
            await self.clock.sleep(self.tick)
        return n

    async def run(self, max_windows: int | None = None) -> int:
        done = 0
        while max_windows is None or done < max_windows:
            ws = window_start(self.clock.now(), self.window)
            await self.record_window(ws)
            done += 1
            # wait for the next window to start
            gap = ws + self.window - self.clock.now()
            if gap > 0:
                await self.clock.sleep(gap)
        return self.written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m polybot.record",
                                description="Record Polymarket BTC 5m books to JSONL (read only, no orders)")
    p.add_argument("--out", default="books.jsonl")
    p.add_argument("--tick-seconds", type=float, default=5.0)
    p.add_argument("--max-windows", type=int)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    from .market import PolymarketProvider
    from .feeds import LiveFeed
    feed = LiveFeed()
    rec = Recorder(PolymarketProvider(), feed.price, a.out, tick_seconds=a.tick_seconds)
    try:
        asyncio.run(rec.run(a.max_windows))
    except KeyboardInterrupt:
        pass
    print(f"wrote {rec.written} snapshots to {a.out}", file=sys.stderr)
    return 0
