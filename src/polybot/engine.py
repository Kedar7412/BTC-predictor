from __future__ import annotations

import logging
from dataclasses import fields
import time
from collections import deque

from . import edge as edge_mod
from . import signal as signal_mod
from .config import Settings
from .execution import Executor
from .feeds import Clock, PriceFeed
from .market import MarketProvider, window_start
from .models import Decision, MarketInfo, Position
from .monitor import Monitor
from .risk import RiskManager

log = logging.getLogger("polybot")


class Engine:
    """feed -> signal -> edge/EV -> risk/Kelly -> execution -> monitor."""

    def __init__(self, s: Settings, feed: PriceFeed, provider: MarketProvider,
                 executor: Executor, monitor: Monitor, clock: Clock):
        self.s, self.feed, self.provider = s, feed, provider
        self.executor, self.monitor, self.clock = executor, monitor, clock
        self.risk = RiskManager(s)
        self.history: deque[tuple[float, float]] = deque()
        self.open: list[Position] = []
        self.opens: dict[int, float] = {}
        self.market: MarketInfo | None = None
        self.current_window = -1
        self.day = ""

    def restore(self) -> None:
        st = self.monitor.load_state()
        if st and st.get("day") == time.strftime("%Y-%m-%d", time.gmtime(self.clock.now())):
            self.risk.daily_pnl = float(st.get("daily_pnl", 0.0))
            self.executor.bankroll = float(st.get("bankroll", self.executor.bankroll))
            known = {f.name for f in fields(Position)}
            for d in st.get("open_positions", []):
                try:
                    pos = Position(**{k: v for k, v in d.items() if k in known})
                except TypeError:
                    continue
                self.open.append(pos)
                self.risk.mark_traded(self.clock.now(), pos.window_start)

    async def _settle_due(self, now: float) -> None:
        for pos in list(self.open):
            if now < pos.window_end:
                continue
            end_px = await self.feed.price_at(pos.window_end)
            if end_px is None:
                end_px = await self.feed.price()
            open_px = self.opens.get(pos.window_start)
            if open_px is None:  # e.g. restored after restart
                open_px = await self.feed.price_at(pos.window_start)
            if open_px is None:
                log.warning("cannot settle window %d: no open price", pos.window_start)
                continue
            up_won = end_px > open_px
            pnl = self.executor.settle(pos, up_won)
            self.risk.record_pnl(pnl)
            self.open.remove(pos)
            self.monitor.settled(pos, pnl, up_won)

    async def tick(self) -> Decision | None:
        s = self.s
        now = self.clock.now()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self.day:
            if self.day:
                self.risk.reset_day()
            self.day = day
        px = await self.feed.price()
        self.history.append((now, px))
        while self.history and now - self.history[0][0] > s.vol_window_seconds:
            self.history.popleft()

        await self._settle_due(now)
        ws = window_start(now, s.window_seconds)
        if ws != self.current_window:
            self.current_window = ws
            self.market = None  # never carry the previous window's market over
        if self.market is None:
            # retried every tick until the market is found
            try:
                self.market = await self.provider.get_market(ws)
            except Exception as e:
                log.warning("market lookup failed for window %d: %r", ws, e)
                self.market = None
        if ws not in self.opens and self.market is not None:
            ref = await self.feed.price_at(ws)
            if ref is not None:
                self.opens[ws] = ref
                self.opens = {k: v for k, v in self.opens.items() if k >= ws - 3 * s.window_seconds}
        if self.market is None:
            log.info("SKIP no market for window %d", ws)
            return None
        if ws not in self.opens:
            log.info("SKIP no window open price for %d", ws)
            return None
        books = await self.provider.get_books(self.market)
        if books is None:
            log.info("SKIP no orderbook")
            return None
        up_book, down_book = books
        sig = signal_mod.estimate(px, self.opens[ws], self.market.end_ts - now, list(self.history),
                                  s.default_sigma_per_sqrt_s, s.min_vol_samples, s.momentum_weight)
        opp, reason = edge_mod.best_opportunity(sig, up_book, down_book, s)
        if opp is None:
            d = Decision(False, reason)
        else:
            book = up_book if opp.side == "UP" else down_book
            d = self.risk.decide(now, ws, self.market.end_ts, book, opp,
                                 self.executor.bankroll, len(self.open))
        if d.trade:
            # mark first: if the order errors or times out it may have filled, so never retry this window
            self.risk.mark_traded(now, ws)
            pos = await self.executor.buy(self.market, d, now)
            if pos is None:
                d = Decision(False, "order not filled", d.opp)
            else:
                self.open.append(pos)
                self.monitor.opened(pos)
        self.monitor.decision(now, ws, d, sig.prob_up)
        self.monitor.save_state(self.executor.bankroll, self.risk.daily_pnl, self.day, self.open)
        return d

    async def run(self, max_cycles: int | None = None) -> None:
        n = 0
        while max_cycles is None or n < max_cycles:
            try:
                await self.tick()
            except Exception as e:  # never crash the loop
                log.warning("tick error: %r", e)
            n += 1
            if n % 20 == 0:
                self.monitor.log_stats()
            await self.clock.sleep(self.s.tick_seconds)
        self.monitor.log_stats()
