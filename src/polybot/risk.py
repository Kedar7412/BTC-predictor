"""Risk gates and fractional Kelly sizing.

Kelly for a binary share bought at price c with win prob p:
    f* = (p - c) / (1 - c)
stake = kelly_fraction * f* * bankroll, capped by max_stake, bankroll and the
USD liquidity resting at the ask.
"""
from __future__ import annotations

from .config import Settings
from .models import Book, Decision, Opportunity


def kelly_fraction_full(p: float, price: float) -> float:
    if price >= 1.0:
        return 0.0
    return max(0.0, (p - price) / (1.0 - price))


def size_stake(opp: Opportunity, bankroll: float, ask_size: float, s: Settings) -> float:
    f = kelly_fraction_full(opp.prob, opp.eff_price) * s.kelly_fraction
    stake = f * bankroll
    stake = min(stake, s.max_stake_per_trade, bankroll, ask_size * opp.eff_price)
    return max(0.0, stake)


class RiskManager:
    def __init__(self, s: Settings):
        self.s = s
        self.daily_pnl = 0.0
        self.last_trade_ts = -1e18
        self.traded_windows: set[int] = set()

    def record_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl

    def reset_day(self) -> None:
        self.daily_pnl = 0.0

    def check(self, now: float, window_start: int, window_end: int, book: Book,
              open_positions: int) -> str | None:
        """Return a skip reason or None when pre-trade gates pass."""
        s = self.s
        if self.daily_pnl <= -s.max_daily_loss:
            return f"kill switch: daily pnl {self.daily_pnl:.2f}"
        if window_start in self.traded_windows:
            return "already traded this window"
        if open_positions >= s.max_open_positions:
            return "max open positions"
        if now - window_start < s.no_trade_first_seconds:
            return "no-trade window (market open)"
        if window_end - now < s.no_trade_last_seconds:
            return "no-trade window (near close)"
        if now - self.last_trade_ts < s.cooldown_seconds:
            return "cooldown"
        if book.spread > s.max_spread:
            return f"spread {book.spread:.3f} > {s.max_spread}"
        if book.best_ask <= 0 or book.best_ask >= 1:
            return "invalid ask"
        return None

    def decide(self, now: float, window_start: int, window_end: int, book: Book,
               opp: Opportunity, bankroll: float, open_positions: int) -> Decision:
        reason = self.check(now, window_start, window_end, book, open_positions)
        if reason:
            return Decision(False, reason, opp)
        if book.ask_size * opp.eff_price < self.s.min_liquidity_usd:
            return Decision(False, "insufficient liquidity", opp)
        stake = size_stake(opp, bankroll, book.ask_size, self.s)
        if stake < 1.0:
            return Decision(False, f"stake too small {stake:.2f}", opp)
        shares = stake / opp.eff_price
        return Decision(True, "ok", opp, shares=shares, stake=stake)

    def mark_traded(self, now: float, window_start: int) -> None:
        self.last_trade_ts = now
        self.traded_windows.add(window_start)
