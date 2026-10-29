"""Edge and expected value.

Buying one share at price c pays 1 if the side wins, else 0.
    eff_price = ask + slippage + fee_per_share,  fee_per_share = fee_rate * ask * (1 - ask)
    edge = p - eff_price
    EV/share = p * (1 - c) - (1 - p) * c - fee,  c = ask + slippage
    ROI = EV / (c + fee)
With mid given, EV uses p_ev = (1-w)*p + w*mid (w = market_shrink), so EV < edge when the model
is more bullish than the market; edge uses the raw model probability.
A trade is only proposed when edge, EV, ROI and conviction all pass.
"""
from __future__ import annotations

from .config import Settings
from .models import Book, Opportunity, Signal


def fee_per_share(ask: float, fee_rate: float) -> float:
    return fee_rate * ask * (1 - ask)


def evaluate_side(side: str, p: float, ask: float, s: Settings, mid: float | None = None) -> Opportunity:
    fee = fee_per_share(ask, s.fee_rate)
    c = min(ask + s.slippage, 0.999)
    eff = c + fee
    # EV uses a probability shrunk toward the market mid, so it is a more conservative
    # estimate than the raw-model edge and the min_ev gate is not redundant.
    p_ev = p if mid is None else (1 - s.market_shrink) * p + s.market_shrink * mid
    ev = p_ev * (1 - c) - (1 - p_ev) * c - fee
    return Opportunity(side=side, prob=p, ask=ask, eff_price=eff, edge=p - eff,
                       ev=ev, roi=ev / eff if eff > 0 else 0.0)


def best_opportunity(sig: Signal, up: Book, down: Book, s: Settings) -> tuple[Opportunity | None, str]:
    """Return (opportunity, reason). Opportunity is None when nothing qualifies."""
    cands = [evaluate_side("UP", sig.conviction_up, up.best_ask, s, (up.best_bid + up.best_ask) / 2),
             evaluate_side("DOWN", sig.conviction_down, down.best_ask, s,
                       (down.best_bid + down.best_ask) / 2)]
    best = max(cands, key=lambda o: o.edge)
    if best.prob < s.min_conviction:
        return None, f"low conviction {best.prob:.3f} < {s.min_conviction}"
    if best.edge < s.min_edge:
        return None, f"edge {best.edge:.3f} < {s.min_edge} ({best.side})"
    if best.ev < s.min_ev:
        return None, f"ev {best.ev:.3f} < {s.min_ev} ({best.side})"
    if best.roi < s.min_roi:
        return None, f"roi {best.roi:.3f} < {s.min_roi} ({best.side})"
    return best, "ok"
