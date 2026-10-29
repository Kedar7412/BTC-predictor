"""Probability model for P(BTC at window end > BTC at window start).

Math (driftless diffusion with momentum tilt):
    x = ln(S_now / K)                  K = window open price
    sigma = realized vol per sqrt(second) (blend with default when few samples)
    T = seconds remaining
    z = x / (sigma * sqrt(T)) + w * m
    P(up) = Phi(z)
where m is a bounded momentum term (recent return in units of sigma*sqrt(lookback))
and w = momentum_weight. P is clamped to [0.01, 0.99].
Less time left or a bigger move away from K gives a larger |z|, so higher
confidence. Conviction is the probability of the favored side.
"""
from __future__ import annotations

import math
from typing import Sequence

from .models import Signal

P_MIN, P_MAX = 0.01, 0.99


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def realized_sigma(history: Sequence[tuple[float, float]], default: float, min_samples: int) -> float:
    """Per sqrt(second) vol from (ts, price) samples; shrinks toward default."""
    rets = []
    for (t0, p0), (t1, p1) in zip(history, history[1:]):
        dt = t1 - t0
        if dt > 0 and p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0) / math.sqrt(dt))
    n = len(rets)
    if n < 2:
        return default
    est = math.sqrt(sum(r * r for r in rets) / n)
    w = min(1.0, n / max(min_samples, 1))
    return max(w * est + (1 - w) * default, default * 0.25)


def estimate(price: float, open_price: float, seconds_left: float,
             history: Sequence[tuple[float, float]], default_sigma: float,
             min_samples: int = 8, momentum_weight: float = 0.15,
             imbalance: float = 0.0) -> Signal:
    sigma = realized_sigma(history, default_sigma, min_samples)
    T = max(seconds_left, 1.0)
    x = math.log(price / open_price)
    z = x / (sigma * math.sqrt(T))
    m = 0.0
    if len(history) >= 2:
        t0, p0 = history[max(0, len(history) - 6)]
        t1, p1 = history[-1]
        dt = t1 - t0
        if dt > 0 and p0 > 0:
            m = math.log(p1 / p0) / (sigma * math.sqrt(dt))
            m = max(-3.0, min(3.0, m))
    z += momentum_weight * (m + max(-1.0, min(1.0, imbalance)))
    p = min(P_MAX, max(P_MIN, norm_cdf(z)))
    return Signal(prob_up=p, sigma=sigma, z=z, conviction_up=p, conviction_down=1 - p)
