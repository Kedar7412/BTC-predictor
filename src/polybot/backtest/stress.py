"""Stress tests: perturb the price data and the cost model, compare each with a baseline.

Data stresses rebuild signals, so they run on the last `stress_days` of data (all of it for the quick
suite) to keep the full suite within minutes. The baseline uses exactly the same period.
All stresses run against the reference assumed market, so the same caveats apply.
"""
from __future__ import annotations

import numpy as np

from ..config import Settings
from . import data as data_mod
from .data import Klines
from .jobs import pmap, run_job
from .scenarios import REFERENCE, implausible
from .sim import build_signals

DD_LIMIT = 0.30


def _rebuild(k: Klines, ts, close, wick_hi, wick_lo) -> Klines:
    opn = np.concatenate([[close[0]], close[:-1]])
    return data_mod.clean(ts, opn, np.maximum(opn, close) * wick_hi, np.minimum(opn, close) * wick_lo, close)


def vol_spikes(k: Klines, factor: float = 3.0, seed: int = 0, length: tuple[int, int] = (120, 360),
               every_days: float = 7.0) -> Klines:
    """Multiply 1m log returns by `factor` inside random episodes (about one per week)."""
    rng = np.random.default_rng(seed)
    n = len(k)
    lr = np.diff(np.log(k.close), prepend=np.log(k.close[0]))
    n_ep = max(1, int(n / 1440 / every_days))
    for _ in range(n_ep):
        a = int(rng.integers(0, max(n - 1, 1)))
        lr[a:a + int(rng.integers(*length))] *= factor
    close = k.close[0] * np.exp(np.cumsum(lr))
    hi = np.maximum(k.high / np.maximum(k.open, k.close), 1.0)
    lo = np.minimum(k.low / np.minimum(k.open, k.close), 1.0)
    return _rebuild(k, k.ts, close, hi, lo)


def data_gaps(k: Klines, frac: float = 0.02, seed: int = 0, length: tuple[int, int] = (3, 60)) -> Klines:
    """Drop random minute ranges until about `frac` of the minutes are missing."""
    rng = np.random.default_rng(seed)
    n = len(k)
    keep = np.ones(n, dtype=bool)
    target = int(frac * n)
    dropped = 0
    while dropped < target:
        a = int(rng.integers(0, n))
        ln = int(rng.integers(*length))
        dropped += int(keep[a:a + ln].sum())
        keep[a:a + ln] = False
    return data_mod.clean(k.ts[keep], k.open[keep], k.high[keep], k.low[keep], k.close[keep])


def flash_gaps(k: Klines, jump: float = 0.06, seed: int = 0, every_days: float = 30.0) -> Klines:
    """10 minutes of missing data, then a +-jump price shock that reverts after 30 minutes."""
    rng = np.random.default_rng(seed)
    n = len(k)
    keep = np.ones(n, dtype=bool)
    o, h, l, c = (x.copy() for x in (k.open, k.high, k.low, k.close))
    for _ in range(max(1, int(n / 1440 / every_days))):
        a = int(rng.integers(0, max(n - 60, 1)))
        keep[a:a + 10] = False
        f = 1.0 + jump * (1 if rng.random() < 0.5 else -1)
        for x in (o, h, l, c):
            x[a + 10:a + 40] *= f
    return data_mod.clean(k.ts[keep], o[keep], h[keep], l[keep], c[keep])


def cutoff(k: Klines, days: float | None) -> Klines:
    if not days:
        return k
    m = k.ts >= k.ts[-1] - int(days * 86400)
    return data_mod.clean(k.ts[m], k.open[m], k.high[m], k.low[m], k.close[m])


def _flags(base: dict, m: dict) -> list[str]:
    out = []
    if m["total_return"] < 0:
        out.append("NEGATIVE return" + (" (baseline was positive)" if base["total_return"] > 0 else ""))
    if m["max_drawdown"] > DD_LIMIT:
        out.append(f"max drawdown {100 * m['max_drawdown']:.0f}% > {100 * DD_LIMIT:.0f}%")
    w = implausible(m)
    if w:
        out.append("implausibly good, treat as artifact")
    return out


def _row(m: dict) -> dict:
    return {k: m.get(k) for k in ("total_return", "max_drawdown", "sharpe", "trades", "win_rate")}


def run_stress(k: Klines, s: Settings, workers: int = 1, stress_days: float | None = None, seed: int = 0,
               market: dict | None = None) -> dict:
    from .scenarios import market_kwargs
    mk = market or {**market_kwargs(REFERENCE), "half_spread": 0.01, "seed": seed}
    sub = cutoff(k, stress_days)
    base_sigs = build_signals(sub, s)
    results: dict = {}
    cost_jobs: dict = {"baseline": {"market": mk}}
    for f in (2, 3):
        cost_jobs[f"fees_and_slippage_x{f}"] = {"market": mk, "settings": {
            "fee_rate": s.fee_rate * f, "slippage": s.slippage * f}}
    for adv in (0.005, 0.01, 0.02, 0.03):
        cost_jobs[f"adverse_selection_ask+{int(adv * 1000) / 10:g}c"] = {"market": {**mk, "adverse_selection": adv}}
    names = list(cost_jobs)
    res = pmap([cost_jobs[n] for n in names], base_sigs, s, workers, "stress-costs")
    for n, r in zip(names, res):
        results[n] = r["metrics"]
    data_variants = {"vol_spike_x3": vol_spikes(sub, 3.0, seed), "data_gaps_2pct": data_gaps(sub, 0.02, seed),
                     "flash_gaps_6pct": flash_gaps(sub, 0.06, seed)}
    spiked = data_variants["vol_spike_x3"]
    combined_k = flash_gaps(data_gaps(spiked, 0.02, seed + 1), 0.06, seed + 1)
    for n, kk in data_variants.items():
        results[n] = run_job({"market": mk}, build_signals(kk, s), s)["metrics"]
    results["combined_worst_case"] = run_job(
        {"market": {**mk, "adverse_selection": 0.02}, "settings": {
            "fee_rate": s.fee_rate * 3, "slippage": s.slippage * 3}},
        build_signals(combined_k, s), s)["metrics"]
    base = results["baseline"]
    table = {n: {**_row(m), "flags": [] if n == "baseline" else _flags(base, m)} for n, m in results.items()}
    table["baseline"]["flags"] = _flags(base, base)
    return {"period_days": float((sub.ts[-1] - sub.ts[0]) / 86400), "market": mk, "results": table,
            "any_flag": any(f for n, v in table.items() if n != "baseline" for f in v["flags"]
                           if not f.startswith("implausibly")),
            "note": "Compared with the baseline on the same period and the same ASSUMED market."}
