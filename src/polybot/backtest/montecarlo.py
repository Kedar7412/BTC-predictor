"""Bootstrap / shuffle of the trade PnL sequence (dollar PnL, resampled; sizing is not re-simulated).

This answers "how much of the equity curve is path luck given THESE trades", not "will the edge
persist". It cannot create information the trade list does not contain.
"""
from __future__ import annotations

import numpy as np

DAY = 86400
PCTS = (5, 50, 95)


def _paths(pnl: np.ndarray, n_paths: int, rng: np.random.Generator, mode: str, block: int):
    n = len(pnl)
    if mode == "shuffle":
        return np.stack([pnl[rng.permutation(n)] for _ in range(n_paths)])
    nb = -(-n // block)
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n]
    return pnl[np.minimum(idx, n - 1)]


def _summ(x: np.ndarray) -> dict:
    p = np.percentile(x, PCTS)
    return {"p5": float(p[0]), "p50": float(p[1]), "p95": float(p[2]), "mean": float(x.mean())}


def run_montecarlo(pnl: np.ndarray, day: np.ndarray, bankroll0: float, n_paths: int = 2000,
                   seed: int = 0, block: int = 20, max_daily_loss: float = 100.0,
                   batch: int = 50) -> dict:
    """pnl: dollar PnL per trade in order; day: epoch day of each trade (for the daily loss check)."""
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    if n < 2:
        return {"n_paths": 0, "note": "fewer than 2 trades, nothing to resample"}
    rng = np.random.default_rng(seed)
    out: dict = {"n_paths": n_paths, "seed": seed, "trades": n, "block": block,
                 "note": "Resamples the dollar PnL of the observed trades. Not a forecast."}
    for mode in ("block_bootstrap", "shuffle"):
        fin, mdd, ruin, dd30 = [], [], [], []
        done = 0
        while done < n_paths:
            m = min(batch, n_paths - done)
            eq = bankroll0 + np.cumsum(_paths(pnl, m, rng, mode, block), axis=1)
            peak = np.maximum.accumulate(np.concatenate([np.full((m, 1), bankroll0), eq], axis=1), axis=1)[:, 1:]
            dd = 1.0 - eq / np.maximum(peak, 1e-12)
            fin.append(eq[:, -1] / bankroll0 - 1.0)
            mdd.append(dd.max(axis=1))
            ruin.append((eq.min(axis=1) <= 0.5 * bankroll0))
            dd30.append(dd.max(axis=1) > 0.30)
            done += m
        fin, mdd = np.concatenate(fin), np.concatenate(mdd)
        out[mode] = {"final_return": _summ(fin), "max_drawdown": _summ(mdd),
                     "prob_loss": float((fin < 0).mean()),
                     "prob_lose_half": float(np.concatenate(ruin).mean()),
                     "prob_drawdown_gt_30pct": float(np.concatenate(dd30).mean())}
    # daily loss kill: resample whole days (blocks of 5 days) of the realised daily PnL
    d = np.asarray(day)
    uniq, inv = np.unique(d, return_inverse=True)
    daily = np.bincount(inv, weights=pnl)
    nd = len(daily)
    if nd >= 2:
        b = max(1, min(5, nd))
        sims = _paths(daily, n_paths, rng, "block_bootstrap", b)
        out["daily_loss_kill"] = {
            "days": int(nd), "limit": max_daily_loss,
            "observed_days_at_or_below_limit": int((daily <= -max_daily_loss).sum()),
            "prob_any_day_hits_limit": float((sims.min(axis=1) <= -max_daily_loss).mean()),
            "worst_day_pnl": _summ(sims.min(axis=1))}
    return out
