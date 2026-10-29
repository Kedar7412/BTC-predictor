"""Performance and calibration metrics. All functions return plain dicts/lists (JSON friendly)."""
from __future__ import annotations

import math

import numpy as np

REGIME_NAMES = {0: "low_vol", 1: "mid_vol", 2: "high_vol"}
DAY = 86400


def drawdown(equity: np.ndarray, times: np.ndarray) -> dict:
    """Max drawdown depth (fraction) and longest time under water (days)."""
    if len(equity) == 0:
        return {"max_drawdown": 0.0, "max_drawdown_duration_days": 0.0}
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    peak_idx = np.maximum.accumulate(np.where(equity >= peak, np.arange(len(equity)), 0))
    under = dd < 0
    durs = [0.0]
    for i in np.nonzero(~under[1:] & under[:-1])[0] + 1:  # recovery points
        durs.append(times[i] - times[peak_idx[i - 1]])
    if under[-1]:
        durs.append(times[-1] - times[peak_idx[-1]])  # still under water at the end
    dur = max(durs) / DAY
    return {"max_drawdown": float(-dd.min()), "max_drawdown_duration_days": float(dur)}


def daily_returns(settle_ts: np.ndarray, bankroll: np.ndarray, bankroll0: float,
                  start_ts: int, end_ts: int) -> np.ndarray:
    d0, d1 = start_ts // DAY, max(end_ts - 1, start_ts) // DAY
    days = np.arange(d0, d1 + 1)
    if len(bankroll) == 0:
        return np.zeros(len(days))
    idx = np.searchsorted(settle_ts // DAY, days, side="right") - 1
    eq = np.where(idx >= 0, bankroll[np.maximum(idx, 0)], bankroll0)
    prev = np.concatenate([[bankroll0], eq[:-1]])
    return eq / prev - 1.0


def _ratio(num: float, den: float):
    return float(num / den) if den > 0 else None


def _group(keys: np.ndarray, pnl: np.ndarray, won: np.ndarray, edge: np.ndarray, names=None) -> dict:
    out = {}
    for kv in np.unique(keys):
        m = keys == kv
        label = names[int(kv)] if names else str(int(kv))
        out[label] = {"trades": int(m.sum()), "pnl": float(pnl[m].sum()),
                      "win_rate": float(won[m].mean()), "avg_edge": float(edge[m].mean())}
    return out


def brier_score(p: np.ndarray, y: np.ndarray) -> float | None:
    return float(np.mean((p - y) ** 2)) if len(p) else None


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()) if m.any() else None,
                    "freq_up": float(y[m].mean()) if m.any() else None})
    return out


def calibration(prob: np.ndarray, valid: np.ndarray, up_won: np.ndarray, bins: int = 10) -> dict:
    """Bot P(up) vs outcome over ALL valid windows (not only traded ones)."""
    m = valid[:, None] & ~np.isnan(prob)
    y = np.broadcast_to(up_won[:, None].astype(float), prob.shape)
    p_all, y_all = prob[m], y[m]
    bs = brier_score(p_all, y_all)
    per_tick = {}
    for k in range(prob.shape[1]):
        mk = m[:, k]
        b = brier_score(prob[mk, k], y[mk, k])
        per_tick[f"elapsed_{60 * (k + 1)}s"] = {"brier": b, "skill_vs_0.5": (1 - b / 0.25) if b is not None else None,
                                                 "n": int(mk.sum())}
    return {"n": int(m.sum()), "brier": bs, "brier_baseline_0.5": 0.25,
            "brier_skill_vs_0.5": (1 - bs / 0.25) if bs is not None else None,
            "per_tick": per_tick, "reliability": reliability_curve(p_all, y_all, bins),
            "base_rate_up": float(up_won[valid].mean()) if valid.any() else None}


def compute_metrics(tr: dict[str, np.ndarray], bankroll0: float, start_ts: int, end_ts: int) -> dict:
    span = max(end_ts - start_ts, 1)
    n = len(tr["pnl"])
    final = float(tr["bankroll"][-1]) if n else float(bankroll0)
    years = span / (365.25 * DAY)
    try:
        cagr = (final / bankroll0) ** (1 / years) - 1.0 if final > 0 else -1.0
    except OverflowError:
        cagr = float("inf")
    m: dict = {"trades": n, "start_bankroll": bankroll0, "final_bankroll": final,
               "total_return": final / bankroll0 - 1.0,
               "cagr": cagr,
               "span_days": span / DAY}
    if n == 0:
        m.update(win_rate=None, avg_edge=None, trades_per_day=0.0, profit_factor=None, exposure=0.0,
                 sharpe=None, sortino=None, max_drawdown=0.0, max_drawdown_duration_days=0.0)
        return m
    pnl, won = tr["pnl"], tr["won"].astype(float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = np.concatenate([[bankroll0], tr["bankroll"]])
    tm = np.concatenate([[start_ts], tr["settle_ts"]]).astype(float)
    r = daily_returns(tr["settle_ts"], tr["bankroll"], bankroll0, start_ts, end_ts)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    dsd = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
    m.update(win_rate=float(won.mean()), avg_edge=float(tr["edge"].mean()),
             avg_stake=float(tr["stake"].mean()), trades_per_day=n / (span / DAY),
             profit_factor=_ratio(wins, losses), gross_profit=float(wins), gross_loss=float(losses),
             exposure=float(min(1.0, n * 300 / span)),
             sharpe=float(r.mean() / sd * math.sqrt(365)) if sd > 0 else None,
             sortino=float(r.mean() / dsd * math.sqrt(365)) if dsd > 0 else None,
             **drawdown(eq, tm))
    ws = tr["ws"]
    year = np.asarray(ws, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    m["pnl_by_regime"] = _group(tr["regime"], pnl, won, tr["edge"], REGIME_NAMES)
    m["pnl_by_hour_utc"] = _group((ws % DAY) // 3600, pnl, won, tr["edge"])
    m["pnl_by_year"] = _group(year, pnl, won, tr["edge"])
    return m
