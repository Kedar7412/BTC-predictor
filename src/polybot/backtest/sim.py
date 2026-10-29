"""Window-level replay of the LIVE decision code over 1m klines.

Reused unchanged from the live bot: signal.estimate, edge.best_opportunity, RiskManager.decide /
mark_traded / record_pnl / reset_day (and through it risk.size_stake), PaperExecutor.buy / settle.
Only the plumbing (which prices and books are fed in) is backtest specific.

Timing: 5 minute windows aligned to epoch // 300. Window open K = open of the first minute,
window end = close of the 5th minute; UP wins iff end > K (same rule as engine._settle_due).
Decision ticks at elapsed 60, 120, 180, 240 s. At elapsed e the only known price is the close of
the minute that ended at e, so there is no look-ahead. Windows with a missing minute are skipped.
"""
from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .. import edge as edge_mod
from .. import signal as signal_mod
from ..config import Settings
from ..execution import PaperExecutor
from ..models import MarketInfo, Signal
from ..risk import RiskManager
from .data import Klines
from .market_model import BookSource, TickContext

WINDOW = 300
TICKS = 4
TRADE_COLS = ("ts", "settle_ts", "ws", "tick", "side_up", "prob", "mid", "ask", "eff", "edge",
              "stake", "shares", "won", "pnl", "bankroll", "regime")


@dataclass
class SignalSet:
    """Everything that depends only on price data and model params (cacheable across scenarios)."""
    ws: np.ndarray           # (n,) window starts, all windows on the grid
    valid: np.ndarray        # (n,) bool, all 5 minutes present
    open_px: np.ndarray      # (n,)
    end_px: np.ndarray       # (n,)
    up_won: np.ndarray       # (n,) bool
    prob: np.ndarray         # (n,4) bot P(up), nan where invalid
    sigma: np.ndarray        # (n,4)
    z: np.ndarray            # (n,4)
    price: np.ndarray        # (n,4)
    regime: np.ndarray       # (n,) 0 low,1 mid,2 high trailing-vol tercile
    grid_close: np.ndarray   # minute grid closes (nan = missing)
    t0: int
    stats: dict = field(default_factory=dict)

    def lagged_price(self, lag_minutes: int) -> np.ndarray:
        """(n,4) close `lag_minutes` before the latest known close (nan if unavailable)."""
        n = len(self.ws)
        base = np.arange(n)[:, None] * 5 + np.arange(TICKS)[None, :] - lag_minutes
        out = np.full((n, TICKS), np.nan)
        ok = base >= 0
        out[ok] = self.grid_close[base[ok]]
        return out


def _grid(k: Klines) -> tuple[int, np.ndarray, np.ndarray]:
    if len(k) == 0:
        raise ValueError("no kline data")
    t0 = int(k.ts[0]) // WINDOW * WINDOW
    n_min = int((int(k.ts[-1]) + 60 - t0) // 60)
    n_min -= n_min % 5
    idx = (k.ts - t0) // 60
    m = (idx >= 0) & (idx < n_min) & ((k.ts - t0) % 60 == 0)
    close = np.full(n_min, np.nan)
    opn = np.full(n_min, np.nan)
    close[idx[m]] = k.close[m]
    opn[idx[m]] = k.open[m]
    return t0, opn, close


_G: dict = {}


def _init_worker(opn, close, valid, t0, s):
    _G.update(opn=opn, close=close, valid=valid, t0=t0, s=s)


def _signal_chunk(args):
    w0, w1 = args
    return _signal_rows(_G["opn"], _G["close"], _G["valid"], _G["t0"], _G["s"], w0, w1)


def _signal_rows(opn, close, valid, t0, s: Settings, w0: int, w1: int):
    nh = int(s.vol_window_seconds // 60)
    m = w1 - w0
    prob = np.full((m, TICKS), np.nan)
    sigma, z, price = prob.copy(), prob.copy(), prob.copy()
    for j in range(m):
        w = w0 + j
        if not valid[w]:
            continue
        base = w * 5
        lo = max(0, base - nh)
        cl = close[lo:base + 5].tolist()
        tl = (t0 + (np.arange(lo, base + 5) + 1) * 60.0).tolist()
        off = base - lo
        k_open = float(opn[base])
        for k in range(TICKS):
            last = off + k
            hist = [(tl[i], cl[i]) for i in range(max(0, last - nh), last + 1) if cl[i] == cl[i]]
            px = cl[last]
            sig = signal_mod.estimate(px, k_open, WINDOW - 60 * (k + 1), hist, s.default_sigma_per_sqrt_s,
                                      s.min_vol_samples, s.momentum_weight)
            prob[j, k], sigma[j, k], z[j, k], price[j, k] = sig.prob_up, sig.sigma, sig.z, px
    return prob, sigma, z, price


def build_signals(k: Klines, s: Settings, workers: int = 1, progress: bool = False) -> SignalSet:
    t0, opn, close = _grid(k)
    n = len(close) // 5
    cl5 = close[:n * 5].reshape(n, 5)
    valid = ~np.isnan(cl5).any(axis=1) & ~np.isnan(opn[:n * 5:5])
    chunk = 20000
    spans = [(a, min(n, a + chunk)) for a in range(0, n, chunk)]
    t_start = time.time()
    parts = []
    if workers > 1 and len(spans) > 1:
        import multiprocessing as mp
        with mp.Pool(workers, _init_worker, (opn, close, valid, t0, s)) as pool:
            for i, r in enumerate(pool.imap(_signal_chunk, spans)):
                parts.append(r)
                if progress:
                    _bar("signals", spans[i][1], n, t_start)
    else:
        for a, b in spans:
            parts.append(_signal_rows(opn, close, valid, t0, s, a, b))
            if progress:
                _bar("signals", b, n, t_start)
    prob, sigma, z, price = (np.concatenate([p[i] for p in parts]) if parts else np.empty((0, TICKS))
                             for i in range(4))
    open_px = opn[:n * 5:5].copy()
    end_px = cl5[:, 4].copy()
    up_won = np.where(valid, end_px > open_px, False)
    regime = np.zeros(n, dtype=np.int8)
    if valid.any():
        sg = sigma[valid, 0]
        c1, c2 = np.quantile(sg, [1 / 3, 2 / 3])
        regime[:] = np.where(sigma[:, 0] <= c1, 0, np.where(sigma[:, 0] <= c2, 1, 2))
    ws = t0 + WINDOW * np.arange(n, dtype=np.int64)
    el = time.time() - t_start
    stats = {"windows_total": int(n), "windows_valid": int(valid.sum()),
             "windows_skipped_missing_minutes": int(n - valid.sum()),
             "signal_seconds": el, "signal_windows_per_sec": (n / el if el > 0 else 0.0)}
    return SignalSet(ws, valid, open_px, end_px, up_won, prob, sigma, z, price, regime, close, t0, stats)


def _bar(label: str, done: int, total: int, t_start: float) -> None:
    el = max(time.time() - t_start, 1e-9)
    rate = done / el
    eta = (total - done) / rate if rate > 0 else 0
    print(f"\r  {label}: {done}/{total} windows ({100 * done / max(total, 1):.0f}%) "
          f"{rate:,.0f}/s eta {eta:,.0f}s ", end="", file=sys.stderr, flush=True)
    if done >= total:
        print(file=sys.stderr)


def _run_sync(coro):
    """Drive a coroutine that never actually suspends (PaperExecutor.buy)."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("executor suspended; only non-blocking paper executors are supported")


_REASON_RE = re.compile(r"\s*-?\d.*$|\s*\(.*$")


@dataclass
class SimResult:
    trades: list
    final_bankroll: float
    counters: dict
    seconds: float

    def arrays(self) -> dict[str, np.ndarray]:
        cols = list(zip(*self.trades)) if self.trades else [()] * len(TRADE_COLS)
        out = {}
        for name, col in zip(TRADE_COLS, cols):
            dtype = bool if name in ("side_up", "won") else (np.int64 if name in ("ts", "settle_ts", "ws", "tick", "regime") else float)
            out[name] = np.asarray(col, dtype=dtype)
        return out


def run_sim(sigs: SignalSet, s: Settings, source: BookSource, bankroll: float | None = None,
            prefilter: bool = True, progress: bool = False) -> SimResult:
    risk = RiskManager(s)
    ex = PaperExecutor(s.bankroll if bankroll is None else bankroll)
    source.prepare(sigs.ws)
    lagp = sigs.lagged_price(source.lag_minutes) if source.lag_minutes > 0 else None
    ok = sigs.valid[:, None] & ~np.isnan(sigs.prob)
    if prefilter:
        # exact: if neither side reaches min_conviction, best_opportunity returns None, no state touched
        conv = np.maximum(sigs.prob, 1.0 - sigs.prob)
        cand = ok & (conv >= s.min_conviction)
    else:
        cand = ok
    rows = np.nonzero(cand.any(axis=1))[0]
    counters = {"windows_total": int(len(sigs.ws)), "windows_valid": int(sigs.valid.sum()),
                "ticks_valid": int(ok.sum()), "ticks_evaluated": int(cand.sum()),
                "ticks_prefiltered": int(ok.sum() - cand.sum()), "no_book": 0, "trades": 0,
                "skip_reasons": {}}
    reasons = counters["skip_reasons"]
    trades: list = []
    cur_day = None
    t_start = time.time()
    total = len(rows)
    for ri, i in enumerate(rows.tolist()):
        ws = int(sigs.ws[i])
        day = ws // 86400
        if day != cur_day:
            if cur_day is not None:
                risk.reset_day()
            cur_day = day
        risk.traded_windows.clear()
        end_ts = ws + WINDOW
        for k in range(TICKS):
            if not cand[i, k]:
                continue
            now = ws + 60 * (k + 1)
            p = float(sigs.prob[i, k])
            sg = float(sigs.sigma[i, k])
            ctx = TickContext(i, ws, k, now, end_ts, float(sigs.open_px[i]), float(sigs.price[i, k]),
                              float(lagp[i, k]) if lagp is not None else math.nan, p, sg, float(end_ts - now))
            bk = source.books(ctx)
            if bk is None:
                counters["no_book"] += 1
                continue
            up, down = bk
            sig = Signal(prob_up=p, sigma=sg, z=float(sigs.z[i, k]), conviction_up=p, conviction_down=1 - p)
            opp, reason = edge_mod.best_opportunity(sig, up, down, s)
            if opp is None:
                key = _REASON_RE.sub("", reason)
                reasons[key] = reasons.get(key, 0) + 1
                continue
            book = up if opp.side == "UP" else down
            d = risk.decide(now, ws, end_ts, book, opp, ex.bankroll, 0)
            if not d.trade:
                key = _REASON_RE.sub("", d.reason)
                reasons[key] = reasons.get(key, 0) + 1
                continue
            risk.mark_traded(now, ws)
            market = MarketInfo(f"btc-updown-5m-{ws}", ws, end_ts, "UP", "DOWN")
            pos = _run_sync(ex.buy(market, d, now))
            if pos is None:
                reasons["order not filled"] = reasons.get("order not filled", 0) + 1
                continue
            up_won = bool(source.outcome(ctx, bool(sigs.up_won[i])))
            pnl = ex.settle(pos, up_won)
            risk.record_pnl(pnl)
            side_up = opp.side == "UP"
            trades.append((now, end_ts, ws, k, side_up, opp.prob, (book.best_bid + book.best_ask) / 2,
                           opp.ask, opp.eff_price, opp.edge, pos.cost, pos.shares,
                           side_up == up_won, pnl, ex.bankroll, int(sigs.regime[i])))
            break  # one position per window (decide would reject later ticks anyway)
        if progress and (ri % 20000 == 0 or ri == total - 1):
            _bar("simulate", ri + 1, total, t_start)
    counters["trades"] = len(trades)
    return SimResult(trades, ex.bankroll, counters, time.time() - t_start)
