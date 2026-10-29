"""Shared plumbing for the hard-test suite: signal slicing, cached signals, and parallel jobs.

A "job" is a plain dict so it can be pickled to worker processes:
  settings : dict of Settings overrides (fee_rate, min_edge, ...)
  market   : kwargs for AssumedMarket
  seg      : (row_start, row_end) slice of the SignalSet rows, or None for everything
  keep     : True to also return the trade arrays
The (large) SignalSet is shipped to each worker once through the pool initializer.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..config import Settings
from .data import Klines
from .market_model import AssumedMarket, BookSource
from .metrics import compute_metrics
from .sim import SignalSet, build_signals, run_sim

_SIGNAL_FIELDS = ("ws", "valid", "open_px", "end_px", "up_won", "prob", "sigma", "z", "price", "regime",
                  "grid_close")
SIGNAL_PARAMS = ("momentum_weight", "vol_window_seconds", "default_sigma_per_sqrt_s", "min_vol_samples")


def slice_sigs(sigs: SignalSet, i0: int, i1: int) -> SignalSet:
    """Rows [i0, i1) as a standalone SignalSet (lagged prices stay consistent)."""
    kw = {f: getattr(sigs, f)[i0:i1] for f in _SIGNAL_FIELDS if f != "grid_close"}
    kw["grid_close"] = sigs.grid_close[i0 * 5:i1 * 5]
    return SignalSet(t0=int(sigs.t0 + i0 * 300), stats={}, **kw)


def make_segments(n_rows: int, seg_rows: int) -> list[tuple[int, int]]:
    """Consecutive segments of seg_rows windows; the remainder is merged into the last one."""
    n = max(1, n_rows // max(seg_rows, 1))
    cuts = [i * seg_rows for i in range(n)] + [n_rows]
    return [(cuts[i], cuts[i + 1]) for i in range(n)]


# ---------------------------------------------------------------- signal cache

def _code_hash() -> str:
    """Hash of the source that determines signal values, so editing it invalidates cached signals."""
    from .. import signal as live_signal
    from . import sim
    h = hashlib.sha256()
    for mod in (live_signal, sim):
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()


def signals_key(k: Klines, s: Settings) -> str:
    h = hashlib.sha256()
    h.update(_code_hash().encode())
    h.update(json.dumps({p: getattr(s, p) for p in SIGNAL_PARAMS}, sort_keys=True).encode())
    h.update(np.int64([len(k), int(k.ts[0]), int(k.ts[-1])]).tobytes())
    h.update(np.float64([float(k.close.sum()), float(k.open.sum())]).tobytes())
    return h.hexdigest()[:16]


def save_signals(sigs: SignalSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f: getattr(sigs, f) for f in _SIGNAL_FIELDS}
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, t0=np.int64(sigs.t0), stats=np.array(json.dumps(sigs.stats)), **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_signals(path: Path) -> SignalSet:
    with np.load(path, allow_pickle=False) as z:
        kw = {f: z[f] for f in _SIGNAL_FIELDS}
        return SignalSet(t0=int(z["t0"]), stats=json.loads(str(z["stats"])), **kw)


def get_signals(k: Klines, s: Settings, cache_dir: str | Path | None, resume: bool, workers: int = 1,
                progress: bool = False) -> SignalSet:
    """build_signals with an .npz cache in cache_dir (loaded only when resume=True, always written)."""
    path = Path(cache_dir) / f"signals_{signals_key(k, s)}.npz" if cache_dir else None
    if path is not None and resume and path.exists():
        try:
            sigs = load_signals(path)
            print(f"  reusing cached signals {path}", file=sys.stderr)
            return sigs
        except Exception as e:  # corrupt or truncated cache: recompute and overwrite
            print(f"  ignoring unreadable signal cache {path} ({type(e).__name__}); recomputing", file=sys.stderr)
    sigs = build_signals(k, s, workers=workers, progress=progress)
    if path is not None:
        save_signals(sigs, path)
    return sigs


# ---------------------------------------------------------------- jobs

_W: dict = {}


def _init(sigs, s):
    _W["sigs"], _W["s"] = sigs, s


def run_job(job: dict, sigs: SignalSet | None = None, s: Settings | None = None,
            source: BookSource | None = None) -> dict:
    sigs = sigs if sigs is not None else _W["sigs"]
    s = s if s is not None else _W["s"]
    if job.get("settings"):
        s = dataclasses.replace(s, **job["settings"])
    seg = job.get("seg")
    sub = slice_sigs(sigs, *seg) if seg else sigs
    if len(sub.ws) == 0:
        return {"metrics": compute_metrics(_empty(), s.bankroll, 0, 1), "counters": {}, "pnl": 0.0}
    src = source or AssumedMarket(**job["market"])
    res = run_sim(sub, s, src)
    tr = res.arrays()
    m = compute_metrics(tr, s.bankroll, int(sub.ws[0]), int(sub.ws[-1]) + 300)
    out = {"metrics": m, "counters": res.counters, "pnl": res.final_bankroll - s.bankroll,
           "log_return": float(np.log(max(res.final_bankroll, 1e-9) / s.bankroll)),
           "stake_sum": float(tr["stake"].sum()) if len(tr["stake"]) else 0.0}
    if job.get("keep"):
        out["trades"] = tr
    if job.get("rets"):
        out["rets"] = tr["pnl"] / np.maximum(tr["stake"], 1e-12)
    return out


def _empty() -> dict:
    z = np.empty(0)
    return {"pnl": z, "bankroll": z, "settle_ts": z, "won": z, "edge": z, "stake": z, "ws": z,
            "regime": z, "ts": z}


def pmap(jobs: list[dict], sigs: SignalSet, s: Settings, workers: int = 1, label: str = "") -> list[dict]:
    """Run jobs, in worker processes when workers > 1. Order of results matches jobs."""
    t0 = time.time()
    out: list[dict] = []

    def tick(i):
        if label:
            el = time.time() - t0
            print(f"\r  {label}: {i}/{len(jobs)} ({el:,.0f}s) ", end="", file=sys.stderr, flush=True)

    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(min(workers, len(jobs)), _init, (sigs, s)) as pool:
            for i, r in enumerate(pool.imap(run_job, jobs)):
                out.append(r)
                tick(i + 1)
    else:
        for i, j in enumerate(jobs):
            out.append(run_job(j, sigs, s))
            tick(i + 1)
    if label:
        print(file=sys.stderr)
    return out
