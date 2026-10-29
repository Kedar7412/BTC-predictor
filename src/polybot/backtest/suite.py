"""`--suite quick|full`: run every hard-test layer and write the report.

Stages: signals (cached) -> scenario matrix -> controls -> reference run -> walk-forward + sweep ->
Monte Carlo -> stress -> report. See README "Backtesting for 5 years" for expected runtimes.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

from ..config import Settings
from . import data as data_mod
from . import montecarlo, report, scenarios, stress, walkforward
from .jobs import get_signals, run_job
from .metrics import calibration

STAGES = 8


def _stage(i: int, name: str) -> None:
    print(f"[{i}/{STAGES}] {name}", file=sys.stderr, flush=True)


def run_suite(a, s: Settings, k: data_mod.Klines, source=None) -> dict:
    t_all = time.time()
    full = a.suite == "full"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _stage(1, "signals (cached in --data-dir as .npz, reuse with --resume)")
    sigs = get_signals(k, s, a.data_dir, a.resume, a.workers, progress=True)
    n_rows = len(sigs.ws)
    start_ts, end_ts = int(sigs.ws[0]), int(sigs.ws[-1]) + 300
    _stage(2, "scenario matrix and cost sweep")
    sc = scenarios.run_scenarios(sigs, s, a.workers, full, a.seed, a.half_spread)
    _stage(3, "no-information controls")
    rw_years = 1.0 if full else 0.3
    rw = data_mod.synthetic(rw_years, a.seed + 1)
    rw_sigs = get_signals(rw, s, None, False)
    ctl = scenarios.run_controls(sigs, s, rw_sigs, a.seed, a.half_spread)
    _stage(4, "reference run (trades, equity, calibration)")
    job = {"market": {**scenarios.market_kwargs(scenarios.REFERENCE), "half_spread": a.half_spread, "seed": a.seed},
           "keep": True}
    if source is not None:
        r = run_job(job, sigs, s, source=source)
        market_desc = source.describe()
    else:
        r = run_job(job, sigs, s)
        market_desc = {"scenario": scenarios.REFERENCE, **job["market"]}
    tr = r.pop("trades")
    cal = calibration(sigs.prob, sigs.valid, sigs.up_won)
    baseline = {"metrics": r["metrics"], "counters": r["counters"], "market": market_desc,
                "warning": scenarios.implausible(r["metrics"]), "calibration": cal}

    mom_cache: dict = {float(s.momentum_weight): sigs}

    def get_sigs(mw: float):
        mw = float(mw)
        if mw not in mom_cache:
            mom_cache[mw] = get_signals(k, dataclasses.replace(s, momentum_weight=mw), a.data_dir, a.resume,
                                        a.workers)
        return mom_cache[mw]

    _stage(5, "walk-forward and parameter sweep (selection on train segments only)")
    wf = walkforward.run_walkforward(get_sigs, s, n_rows, a.workers, full, seed=a.seed)
    _stage(6, "Monte Carlo")
    mc = montecarlo.run_montecarlo(tr["pnl"], tr["settle_ts"] // 86400, s.bankroll,
                                   n_paths=a.mc_paths or (2000 if full else 300), seed=a.seed,
                                   max_daily_loss=s.max_daily_loss)
    _stage(7, "stress tests")
    st = stress.run_stress(k, s, a.workers, a.stress_days if a.stress_days is not None else (365 if full else None),
                           a.seed)
    _stage(8, "report")
    first = dt.datetime.fromtimestamp(start_ts, dt.timezone.utc).date()
    last = dt.datetime.fromtimestamp(end_ts, dt.timezone.utc).date()
    R = {"meta": {"suite": a.suite, "period": f"{first} to {last}", "workers": a.workers, "out": str(out),
                  "wall_seconds": time.time() - t_all, "seed": a.seed,
                  "signal_stats": sigs.stats},
         "data": k.quality, "windows": sigs.stats, "scenarios": sc, "controls": ctl, "baseline": baseline,
         "walkforward": wf, "montecarlo": mc, "stress": st,
         "settings": {f.name: getattr(s, f.name) for f in dataclasses.fields(s)}}
    if source is not None:
        R["books"] = source.describe()
    eq = np.concatenate([[s.bankroll], tr["bankroll"]])
    tm = np.concatenate([[start_ts], tr["settle_ts"]])
    path = report.write_report(out, R, eq, tm)
    from .cli import write_outputs
    write_outputs(out, tr, {"data": {"first": start_ts}, "metrics": baseline["metrics"]})
    print(f"report: {path}", file=sys.stderr)
    return R
