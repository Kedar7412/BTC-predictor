"""Scenario matrix: market efficiency x cost grid, sensitivity tables, break-even, and controls.

Every number here depends on the ASSUMED market (see market_model.py). The point of this module is
to show how the result changes with that assumption, not to produce one headline number.

Scenario order matters for the report: 'efficient' (no information edge, costs only) and a
moderately noisy market come first; the exploitable stale scenarios come last and are flagged.
"""
from __future__ import annotations

import numpy as np

from ..config import Settings
from .jobs import pmap, run_job
from .market_model import AssumedMarket, NoInfoControl
from .sim import SignalSet

MODERATE_NOISE = 0.3
# (name, AssumedMarket kwargs). Order = report order.
SCENARIO_LIST = [
    ("efficient", dict(scenario="efficient")),
    (f"noisy_sd{MODERATE_NOISE}", dict(scenario="noisy", noise_sd=MODERATE_NOISE)),
    ("noisy_sd0.1", dict(scenario="noisy", noise_sd=0.1)),
    ("noisy_sd0.2", dict(scenario="noisy", noise_sd=0.2)),
    ("noisy_sd0.5", dict(scenario="noisy", noise_sd=0.5)),
    ("lagged_1m", dict(scenario="lagged", lag_minutes=1)),
    ("lagged_2m", dict(scenario="lagged", lag_minutes=2)),
    ("lagged_3m", dict(scenario="lagged", lag_minutes=3)),
    ("stale_noisy_lag2_sd0.15", dict(scenario="stale_noisy", lag_minutes=2, noise_sd=0.15)),
]
REFERENCE = f"noisy_sd{MODERATE_NOISE}"  # reference market for sweeps and stress tests
SPREADS = (0.005, 0.01, 0.02)
FEES = (0.0, 0.02, 0.04)
COST_SCENARIOS = ("efficient", REFERENCE, "stale_noisy_lag2_sd0.15")

# Plausibility limits: beyond these the result is an artifact of the assumed market, not a strategy.
MAX_PLAUSIBLE_SHARPE = 5.0
MAX_PLAUSIBLE_MULTIPLE = 100.0


def market_kwargs(name: str) -> dict:
    return dict(dict(SCENARIO_LIST)[name])


def implausible(m: dict) -> str | None:
    """Warning text when a result is too good to be a strategy result, else None."""
    sh = m.get("sharpe")
    mult = m["final_bankroll"] / m["start_bankroll"] if m.get("start_bankroll") else 0.0
    why = []
    if sh is not None and sh > MAX_PLAUSIBLE_SHARPE:
        why.append(f"Sharpe {sh:.1f} > {MAX_PLAUSIBLE_SHARPE:g}")
    if mult > MAX_PLAUSIBLE_MULTIPLE:
        why.append(f"{mult:,.0f}x return > {MAX_PLAUSIBLE_MULTIPLE:g}x")
    if not why:
        return None
    return ("WARNING implausible: " + ", ".join(why) + ". This is an artifact of the assumed market "
            "staleness/noise (a stale market is exploitable by construction), not evidence of an edge.")


def _slim(r: dict) -> dict:
    m = r["metrics"]
    return {"metrics": m, "counters": {k: v for k, v in r["counters"].items() if k != "skip_reasons"},
            "skip_reasons": r["counters"].get("skip_reasons", {}), "warning": implausible(m)}


def run_scenarios(sigs: SignalSet, s: Settings, workers: int = 1, full: bool = False, seed: int = 0,
                  half_spread: float = 0.01, progress: bool = True) -> dict:
    base_mk = dict(half_spread=half_spread, seed=seed)
    names = [n for n, _ in SCENARIO_LIST]
    jobs = [{"market": {**base_mk, **market_kwargs(n)}} for n in names]
    spreads = SPREADS if full else (0.01, 0.02)
    fees = FEES if full else (0.0, 0.04)
    cost_names = COST_SCENARIOS if full else ("efficient", REFERENCE)
    cost_keys = []
    for sc in cost_names:
        for hs in spreads:
            for fee in fees:
                cost_keys.append((sc, hs, fee))
                jobs.append({"market": {**base_mk, **market_kwargs(sc), "half_spread": hs},
                             "settings": {"fee_rate": fee}})
    res = pmap(jobs, sigs, s, workers, "scenarios" if progress else "")
    baseline = {n: _slim(r) for n, r in zip(names, res[:len(names)])}
    grid: dict = {}
    for (sc, hs, fee), r in zip(cost_keys, res[len(names):]):
        grid.setdefault(sc, {})[f"spread={hs}|fee={fee}"] = {
            "half_spread": hs, "fee_rate": fee, "total_return": r["metrics"]["total_return"],
            "trades": r["metrics"]["trades"], "sharpe": r["metrics"]["sharpe"],
            "warning": implausible(r["metrics"])}
    return {"baseline": baseline, "cost_grid": grid, "sensitivity": sensitivity(baseline),
            "break_even": break_even(baseline, grid)}


def sensitivity(baseline: dict) -> dict:
    """PnL vs market noise and vs staleness, read off the baseline results."""
    def row(n):
        m = baseline[n]["metrics"]
        return {"scenario": n, "total_return": m["total_return"], "trades": m["trades"], "sharpe": m["sharpe"]}
    noise = sorted((market_kwargs(n)["noise_sd"], n) for n in baseline if n.startswith("noisy_sd"))
    lag = sorted((market_kwargs(n)["lag_minutes"], n) for n in baseline if n.startswith("lagged_"))
    return {"vs_noise_sd": [{"noise_sd": v, **row(n)} for v, n in noise],
            "vs_lag_minutes": [{"lag_minutes": v, **row(n)} for v, n in lag]}


def break_even(baseline: dict, grid: dict) -> dict:
    """Largest market efficiency / cost at which total return stays above zero."""
    out: dict = {}
    pos_noise = [r["noise_sd"] for r in sensitivity(baseline)["vs_noise_sd"] if r["total_return"] > 0]
    out["min_noise_sd_with_positive_return"] = min(pos_noise) if pos_noise else None
    out["note_noise"] = ("Below this market noise the assumed market is too efficient for the bot to "
                         "profit. Real markets may be more or less efficient than any of these.")
    pos_lag = [r["lag_minutes"] for r in sensitivity(baseline)["vs_lag_minutes"] if r["total_return"] > 0]
    out["min_lag_minutes_with_positive_return"] = min(pos_lag) if pos_lag else None
    costs: dict = {}
    for sc, cells in grid.items():
        pos = [c for c in cells.values() if c["total_return"] > 0]
        costs[sc] = {
            "max_half_spread_positive": max((c["half_spread"] for c in pos), default=None),
            "max_fee_rate_positive": max((c["fee_rate"] for c in pos), default=None),
            "cells_positive": f"{len(pos)}/{len(cells)}"}
    out["costs"] = costs
    return out


def monotonic_in_costs(grid_cells: dict, tol: float = 1e-9) -> bool:
    """Sanity check: on the same signals, a higher fee at the same spread never improves PnL."""
    by_spread: dict = {}
    for c in grid_cells.values():
        by_spread.setdefault(c["half_spread"], []).append((c["fee_rate"], c["total_return"]))
    for lst in by_spread.values():
        lst.sort()
        # compounding with sizing can reorder tiny differences; require no material improvement
        for (_, a), (_, b) in zip(lst, lst[1:]):
            if b > a + tol and not (a == 0 and b == 0):
                return False
    return True


# ---------------------------------------------------------------- no-information controls

def run_controls(sigs: SignalSet, s: Settings, rw_sigs: SignalSet | None = None, seed: int = 0,
                 half_spread: float = 0.01) -> dict:
    """Show that costs alone lose money, so the harness is not generating free money.

    efficient        market mid == bot probability, gates as configured. The bot sees no edge, so it
                     should not trade (zero PnL): the gates and cost model block it.
    no_info_noisy_*  the bot trades against a noisy market whose mid is the TRUE probability
                     (outcomes drawn from the mid, see NoInfoControl). The bot has edges on paper only,
                     so it should LOSE money: spread, fee and slippage with nothing to offset them.
    The same runs are repeated on an independent pure random-walk price series.
    """
    out: dict = {}
    sets = [("real_data", sigs)] + ([("random_walk", rw_sigs)] if rw_sigs is not None else [])
    for tag, sg in sets:
        runs = [("efficient", AssumedMarket("efficient", half_spread=half_spread, seed=seed))]
        runs += [(f"no_info_noisy_sd{sd}", NoInfoControl("noisy", noise_sd=sd, half_spread=half_spread, seed=seed))
                 for sd in (0.3, 0.5)]
        for name, src in runs:
            m = run_job({}, sg, s, source=src)["metrics"]
            out[f"{tag}:{name}"] = {"total_return": m["total_return"], "trades": m["trades"],
                                    "final_bankroll": m["final_bankroll"], "win_rate": m["win_rate"],
                                    "sharpe": m["sharpe"]}
    noinfo = [v for k, v in out.items() if ":no_info" in k]
    traded = [v for v in noinfo if v["trades"] > 0]
    if not traded:
        verdict = ("INCONCLUSIVE: the no-information controls placed no trades, so they show no costs. "
                   "Try a longer period.")
    elif all(v["total_return"] < 0 for v in traded):
        verdict = "PASS: with no information edge, costs alone produce negative PnL. Costs are charged (this validates cost accounting only, not absence of look-ahead)."
    else:
        verdict = ("CHECK: a no-information control made money. Cost accounting may be wrong or the "
                   "sample is too small. Investigate before trusting any other number.")
    out["verdict"] = verdict
    return out
